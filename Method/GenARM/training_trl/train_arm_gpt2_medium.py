from __future__ import annotations

import gc
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import DPOConfig

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-medium"

HH_DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "GenARM"
    / "full-hh-rlhf"
    / "data"
)

ADAPTER_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "genarm-gpt2-medium-hh-adapter"
)

MERGED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "genarm-gpt2-medium-hh"
)


# ============================================================
# DATA CONFIG — chỉnh ở đây
# ============================================================

# True: test nhanh trên một phần nhỏ.
# False: train toàn bộ HH-RLHF.
SANITY_CHECK = True

SANITY_TRAIN_SAMPLES = 1_000
SANITY_EVAL_SAMPLES = 500

# None nghĩa là dùng toàn bộ split.
NUM_TRAIN_SAMPLES: Optional[int] = None
NUM_EVAL_SAMPLES: Optional[int] = 2_000

MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256

SEED = 42
NUM_PROC = 1


# ============================================================
# TRAINING CONFIG — chỉnh ở đây
# ============================================================

EPOCHS = 1
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03

BETA = 0.1
GAMMA = 0.0
LENGTH_NORMALIZATION = False
LOSS_TYPE = "sigmoid"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LOAD_IN_4BIT = True
USE_FP16 = True

LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2


# ============================================================
# LOCAL IMPORT
# ============================================================

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from arm_trainer import ARMTrainer


@dataclass
class ARMConfig(DPOConfig):
    gamma: float = field(
        default=0.0,
        metadata={"help": "Target reward margin."},
    )
    length_normalization: bool = field(
        default=False,
        metadata={"help": "Normalize sequence log-probability by length."},
    )


def find_single_parquet(prefix: str) -> Path:
    matches = sorted(HH_DATA_DIR.glob(f"{prefix}-*.parquet"))

    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy file {prefix}-*.parquet trong {HH_DATA_DIR}"
        )

    if len(matches) > 1:
        print(
            f"Cảnh báo: tìm thấy {len(matches)} file cho split {prefix}; "
            f"sẽ dùng tất cả."
        )

    return matches[0]


def load_hh_rlhf() -> DatasetDict:
    train_files = sorted(HH_DATA_DIR.glob("train-*.parquet"))
    test_files = sorted(HH_DATA_DIR.glob("test-*.parquet"))

    if not train_files:
        raise FileNotFoundError(
            f"Không tìm thấy train-*.parquet trong {HH_DATA_DIR}"
        )

    if not test_files:
        raise FileNotFoundError(
            f"Không tìm thấy test-*.parquet trong {HH_DATA_DIR}"
        )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": [str(path) for path in train_files],
            "test": [str(path) for path in test_files],
        },
    )

    required_columns = {"prompt", "chosen", "rejected"}

    for split_name in ("train", "test"):
        missing = required_columns - set(dataset[split_name].column_names)

        if missing:
            raise ValueError(
                f"Split {split_name} thiếu các cột: {sorted(missing)}"
            )

    return dataset


def select_samples(
    dataset: Dataset,
    requested_size: Optional[int],
    seed: int,
) -> Dataset:
    if requested_size is None or requested_size >= len(dataset):
        return dataset.shuffle(seed=seed)

    return dataset.shuffle(seed=seed).select(range(requested_size))


def remove_empty_rows(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda row: (
            bool(str(row["prompt"]).strip())
            and bool(str(row["chosen"]).strip())
            and bool(str(row["rejected"]).strip())
        ),
        num_proc=NUM_PROC,
        desc="Removing empty preference rows",
    )


def main() -> None:
    set_seed(SEED)
    random.seed(SEED)

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy GPT-2 Medium: {BASE_MODEL_PATH}"
        )

    if not HH_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy HH-RLHF local: {HH_DATA_DIR}"
        )

    print("=" * 72)
    print("Train GPT-2 Medium ARM on HH-RLHF")
    print("=" * 72)
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Dataset: {HH_DATA_DIR}")
    print(f"Sanity check: {SANITY_CHECK}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Adapter output: {ADAPTER_OUTPUT_DIR}")
    print(f"Merged output: {MERGED_OUTPUT_DIR}")
    print("=" * 72)

    raw_dataset = load_hh_rlhf()

    if SANITY_CHECK:
        train_size = min(SANITY_TRAIN_SAMPLES, len(raw_dataset["train"]))
        eval_size = min(SANITY_EVAL_SAMPLES, len(raw_dataset["test"]))
    else:
        train_size = NUM_TRAIN_SAMPLES
        eval_size = NUM_EVAL_SAMPLES

    train_dataset = select_samples(
        raw_dataset["train"],
        train_size,
        SEED,
    )

    eval_dataset = select_samples(
        raw_dataset["test"],
        eval_size,
        SEED + 1,
    )

    train_dataset = remove_empty_rows(train_dataset)
    eval_dataset = remove_empty_rows(eval_dataset)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quantization_config = None

    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        quantization_config=quantization_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "c_attn",
            "c_proj",
            "c_fc",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = ARMConfig(
        output_dir=str(ADAPTER_OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        beta=BETA,
        gamma=GAMMA,
        length_normalization=LENGTH_NORMALIZATION,
        loss_type=LOSS_TYPE,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=USE_FP16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )

    trainer = ARMTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

    print(f"Đã lưu LoRA adapter: {ADAPTER_OUTPUT_DIR}")

    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Đang merge LoRA adapter vào GPT-2 Medium...")

    merge_base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    merged_model = PeftModel.from_pretrained(
        merge_base,
        str(ADAPTER_OUTPUT_DIR),
    ).merge_and_unload()

    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(MERGED_OUTPUT_DIR),
        safe_serialization=True,
    )

    tokenizer.save_pretrained(str(MERGED_OUTPUT_DIR))

    print("=" * 72)
    print("Training completed")
    print(f"Merged ARM model: {MERGED_OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()