#!/usr/bin/env python3
"""Train a FUDGE-style prefix scorer directly from Amazon Polarity."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

CD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = PROJECT_ROOT / "dataset" / "RAD_train" / "amazon_polarity"
MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-small"
OUTPUT_DIR = PROJECT_ROOT / "models" / "cd_prefix_scorer"

NUM_TRAIN_SAMPLES: int | None = 100_000
NUM_EVAL_SAMPLES: int = 5_000
MAX_LENGTH: int = 256
BATCH_SIZE: int = 8
GRADIENT_ACCUMULATION: int = 4
EPOCHS: int = 1
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.03
MAX_GRAD_NORM: float = 1.0
NUM_WORKERS: int = 2
SEED: int = 42
SAVE_BEST_ONLY: bool = True

if str(CD_ROOT) not in sys.path:
    sys.path.insert(0, str(CD_ROOT))

from models.prefix_scorer import PrefixScorer


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    shuffled = dataset.shuffle(seed=seed)
    if size is None or size >= len(shuffled):
        return shuffled
    return shuffled.select(range(size))


def normalize_text(sample: dict[str, Any]) -> dict[str, Any]:
    title = str(sample.get("title", "")).strip()
    content = str(sample.get("content", "")).strip()
    text = f"{title}\n\n{content}" if title and content else (title or content)
    return {"text": text, "labels": float(sample["label"])}


class PrefixClassificationCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in samples]
        labels = torch.tensor(
            [float(sample["labels"]) for sample in samples], dtype=torch.float
        )

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        attention_mask = encoded["attention_mask"]
        prefix_mask = attention_mask.bool()

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
            "prefix_mask": prefix_mask,
            "labels": labels,
        }


@torch.inference_mode()
def evaluate(model: PrefixScorer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_last_token = 0
    total_examples = 0

    for batch in tqdm(loader, desc="Validation", unit="batch", leave=False):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        output = model(**batch)
        total_loss += float(output["loss"].item())
        total_batches += 1

        values = output["values"]
        attention_mask = batch["attention_mask"]
        last_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        last_scores = values[
            torch.arange(values.size(0), device=device), last_indices
        ]

        predictions = (last_scores >= 0.5).long()
        labels = batch["labels"].long()
        correct_last_token += int((predictions == labels).sum().item())
        total_examples += labels.numel()

    model.train()

    return {
        "eval_loss": total_loss / max(total_batches, 1),
        "eval_accuracy": correct_last_token / max(total_examples, 1),
    }


def save_training_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def main() -> None:
    set_all_seeds(SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Amazon Polarity dataset not found: {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"GPT-2 Small model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CD-FUDGE Prefix Scorer Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Backbone: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Train samples: {NUM_TRAIN_SAMPLES}")
    print(f"Eval samples: {NUM_EVAL_SAMPLES}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 70)

    raw_dataset = load_from_disk(str(DATASET_PATH))
    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict with train and test splits.")
    if "train" not in raw_dataset or "test" not in raw_dataset:
        raise KeyError("Amazon Polarity must contain train and test splits.")

    train_dataset = select_subset(raw_dataset["train"], NUM_TRAIN_SAMPLES, SEED)
    eval_dataset = select_subset(raw_dataset["test"], NUM_EVAL_SAMPLES, SEED + 1)

    train_dataset = train_dataset.map(
        normalize_text,
        remove_columns=train_dataset.column_names,
        desc="Preparing train data",
    )
    eval_dataset = eval_dataset.map(
        normalize_text,
        remove_columns=eval_dataset.column_names,
        desc="Preparing eval data",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_LENGTH

    collator = PrefixClassificationCollator(tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = PrefixScorer(str(MODEL_PATH)).to(device)
    model.backbone.config.use_cache = False

    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION
    )
    total_optimizer_steps = optimizer_steps_per_epoch * EPOCHS
    warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_eval_loss = float("inf")
    global_optimizer_step = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
        )

        for batch_index, batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch)
                raw_loss = output["loss"]
                loss = raw_loss / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.item())
            seen_batches += 1

            should_step = (
                batch_index % GRADIENT_ACCUMULATION == 0
                or batch_index == len(train_loader)
            )

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_optimizer_step += 1

            progress.set_postfix(
                loss=f"{raw_loss.item():.4f}",
                avg=f"{running_loss / seen_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        metrics = evaluate(model, eval_loader, device)
        train_loss = running_loss / max(seen_batches, 1)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, "
            f"eval_loss={metrics['eval_loss']:.6f}, "
            f"eval_accuracy={metrics['eval_accuracy']:.4f}"
        )

        improved = metrics["eval_loss"] < best_eval_loss
        if improved:
            best_eval_loss = metrics["eval_loss"]
            model.save_pretrained(OUTPUT_DIR, tokenizer=tokenizer)
            save_training_metadata(
                OUTPUT_DIR,
                {
                    "dataset_path": str(DATASET_PATH),
                    "model_path": str(MODEL_PATH),
                    "num_train_samples": len(train_dataset),
                    "num_eval_samples": len(eval_dataset),
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation": GRADIENT_ACCUMULATION,
                    "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "seed": SEED,
                    "best_epoch": epoch + 1,
                    "best_eval_loss": best_eval_loss,
                    "eval_accuracy": metrics["eval_accuracy"],
                    "global_optimizer_step": global_optimizer_step,
                },
            )
            print(f"Saved best checkpoint to: {OUTPUT_DIR}")
        elif not SAVE_BEST_ONLY:
            epoch_dir = OUTPUT_DIR / f"epoch-{epoch + 1}"
            model.save_pretrained(epoch_dir, tokenizer=tokenizer)

    print("=" * 70)
    print("Training finished")
    print(f"Best eval loss: {best_eval_loss:.6f}")
    print(f"Saved model: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()