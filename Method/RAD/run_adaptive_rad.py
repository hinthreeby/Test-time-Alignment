#!/usr/bin/env python3
"""
Adaptive RAD dựa trực tiếp trên file generate RAD gốc.

Yêu cầu:
RewardAugmentedLogitsProcessor phải hỗ trợ:
    beta_strategy
    beta_params
    max_new_tokens
    _beta_history
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    set_seed,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "rad_benchmark"
    / "negative_prompts.jsonl"
)

LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"

RM_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
RM_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
RM_CHECKPOINT_PATH = RM_TOKENIZER_PATH / "pytorch_model.bin"

OUTPUT_DIR = PROJECT_ROOT / "results"

STRATEGIES = [
    "fixed",
    "confidence",
    "reward",
    "hybrid",
    "length",
]

NUM_PROMPTS = 300
MAX_NEW_TOKENS = 64
TOP_K = 20
BASE_BETA = 50.0
SEED = 1


# ============================================================
# PROJECT IMPORTS
# ============================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.reward_modeling.reward_model import GPT2RewardModel
from Method.RAD.utils.logits_processor import RewardAugmentedLogitsProcessor


def load_dataset(path: Path) -> list[dict[str, Any]]:
    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            prompt = sample.get("prompt", "")

            if isinstance(prompt, dict):
                prompt = prompt.get("text", "")

            prompt = str(prompt).strip()

            if prompt:
                sample["_prompt_text"] = prompt
                samples.append(sample)

            if len(samples) >= NUM_PROMPTS:
                break

    return samples


def load_models(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(LM_PATH),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    lm = AutoModelForCausalLM.from_pretrained(
        str(LM_PATH),
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)

    lm.config.pad_token_id = tokenizer.pad_token_id
    lm.config.use_cache = True
    lm.eval()

    rm_tokenizer = AutoTokenizer.from_pretrained(
        str(RM_TOKENIZER_PATH),
        local_files_only=True,
    )

    if rm_tokenizer.pad_token_id is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_tokenizer.padding_side = "right"
    rm_tokenizer.model_max_length = 256

    rm = GPT2RewardModel(
        reward_model_name=str(RM_BASE_PATH),
        out_features=1,
        loss_fn="cumulative_mse",
    )

    state_dict = torch.load(
        str(RM_CHECKPOINT_PATH),
        map_location="cpu",
    )

    rm.load_state_dict(state_dict, strict=True)
    rm.eval().to(device)

    max_length = int(
        getattr(
            lm.config,
            "n_positions",
            getattr(lm.config, "max_position_embeddings", 1024),
        )
    )

    return lm, tokenizer, rm, rm_tokenizer, max_length


def get_beta_params(strategy: str) -> dict:
    beta = BASE_BETA

    if strategy == "fixed":
        return {}

    if strategy == "confidence":
        return {
            "beta_min": beta * 0.1,
            "beta_max": beta * 3.0,
            "smoothing": 0.3,
            "alpha": 1.0,
        }

    if strategy == "reward":
        return {
            "beta_min": beta * 0.1,
            "beta_max": beta * 4.0,
            "eta": 0.15,
            "gamma": 0.85,
            "warmup_steps": 1,
        }

    if strategy == "hybrid":
        return {
            "beta_min": beta * 0.1,
            "beta_max": beta * 3.0,
            "lam": 0.5,
            "eta": 0.15,
            "gamma": 0.85,
        }

    if strategy == "length":
        return {
            "beta_start": beta * 0.1,
            "beta_end": beta * 2.0,
            "length_schedule": "cosine",
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def save_results(strategy: str, results: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"rad_{strategy}.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    return path


def generate_one(
    prompt: str,
    strategy: str,
    lm,
    lm_tokenizer,
    rm,
    rm_tokenizer,
    max_length: int,
    device: torch.device,
):
    inputs = lm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length - MAX_NEW_TOKENS,
    ).to(device)

    processor = RewardAugmentedLogitsProcessor(
        lm_tokenizer,
        rm_tokenizer,
        rm,
        topk=TOP_K,
        method="linear",
        beta=BASE_BETA,
        num_gpus=max(torch.cuda.device_count(), 1),
        inverse=False,
        beta_strategy=strategy,
        beta_params=get_beta_params(strategy),
        max_new_tokens=MAX_NEW_TOKENS,
    )

    with torch.inference_mode():
        output = lm.generate(
            **inputs,
            logits_processor=LogitsProcessorList([processor]),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=lm_tokenizer.pad_token_id,
            eos_token_id=lm_tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[0, prompt_length:]

    response = lm_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    beta_history = list(
        getattr(processor, "_beta_history", [])
    )

    return response, beta_history


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Adaptive RAD cần CUDA.")

    for path in (
        DATASET_PATH,
        LM_PATH,
        RM_BASE_PATH,
        RM_TOKENIZER_PATH,
        RM_CHECKPOINT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")

    device = torch.device("cuda")
    set_seed(SEED)

    print("Loading LM and RM...")

    lm, lm_tokenizer, rm, rm_tokenizer, max_length = load_models(device)
    samples = load_dataset(DATASET_PATH)

    print(f"Loaded {len(samples)} prompts.")

    for strategy in STRATEGIES:
        print(f"\nStrategy: {strategy}")

        set_seed(SEED)
        results = []

        for index, sample in enumerate(
            tqdm(samples, desc=strategy, unit="prompt")
        ):
            prompt = sample["_prompt_text"]
            start = time.perf_counter()

            try:
                response, beta_history = generate_one(
                    prompt,
                    strategy,
                    lm,
                    lm_tokenizer,
                    rm,
                    rm_tokenizer,
                    max_length,
                    device,
                )

                status = "success"
                error = None

            except Exception as exc:
                response = None
                beta_history = []
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"

            continuation = sample.get("continuation")
            reference = (
                continuation.get("text")
                if isinstance(continuation, dict)
                else continuation
            )

            results.append(
                {
                    "id": index,
                    "md5_hash": sample.get("md5_hash"),
                    "prompt": prompt,
                    "reference": reference,
                    "response": response,
                    "num_positive": sample.get("num_positive"),
                    "method": "RAD",
                    "strategy": strategy,
                    "base_model": str(LM_PATH),
                    "reward_model": str(RM_CHECKPOINT_PATH),
                    "base_beta": BASE_BETA,
                    "beta_history": beta_history,
                    "top_k": TOP_K,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "latency": round(
                        time.perf_counter() - start,
                        4,
                    ),
                    "status": status,
                    "error": error,
                }
            )

            save_results(strategy, results)

        print(f"Saved: {save_results(strategy, results)}")


if __name__ == "__main__":
    main()