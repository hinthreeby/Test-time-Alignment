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

import argparse
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
    / "all.jsonl"
)

LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"

RM_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
RM_TOKENIZER_PATH = PROJECT_ROOT / "models" / "gpt2-small"
RM_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment" / "pytorch_model.bin"

OUTPUT_DIR = PROJECT_ROOT / "results"

ADAPTIVE_STRATEGIES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable adaptive RAD decoding.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--base-beta", type=float, default=BASE_BETA)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", *ADAPTIVE_STRATEGIES],
        default=ADAPTIVE_STRATEGIES,
    )
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts must be >= 0")
    if args.max_new_tokens <= 0 or args.topk <= 0:
        parser.error("--max-new-tokens and --topk must be > 0")
    return args


def load_dataset(path: Path, num_prompts: int) -> list[dict[str, Any]]:
    if num_prompts == 0:
        return []
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
                sample["_dataset_index"] = len(samples)
                samples.append(sample)

            if len(samples) >= num_prompts:
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

    load_result = rm.load_state_dict(state_dict, strict=False)
    allowed_legacy_suffixes = (".attn.bias", ".attn.masked_bias")
    unexpected_learned_keys = [
        key
        for key in load_result.unexpected_keys
        if not key.endswith(allowed_legacy_suffixes)
    ]
    if load_result.missing_keys or unexpected_learned_keys:
        raise RuntimeError(
            "Incompatible RAD reward checkpoint: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={unexpected_learned_keys}"
        )
    rm.eval().to(device)

    max_length = int(
        getattr(
            lm.config,
            "n_positions",
            getattr(lm.config, "max_position_embeddings", 1024),
        )
    )

    return lm, tokenizer, rm, rm_tokenizer, max_length


def get_beta_params(strategy: str, base_beta: float) -> dict:
    beta = base_beta

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


def output_path(output_dir: Path, strategy: str) -> Path:
    return output_dir / f"rad_{strategy}.json"


def save_results(output_dir: Path, strategy: str, results: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path(output_dir, strategy)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)

    return path


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    md5_hash = record.get("md5_hash")
    if md5_hash:
        return "md5", str(md5_hash)
    prompt = record.get("_prompt_text", record.get("prompt", ""))
    if isinstance(prompt, dict):
        prompt = prompt.get("text", "")
    return "prompt", str(prompt).strip()


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Resume output must be a JSON list: {path}")
    return [record for record in payload if isinstance(record, dict)]


def is_completed(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def generate_one(
    prompt: str,
    strategy: str,
    lm,
    lm_tokenizer,
    rm,
    rm_tokenizer,
    max_length: int,
    device: torch.device,
    max_new_tokens: int,
    topk: int,
    base_beta: float,
):
    inputs = lm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length - max_new_tokens,
    ).to(device)

    processor = RewardAugmentedLogitsProcessor(
        lm_tokenizer,
        rm_tokenizer,
        rm,
        topk=topk,
        method="linear",
        beta=base_beta,
        num_gpus=max(torch.cuda.device_count(), 1),
        inverse=False,
        beta_strategy=strategy,
        beta_params=get_beta_params(strategy, base_beta),
        max_new_tokens=max_new_tokens,
    )

    with torch.inference_mode():
        output = lm.generate(
            **inputs,
            logits_processor=LogitsProcessorList([processor]),
            max_new_tokens=max_new_tokens,
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
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Adaptive RAD cần CUDA.")

    for path in (
        args.dataset_path,
        LM_PATH,
        RM_BASE_PATH,
        RM_TOKENIZER_PATH,
        RM_CHECKPOINT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")

    samples = load_dataset(args.dataset_path, args.num_prompts)

    print(f"Loaded {len(samples)} prompts.")

    strategy_states = {}
    for strategy in args.strategies:
        existing = load_existing_results(output_path(args.output_dir, strategy))
        by_key = {record_key(record): record for record in existing}
        pending = [
            sample
            for sample in samples
            if not is_completed(by_key.get(record_key(sample)))
        ]
        strategy_states[strategy] = (existing, by_key, pending)
        print(
            f"{strategy}: completed={len(samples) - len(pending)}, "
            f"remaining={len(pending)}"
        )

    if not any(state[2] for state in strategy_states.values()):
        print("No prompts need adaptive decoding; outputs are unchanged.")
        return

    device = torch.device("cuda")
    set_seed(SEED)
    print("Loading LM and RM...")
    lm, lm_tokenizer, rm, rm_tokenizer, max_length = load_models(device)

    for strategy in args.strategies:
        print(f"\nStrategy: {strategy}")

        set_seed(SEED)
        existing, results_by_key, pending = strategy_states[strategy]

        if not pending:
            print(f"Already complete: {output_path(args.output_dir, strategy)}")
            continue

        for sample in tqdm(pending, desc=strategy, unit="prompt"):
            index = sample["_dataset_index"]
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
                    args.max_new_tokens,
                    args.topk,
                    args.base_beta,
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

            results_by_key[record_key(sample)] = {
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
                "base_beta": args.base_beta,
                "beta_history": beta_history,
                "top_k": args.topk,
                "max_new_tokens": args.max_new_tokens,
                "latency": round(
                    time.perf_counter() - start,
                    4,
                ),
                "status": status,
                "error": error,
            }

            save_results(args.output_dir, strategy, list(results_by_key.values()))

        print(
            f"Saved: {save_results(args.output_dir, strategy, list(results_by_key.values()))}"
        )


if __name__ == "__main__":
    main()
