#!/usr/bin/env python3
"""Run original fixed-beta RAD or the new learned-router RAD path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.adaptive_router_decoding import (
    AdaptiveRADConfig,
    adaptive_rad_generate,
    load_router_checkpoint_strict,
    original_fixed_beta_rad_generate,
    write_adaptive_generation_json,
)
from router.scripts.validate_models import configure_gpt2_padding, load_reward_model


DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"
DEFAULT_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_RM_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_RM_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_RM_CHECKPOINT_PATH = DEFAULT_RM_TOKENIZER_PATH / "pytorch_model.bin"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "adaptive_rad.json"


def read_prompts(path: Path, limit: int | None) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_value = row.get("prompt", "")
            prompt = prompt_value.get("text", "") if isinstance(prompt_value, dict) else prompt_value
            prompt = str(prompt).strip()
            if prompt:
                prompts.append(prompt)
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def load_base_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    configure_gpt2_padding(tokenizer, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, tokenizer


def load_reward(path: Path, tokenizer_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    reward_model, tokenizer, _metadata = load_reward_model(path, tokenizer_path, checkpoint_path, device)
    configure_gpt2_padding(tokenizer, padding_side="left")
    reward_model.eval().to(device)
    for parameter in reward_model.parameters():
        parameter.requires_grad = False
    return reward_model, tokenizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAD with separate original and learned-router modes.")
    parser.add_argument("--decoding-mode", choices=["original_fixed_beta", "learned_router"], required=True)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--lm-path", type=Path, default=DEFAULT_LM_PATH)
    parser.add_argument("--rm-base-path", type=Path, default=DEFAULT_RM_BASE_PATH)
    parser.add_argument("--rm-tokenizer-path", type=Path, default=DEFAULT_RM_TOKENIZER_PATH)
    parser.add_argument("--rm-checkpoint-path", type=Path, default=DEFAULT_RM_CHECKPOINT_PATH)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-reward-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--beta-max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device)
    config = AdaptiveRADConfig(
        top_k=int(args.top_k),
        beta_max=float(args.beta_max),
        max_new_tokens=int(args.max_new_tokens),
        max_reward_length=int(args.max_reward_length),
        do_sample=not bool(args.greedy),
        temperature=float(args.temperature),
        seed=int(args.seed),
        inverse=bool(args.inverse),
        base_model_id=str(args.lm_path),
        reward_model_id=str(args.rm_checkpoint_path),
    )
    base_model, base_tokenizer = load_base_model(args.lm_path, device)
    reward_model, reward_tokenizer = load_reward(args.rm_base_path, args.rm_tokenizer_path, args.rm_checkpoint_path, device)
    prompts = read_prompts(args.dataset_path, args.num_prompts)
    if args.decoding_mode == "original_fixed_beta":
        samples = original_fixed_beta_rad_generate(
            prompts,
            base_model,
            base_tokenizer,
            reward_model,
            reward_tokenizer,
            beta=float(args.beta),
            config=config,
        )
    else:
        if args.router_checkpoint is None:
            raise ValueError("--router-checkpoint is required for --decoding-mode learned_router")
        router = load_router_checkpoint_strict(args.router_checkpoint, config, map_location=device).to(device)
        samples = adaptive_rad_generate(
            prompts,
            base_model,
            base_tokenizer,
            reward_model,
            reward_tokenizer,
            router,
            config,
        )
    write_adaptive_generation_json(
        args.output_path,
        samples,
        config,
        router_checkpoint_path=args.router_checkpoint if args.decoding_mode == "learned_router" else None,
        reward_checkpoint_path=args.rm_checkpoint_path,
        decoding_mode=args.decoding_mode,
    )
    report = {
        "decoding_mode": args.decoding_mode,
        "num_samples": len(samples),
        "output_path": str(args.output_path),
        "average_latency_per_token": (
            sum(sample.latency["average_latency_per_token"] for sample in samples) / max(len(samples), 1)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
