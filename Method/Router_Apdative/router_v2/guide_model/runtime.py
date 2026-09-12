"""Shared deterministic runtime for scientific cache extraction and audit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from router_v2.guide_model.provenance import (
    tokenizers_exactly_compatible,
)


@dataclass(frozen=True)
class CompatibleTokenizerPair:
    base: Any
    guide: Any
    compatibility_checks: dict[str, bool]


@dataclass(frozen=True)
class FrozenCausalModelPair:
    base: Any
    guide: Any
    dtype: torch.dtype


def configure_deterministic_inference(
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("Inference seed must be non-negative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "device_type": device.type,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
    }


def load_compatible_tokenizer_pair(
    base_path: str | Path,
    guide_path: str | Path,
) -> CompatibleTokenizerPair:
    base = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        use_fast=True,
    )
    guide = AutoTokenizer.from_pretrained(
        guide_path,
        local_files_only=True,
        use_fast=True,
    )
    for tokenizer in (base, guide):
        if tokenizer.eos_token_id is None:
            raise ValueError("Cache tokenizer must define EOS")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    compatible, checks = tokenizers_exactly_compatible(base, guide)
    if not compatible:
        raise ValueError(f"Base/guide tokenizer mismatch: {checks}")
    return CompatibleTokenizerPair(
        base=base,
        guide=guide,
        compatibility_checks=checks,
    )


def load_frozen_causal_model_pair(
    base_path: str | Path,
    guide_path: str | Path,
    *,
    device: torch.device,
    precision: str,
) -> FrozenCausalModelPair:
    if precision not in {"fp16", "fp32"}:
        raise ValueError("Inference precision must be fp16 or fp32")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 inference requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    guide_base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    guide = PeftModel.from_pretrained(
        guide_base,
        guide_path,
        is_trainable=False,
    ).to(device)
    for model in (base, guide):
        model.config.use_cache = False
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if precision == "fp32":
        non_fp32 = {
            str(parameter.dtype)
            for model in (base, guide)
            for parameter in model.parameters()
            if parameter.is_floating_point()
            and parameter.dtype != torch.float32
        }
        if non_fp32:
            raise TypeError(
                "Scientific FP32 inference loaded non-FP32 parameters: "
                f"{sorted(non_fp32)}"
            )
    return FrozenCausalModelPair(base=base, guide=guide, dtype=dtype)
