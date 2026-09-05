"""Lazy production runtime construction for PARM-TARO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.adapters.parm_adapter import (
    activate_vendored_parm_runtime,
    freeze_for_inference,
    inspect_parm_adapter_config,
    named_preference_to_parm,
    set_parm_preference,
)
from PARM_TARO.adapters.router_adapter import SmartRouterLambdaProvider
from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.config import ParmTaroConfig
from router_v2.device import resolve_device
from router_v2.guide_model.provenance import tokenizer_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ParmTaroRuntime:
    base_model: Any
    guide_model: Any
    tokenizer: Any
    alignment: PARMTokenAlignment
    lambda_provider: SmartRouterLambdaProvider
    device: torch.device
    tokenizer_semantic_sha256: str
    vendored_origins: dict[str, str]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_runtime(config: ParmTaroConfig) -> ParmTaroRuntime:
    """Load frozen PARM models and a compatible preference-aware Router V2."""

    base_path = _resolve(config.base_model_path)
    tokenizer_path = _resolve(config.tokenizer_path)
    adapter_path = _resolve(config.parm_adapter_path)
    router_path = _resolve(config.router_checkpoint_path)
    for label, path in (
        ("base model", base_path),
        ("tokenizer", tokenizer_path),
        ("PARM adapter", adapter_path),
        ("Router checkpoint", router_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    adapter_config = inspect_parm_adapter_config(
        adapter_path,
        expected_preference_dim=config.preference_dim,
    )
    guide_base_path = _resolve(str(adapter_config["base_model_name_or_path"]))
    if not guide_base_path.exists():
        raise FileNotFoundError(f"Missing PBLORA base model: {guide_base_path}")

    vendored = activate_vendored_parm_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(
        config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    dtype = _torch_dtype(config.model_dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    guide_backbone = AutoModelForCausalLM.from_pretrained(
        guide_base_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    guide_model = vendored.peft.PeftModel.from_pretrained(
        guide_backbone,
        adapter_path,
        is_trainable=False,
    )
    set_parm_preference(
        guide_model,
        named_preference_to_parm(
            torch.tensor(config.preference, device=device)
        ),
    )
    freeze_for_inference(base_model, guide_model)
    alignment = PARMTokenAlignment.validate(
        tokenizer,
        base_vocab_size=int(base_model.config.vocab_size),
        guide_vocab_size=int(guide_model.config.vocab_size),
    )
    descriptor = tokenizer_descriptor(tokenizer, tokenizer_path)
    semantic_hash = str(descriptor["semantic_sha256"])
    if semantic_hash != config.router_tokenizer_semantic_sha256:
        raise ValueError(
            "Configured Router tokenizer hash does not match the loaded tokenizer"
        )
    provider = SmartRouterLambdaProvider.from_checkpoint(
        router_path,
        device=device,
        expected_vocab_size=alignment.base_vocab_size,
        preference_dim=config.preference_dim,
        tokenizer_semantic_sha256=semantic_hash,
    )
    if provider.router.config.top_k != config.top_k:
        raise ValueError("Router checkpoint top_k does not match PARM-TARO config")
    return ParmTaroRuntime(
        base_model=base_model,
        guide_model=guide_model,
        tokenizer=tokenizer,
        alignment=alignment,
        lambda_provider=provider,
        device=device,
        tokenizer_semantic_sha256=semantic_hash,
        vendored_origins=vendored.origins,
    )
