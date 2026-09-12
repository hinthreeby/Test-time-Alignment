"""Strict configuration contract for PARM-TARO inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ParmTaroConfig:
    """Production paths and decoding controls for adaptive PARM guidance."""

    schema_version: int = 1
    method_label: str = "PARM_TARO"
    base_model_path: str = "models/tulu-2-7b"
    tokenizer_path: str = "models/tulu-2-7b"
    parm_adapter_path: str = "results/parm_taro/checkpoints/parm_pku_pblora"
    router_checkpoint_path: str = (
        "results/router_v2/training/alpha_parm/best.pt"
    )
    preference_dim: int = 2
    preference: tuple[float, ...] = (0.5, 0.5)
    router_tokenizer_semantic_sha256: str = "REQUIRED_AFTER_TRAINING"
    guide_vocab_policy: str = "base_tokenizer_prefix"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    model_dtype: str = "float16"
    top_k: int = 20
    max_new_tokens: int = 64
    temperature: float = 1.0
    do_sample: bool = False
    stop_on_eos: bool = True
    seed: int = 2026
    static_reference_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported PARM-TARO schema_version")
        if self.method_label != "PARM_TARO":
            raise ValueError("method_label must be exactly 'PARM_TARO'")
        if self.preference_dim <= 0:
            raise ValueError("preference_dim must be positive")
        if len(self.preference) != self.preference_dim:
            raise ValueError("preference length must equal preference_dim")
        if not all(float(value) >= 0.0 for value in self.preference):
            raise ValueError("preference values must be non-negative")
        if self.guide_vocab_policy != "base_tokenizer_prefix":
            raise ValueError("Unsupported guide_vocab_policy")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported model_dtype")
        if self.top_k <= 0 or self.max_new_tokens <= 0:
            raise ValueError("top_k and max_new_tokens must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.static_reference_scale < 0.0:
            raise ValueError("static_reference_scale must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["preference"] = list(self.preference)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ParmTaroConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown PARM-TARO config fields: {unknown}")
        normalized = dict(values)
        if "preference" in normalized:
            preference = normalized["preference"]
            if not isinstance(preference, (list, tuple)):
                raise TypeError("preference must be a JSON array")
            normalized["preference"] = tuple(float(value) for value in preference)
        return cls(**normalized)

    @classmethod
    def load_json(cls, path: str | Path) -> "ParmTaroConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("PARM-TARO config must be a JSON object")
        return cls.from_dict(values)
