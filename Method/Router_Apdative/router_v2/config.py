"""Configuration contract for Router V2 TARO models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True)
class TARORouterConfig:
    """Validated configuration for the independent TARO router."""

    SUPPORTED_MODES: ClassVar[frozenset[str]] = frozenset(
        {"taro_topk_nll", "taro_topk_nll_entropy"}
    )
    SUPPORTED_DEVICES: ClassVar[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})
    SUPPORTED_REDUCTIONS: ClassVar[frozenset[str]] = frozenset({"mean", "sum"})

    schema_version: int = 2
    method_label: str = "TARO"
    architecture: str = "taro_topk_flatten_tanh"
    activation: str = "tanh"
    mode: str = "taro_topk_nll"
    vocab_size: int = 50257
    top_k: int = 20
    token_embedding_dim: int = 32
    hidden_dim: int = 128
    entropy_weight: float = 0.0
    alpha_eps: float = 1e-6
    initial_alpha: float = 0.5
    loss_reduction: str = "mean"
    device: str = "auto"
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Unsupported config schema_version: {self.schema_version}")
        if self.method_label != "TARO":
            raise ValueError("method_label must be exactly 'TARO'")
        if self.architecture != "taro_topk_flatten_tanh":
            raise ValueError("Unsupported TARO architecture")
        if self.activation != "tanh":
            raise ValueError("Faithful TARO Router V2 requires activation='tanh'")
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported TARO mode: {self.mode}")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if not 1 <= self.top_k <= self.vocab_size:
            raise ValueError("top_k must be in [1, vocab_size]")
        if self.token_embedding_dim <= 0:
            raise ValueError("token_embedding_dim must be positive")
        if self.hidden_dim != 128:
            raise ValueError("TARO Router V2 requires hidden_dim=128")
        if self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative")
        if self.mode == "taro_topk_nll" and self.entropy_weight != 0.0:
            raise ValueError("taro_topk_nll requires entropy_weight=0")
        if not 0.0 < self.alpha_eps < 0.5:
            raise ValueError("alpha_eps must be in (0, 0.5)")
        if not 0.0 < self.initial_alpha < 1.0:
            raise ValueError("initial_alpha must be in (0, 1)")
        if self.loss_reduction not in self.SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported loss_reduction: {self.loss_reduction}")
        if self.device not in self.SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device: {self.device}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TARORouterConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown TARO config fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "TARORouterConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("TARO config must be a JSON object")
        return cls.from_dict(values)

    def save_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite config: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
