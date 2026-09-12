"""Configuration contract for the feature-conditioned Smart Router V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True)
class SmartRouterConfig:
    """Validated architecture and ablation toggles for Smart Router V2."""

    VARIANT_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "v2_topk_confidence": {
            "candidate_feature_mode": "token_aware_mean_max",
            "use_confidence": True,
            "use_position": False,
            "use_history": False,
            "use_preference": False,
        },
        "v2_topk_confidence_position": {
            "candidate_feature_mode": "token_aware_mean_max",
            "use_confidence": True,
            "use_position": True,
            "use_history": False,
            "use_preference": False,
        },
        "v2_topk_state_history": {
            "candidate_feature_mode": "token_aware_mean_max",
            "use_confidence": True,
            "use_position": True,
            "use_history": True,
            "use_preference": False,
        },
        "v2_topk_state_history_alpha": {
            "candidate_feature_mode": "token_aware_mean_max",
            "use_confidence": True,
            "use_position": True,
            "use_history": True,
            "use_preference": True,
        },
    }
    SUPPORTED_VARIANTS: ClassVar[frozenset[str]] = frozenset(
        {"custom", *VARIANT_PRESETS}
    )
    SUPPORTED_CANDIDATE_MODES: ClassVar[frozenset[str]] = frozenset(
        {"disabled", "taro_flatten", "token_aware_mean_max"}
    )
    SUPPORTED_ACTIVATIONS: ClassVar[frozenset[str]] = frozenset(
        {"gelu", "tanh"}
    )
    SUPPORTED_DEVICES: ClassVar[frozenset[str]] = frozenset(
        {"auto", "cuda", "cpu"}
    )

    schema_version: int = 1
    method_label: str = "SMART_ROUTER_V2"
    architecture: str = "token_aware_state_router"
    variant: str = "custom"
    vocab_size: int = 50257
    top_k: int = 20
    token_embedding_dim: int = 32
    candidate_hidden_dim: int = 64
    candidate_feature_mode: str = "token_aware_mean_max"
    use_confidence: bool = True
    use_position: bool = True
    use_history: bool = True
    use_preference: bool = False
    max_position: int = 79
    history_hidden_dim: int = 32
    preference_dim: int = 1
    preference_hidden_dim: int = 16
    preference_embedding_dim: int = 16
    fusion_hidden_dim: int = 128
    fusion_bottleneck_dim: int = 64
    activation: str = "gelu"
    lambda_max: float = 1.0
    lambda_eps: float = 1e-6
    initial_gate: float = 0.5
    device: str = "auto"
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported Smart Router schema_version: {self.schema_version}"
            )
        if self.method_label != "SMART_ROUTER_V2":
            raise ValueError("method_label must be exactly 'SMART_ROUTER_V2'")
        if self.architecture != "token_aware_state_router":
            raise ValueError("Unsupported Smart Router architecture")
        if self.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported Smart Router variant: {self.variant}")
        if self.candidate_feature_mode not in self.SUPPORTED_CANDIDATE_MODES:
            raise ValueError(
                "Unsupported candidate_feature_mode: "
                f"{self.candidate_feature_mode}"
            )
        if self.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.device not in self.SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if not 1 <= self.top_k <= self.vocab_size:
            raise ValueError("top_k must be in [1, vocab_size]")
        positive_dimensions = {
            "token_embedding_dim": self.token_embedding_dim,
            "candidate_hidden_dim": self.candidate_hidden_dim,
            "history_hidden_dim": self.history_hidden_dim,
            "preference_dim": self.preference_dim,
            "preference_hidden_dim": self.preference_hidden_dim,
            "preference_embedding_dim": self.preference_embedding_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "fusion_bottleneck_dim": self.fusion_bottleneck_dim,
        }
        invalid = [name for name, value in positive_dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"Dimensions must be positive: {invalid}")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive")
        if not 0.0 < self.lambda_eps < 0.5:
            raise ValueError("lambda_eps must be in (0, 0.5)")
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1)")
        if not any(
            (
                self.candidate_feature_mode != "disabled",
                self.use_confidence,
                self.use_position,
                self.use_history,
                self.use_preference,
            )
        ):
            raise ValueError("At least one Smart Router feature group is required")
        if self.variant != "custom":
            expected = self.VARIANT_PRESETS[self.variant]
            mismatched = {
                name: (getattr(self, name), value)
                for name, value in expected.items()
                if getattr(self, name) != value
            }
            if mismatched:
                raise ValueError(
                    f"Variant {self.variant!r} feature contract mismatch: "
                    f"{mismatched}. Use variant='custom' for ablations."
                )

    @property
    def feature_toggles(self) -> dict[str, Any]:
        return {
            "candidate_feature_mode": self.candidate_feature_mode,
            "use_confidence": self.use_confidence,
            "use_position": self.use_position,
            "use_history": self.use_history,
            "use_preference": self.use_preference,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SmartRouterConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Smart Router config fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "SmartRouterConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Smart Router config must be a JSON object")
        return cls.from_dict(values)

    def save_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite Smart Router config: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
