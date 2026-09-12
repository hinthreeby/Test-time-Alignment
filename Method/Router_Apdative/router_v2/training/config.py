"""Validated Stage 5 Router V2 training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True)
class RouterTrainingConfig:
    STAGES: ClassVar[tuple[str, ...]] = ("taro", "state", "history", "alpha")
    ROUTER_KINDS: ClassVar[frozenset[str]] = frozenset({"taro", "smart"})
    PREFERENCE_SOURCES: ClassVar[frozenset[str]] = frozenset({"none", "cache"})
    DEVICES: ClassVar[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})

    schema_version: int = 1
    stage: str = "taro"
    router_kind: str = "taro"
    router_config_path: str = "router_v2/configs/taro_topk_nll.json"
    cache_root: str = "dataset/router_v2_cache/rad"
    output_dir: str = "results/router_v2/training/taro"
    task: str = "sentiment"
    seed: int = 2026
    device: str = "auto"
    allow_cpu_fallback: bool = True
    online_model_precision: str = "fp32"
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    entropy_weight: float = 0.0
    smoothness_weight: float = 0.0
    strength_weight: float = 0.0
    selected_score_source: str = "base_gold_logprob"
    preference_source: str = "none"
    same_average_baseline: bool = True
    enforce_stage_order: bool = True
    log_every_optimizer_steps: int = 10
    checkpoint_every_optimizer_steps: int = 250
    alpha_bins: int = 5
    near_boundary_fraction: float = 0.05
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported training schema_version: {self.schema_version}"
            )
        if self.stage not in self.STAGES:
            raise ValueError(f"Unsupported training stage: {self.stage}")
        if self.router_kind not in self.ROUTER_KINDS:
            raise ValueError(f"Unsupported router_kind: {self.router_kind}")
        if self.stage == "taro" and self.router_kind != "taro":
            raise ValueError("TARO stage requires router_kind='taro'")
        if self.stage != "taro" and self.router_kind != "smart":
            raise ValueError("Smart stages require router_kind='smart'")
        if self.task != "sentiment":
            raise ValueError("Stage 5 currently supports task='sentiment' only")
        if self.device not in self.DEVICES:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.online_model_precision != "fp32":
            raise ValueError(
                "Scientific Stage 5 training requires FP32 online logits"
            )
        positive_ints = {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "log_every_optimizer_steps": self.log_every_optimizer_steps,
            "checkpoint_every_optimizer_steps": (
                self.checkpoint_every_optimizer_steps
            ),
            "alpha_bins": self.alpha_bins,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Training integer fields must be positive: {invalid}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        regularizers = {
            "entropy_weight": self.entropy_weight,
            "smoothness_weight": self.smoothness_weight,
            "strength_weight": self.strength_weight,
        }
        invalid_regularizers = [
            name for name, value in regularizers.items() if value < 0.0
        ]
        if invalid_regularizers:
            raise ValueError(
                f"Regularizer weights must be non-negative: {invalid_regularizers}"
            )
        if self.stage == "taro" and (
            self.smoothness_weight != 0.0 or self.strength_weight != 0.0
        ):
            raise ValueError(
                "TARO stage does not enable Smart smoothness/strength regularizers"
            )
        if self.selected_score_source != "base_gold_logprob":
            raise ValueError(
                "selected_score_source must be 'base_gold_logprob'"
            )
        if self.preference_source not in self.PREFERENCE_SOURCES:
            raise ValueError(
                f"Unsupported preference_source: {self.preference_source}"
            )
        if self.stage == "alpha" and self.preference_source != "cache":
            raise ValueError("Alpha stage requires real cached preferences")
        if self.stage != "alpha" and self.preference_source != "none":
            raise ValueError("Only alpha stage may consume preference vectors")
        if not 0.0 < self.near_boundary_fraction < 0.5:
            raise ValueError("near_boundary_fraction must be in (0, 0.5)")
        for name, value in (
            ("max_train_samples", self.max_train_samples),
            ("max_validation_samples", self.max_validation_samples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null")

    @property
    def stage_index(self) -> int:
        return self.STAGES.index(self.stage)

    @property
    def predecessor_stage(self) -> str | None:
        if self.stage_index == 0:
            return None
        return self.STAGES[self.stage_index - 1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RouterTrainingConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown training config fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "RouterTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Training config must be a JSON object")
        return cls.from_dict(values)
