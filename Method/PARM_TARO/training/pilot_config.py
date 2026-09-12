"""Strict isolated config for the Stage 9 alpha-preference GPU pilot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AlphaPreferencePilotConfig:
    schema_version: int = 1
    method_label: str = "PARM_TARO_ALPHA_PREFERENCE_PILOT"
    base_training_config_path: str = (
        "PARM_TARO/configs/train_stage9_v2_alpha.json"
    )
    required_diagnostic_path: str = (
        "PARM_TARO/reports/stage9_alpha_collapse_diagnostic.json"
    )
    initial_router_checkpoint: str = (
        "results/parm_taro/training/v2_alpha/best.pt"
    )
    no_alpha_checkpoint: str = (
        "results/parm_taro/training/v2_no_alpha/best.pt"
    )
    output_dir: str = "results/parm_taro/training/v2_alpha_preference_pilot"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    seed: int = 2026
    max_train_samples: int = 128
    max_validation_samples: int = 50
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    preference_loss_weight: float = 1.0
    nll_weight: float = 0.25
    strength_weight: float = 0.1
    strength_target: float = 0.02
    pairwise_logit_scale: float = 1.0
    min_mean_lambda: float = 0.005
    min_control_lambda_delta: float = 0.001
    max_nll_degradation_vs_no_alpha: float = 0.05
    require_shuffled_preference_degradation: bool = True
    log_every_optimizer_steps: int = 5

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported alpha-preference pilot schema")
        if self.method_label != "PARM_TARO_ALPHA_PREFERENCE_PILOT":
            raise ValueError("Invalid alpha-preference pilot method_label")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("Pilot device must be auto, cuda, or cpu")
        positive_ints = (
            self.max_train_samples,
            self.max_validation_samples,
            self.num_epochs,
            self.gradient_accumulation_steps,
            self.log_every_optimizer_steps,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("Pilot integer controls must be positive")
        if self.seed < 0 or self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("Invalid pilot seed or optimizer controls")
        if self.weight_decay < 0.0:
            raise ValueError("Pilot weight_decay must be non-negative")
        nonnegative = (
            self.preference_loss_weight,
            self.nll_weight,
            self.strength_weight,
            self.strength_target,
            self.min_mean_lambda,
            self.min_control_lambda_delta,
            self.max_nll_degradation_vs_no_alpha,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Pilot objective/gate controls must be non-negative")
        if self.pairwise_logit_scale <= 0.0:
            raise ValueError("pairwise_logit_scale must be positive")
        if self.min_control_lambda_delta <= 0.0:
            raise ValueError("min_control_lambda_delta must remain positive")
        if (
            self.preference_loss_weight == 0.0
            or self.nll_weight == 0.0
        ):
            raise ValueError("Pilot requires pairwise preference and NLL terms")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AlphaPreferencePilotConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown alpha-preference pilot fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "AlphaPreferencePilotConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Alpha-preference pilot config must be an object")
        return cls.from_dict(values)
