"""Strict configuration for the measured Stage 9 alpha pilot V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AlphaPreferencePilotV2Config:
    schema_version: int = 1
    method_label: str = "PARM_TARO_ALPHA_PREFERENCE_PILOT_V2"
    base_training_config_path: str = "PARM_TARO/configs/train_stage9_v2_alpha.json"
    required_failure_diagnostic_path: str = (
        "PARM_TARO/reports/stage9_alpha_pilot_v1_failure_diagnostic.json"
    )
    first_pilot_report_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot/pilot_report.json"
    )
    first_pilot_checkpoint_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot/pilot_final.pt"
    )
    alpha_checkpoint: str = "results/parm_taro/training/v2_alpha/best.pt"
    no_alpha_checkpoint: str = "results/parm_taro/training/v2_no_alpha/best.pt"
    output_dir: str = "results/parm_taro/training/v2_alpha_preference_pilot_v2"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    seed: int = 2026
    max_train_samples: int = 256
    max_validation_samples: int = 50
    calibration_samples: int = 16
    warmup_epochs: int = 1
    quality_epochs: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    preference_weight: float = 1.0
    max_nll_weight: float = 0.25
    nll_gradient_target_ratio: float = 0.25
    strength_weight: float = 1.0
    sensitivity_gradient_target_ratio: float = 1.0
    max_sensitivity_weight: float = 100.0
    min_control_lambda_delta: float = 0.001
    min_mean_lambda: float = 0.005
    max_nll_degradation_vs_no_alpha: float = 0.05
    require_shuffled_preference_degradation: bool = True
    log_every_optimizer_steps: int = 10

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported alpha pilot V2 schema")
        if self.method_label != "PARM_TARO_ALPHA_PREFERENCE_PILOT_V2":
            raise ValueError("Invalid alpha pilot V2 method_label")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("Pilot V2 device must be auto, cuda, or cpu")
        positive_ints = (
            self.max_train_samples,
            self.max_validation_samples,
            self.calibration_samples,
            self.warmup_epochs,
            self.quality_epochs,
            self.gradient_accumulation_steps,
            self.log_every_optimizer_steps,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("Pilot V2 integer controls must be positive")
        if self.calibration_samples > self.max_train_samples:
            raise ValueError("calibration_samples cannot exceed max_train_samples")
        if self.seed < 0 or self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("Invalid pilot V2 optimizer controls")
        nonnegative = (
            self.weight_decay,
            self.preference_weight,
            self.max_nll_weight,
            self.nll_gradient_target_ratio,
            self.strength_weight,
            self.sensitivity_gradient_target_ratio,
            self.max_sensitivity_weight,
            self.min_mean_lambda,
            self.max_nll_degradation_vs_no_alpha,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Pilot V2 controls must be non-negative")
        if self.preference_weight <= 0.0:
            raise ValueError("Pilot V2 requires preference loss")
        if self.min_control_lambda_delta != 0.001:
            raise ValueError("Pilot V2 must preserve min_control_lambda_delta=0.001")
        if self.max_sensitivity_weight <= 0.0:
            raise ValueError("max_sensitivity_weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AlphaPreferencePilotV2Config":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown alpha pilot V2 fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "AlphaPreferencePilotV2Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Alpha pilot V2 config must be an object")
        return cls.from_dict(values)

