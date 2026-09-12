"""Strict configuration for Stage 9 alpha-preference production training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AlphaPreferenceProductionConfig:
    schema_version: int = 1
    method_label: str = "PARM_TARO_ALPHA_PREFERENCE_PRODUCTION"
    base_training_config_path: str = "PARM_TARO/configs/train_stage9_v2_alpha.json"
    required_alpha_path_diagnostic: str = (
        "PARM_TARO/reports/stage9_alpha_path_v2_diagnostic.json"
    )
    required_pilot_v3_report_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot_v3/pilot_report.json"
    )
    required_pilot_v3_checkpoint_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot_v3/pilot_final.pt"
    )
    pilot_v2_report_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot_v2/pilot_report.json"
    )
    pilot_v2_checkpoint_path: str = (
        "results/parm_taro/training/v2_alpha_preference_pilot_v2/pilot_final.pt"
    )
    no_alpha_checkpoint: str = "results/parm_taro/training/v2_no_alpha/best.pt"
    output_dir: str = "results/parm_taro/training/v2_alpha_preference"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    seed: int = 2026
    expected_train_samples: int = 8000
    expected_validation_samples: int = 500
    calibration_samples: int = 16
    warmup_epochs: int = 1
    quality_epochs: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    residual_hidden_dim: int = 32
    residual_output_weight_std: float = 0.01
    preference_weight: float = 1.0
    max_nll_weight: float = 0.25
    nll_gradient_target_ratio: float = 0.25
    strength_weight: float = 1.0
    logit_sensitivity_gradient_target_ratio: float = 1.0
    max_logit_sensitivity_weight: float = 100.0
    freeze_state_router_during_warmup: bool = True
    min_control_lambda_delta: float = 0.001
    min_mean_lambda: float = 0.005
    max_nll_degradation_vs_no_alpha: float = 0.05
    require_shuffled_preference_degradation: bool = True
    min_router_lambda_std: float = 1e-6
    log_every_optimizer_steps: int = 10
    checkpoint_every_optimizer_steps: int = 100

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported alpha-preference production schema")
        if self.method_label != "PARM_TARO_ALPHA_PREFERENCE_PRODUCTION":
            raise ValueError("Invalid alpha-preference production method_label")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("Production device must be auto, cuda, or cpu")
        positive_ints = (
            self.expected_train_samples,
            self.expected_validation_samples,
            self.calibration_samples,
            self.warmup_epochs,
            self.quality_epochs,
            self.gradient_accumulation_steps,
            self.residual_hidden_dim,
            self.log_every_optimizer_steps,
            self.checkpoint_every_optimizer_steps,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("Production integer controls must be positive")
        if self.expected_train_samples != 8000:
            raise ValueError("Stage 9 production requires all 8,000 train examples")
        if self.expected_validation_samples != 500:
            raise ValueError("Stage 9 production requires all 500 validation examples")
        if self.calibration_samples > self.expected_train_samples:
            raise ValueError("calibration_samples exceeds the train split")
        positive = (
            self.learning_rate,
            self.max_grad_norm,
            self.residual_output_weight_std,
            self.preference_weight,
            self.max_logit_sensitivity_weight,
            self.min_router_lambda_std,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Production positive controls must be positive")
        nonnegative = (
            self.weight_decay,
            self.max_nll_weight,
            self.nll_gradient_target_ratio,
            self.strength_weight,
            self.logit_sensitivity_gradient_target_ratio,
            self.min_mean_lambda,
            self.max_nll_degradation_vs_no_alpha,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Production controls must be non-negative")
        if not self.freeze_state_router_during_warmup:
            raise ValueError("Production requires alpha-path-only warm-up")
        if self.min_control_lambda_delta != 0.001:
            raise ValueError("Production must preserve min_control_lambda_delta=0.001")
        if self.output_dir != "results/parm_taro/training/v2_alpha_preference":
            raise ValueError("Production output_dir must use the isolated production path")

    @property
    def max_train_samples(self) -> int:
        return self.expected_train_samples

    @property
    def max_validation_samples(self) -> int:
        return self.expected_validation_samples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Any]
    ) -> "AlphaPreferenceProductionConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown production config fields: {unknown}")
        return cls(**dict(values))

    @classmethod
    def load_json(cls, path: str | Path) -> "AlphaPreferenceProductionConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Production config must be a JSON object")
        return cls.from_dict(values)
