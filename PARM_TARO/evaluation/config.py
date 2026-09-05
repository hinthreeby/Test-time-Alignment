"""Strict, frozen configuration for final PARM-TARO evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


METHODS = (
    "parm_static",
    "parm_taro",
    "parm_v2_no_alpha",
    "parm_v2_full_alpha",
    "parm_v2_full_alpha_same_average",
    "parm_v2_full_alpha_shuffled_alpha",
    "parm_v2_full_alpha_fixed_alpha",
)


@dataclass(frozen=True)
class ParmTaroEvaluationConfig:
    """Protocol fields are intentionally explicit and schema validated."""

    DEVICES: ClassVar[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})
    schema_version: int = 1
    method_label: str = "PARM_TARO_FINAL_EVALUATION"
    output_root: str = "results/parm_taro/evaluation"
    data_root: str = "dataset/parm_taro"
    stage9_runtime_config: str = "PARM_TARO/configs/train_stage9_taro.json"
    taro_checkpoint: str = "results/parm_taro/training/taro/best.pt"
    no_alpha_checkpoint: str = "results/parm_taro/training/v2_no_alpha/best.pt"
    full_alpha_checkpoint: str = (
        "results/parm_taro/training/v2_alpha_preference/final.pt"
    )
    full_alpha_checkpoint_sha256: str = (
        "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20"
    )
    helpfulness_model_path: str = "models/beaver-7b-v1.0-reward"
    harmlessness_cost_model_path: str = "models/beaver-7b-v1.0-cost"
    safe_rlhf_source_path: str = "models/safe-rlhf-source"
    scorer_backend: str = "safe_rlhf_auto_model_for_score"
    methods: tuple[str, ...] = METHODS
    alpha_helpfulness_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    fixed_alpha: tuple[float, float] = (0.5, 0.5)
    shuffled_alpha_offset: int = 1
    guide_alpha_policy_for_router_controls: str = "hold_requested_alpha"
    same_average_lambda_policy: str = (
        "token_weighted_mean_of_full_alpha_in_same_run"
    )
    evaluation_split: str = "test_prompt_only"
    calibration_split: str = "validation"
    calibration_pool: str = "both_reference_responses"
    normalization: str = "validation_quantile_minmax_clip"
    normalization_lower_quantile: float = 0.01
    normalization_upper_quantile: float = 0.99
    hv_reference_point: tuple[float, float] = (0.0, 0.0)
    regret_oracle: str = "shared_all_method_candidate_pool_per_prompt_alpha_seed"
    max_new_tokens: int = 64
    temperature: float = 1.0
    do_sample: bool = False
    stop_on_eos: bool = True
    seeds: tuple[int, ...] = (2026,)
    smoke_prompt_limit: int = 2
    scorer_batch_size: int = 2
    scorer_max_length: int = 512
    scorer_load_in_4bit: bool = True
    model_dtype: str = "float16"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    bootstrap_samples: int = 2000
    permutation_samples: int = 10000
    statistics_seed: int = 2026
    multiple_comparison_correction: str = "holm"
    primary_comparisons: tuple[tuple[str, str], ...] = (
        ("parm_v2_full_alpha", "parm_static"),
        ("parm_v2_full_alpha", "parm_taro"),
        ("parm_v2_full_alpha", "parm_v2_no_alpha"),
        ("parm_v2_full_alpha", "parm_v2_full_alpha_same_average"),
    )
    primary_metrics: tuple[str, ...] = (
        "hypervolume",
        "mip",
        "pcs",
        "preference_regret",
        "perplexity",
        "coherence",
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.method_label != "PARM_TARO_FINAL_EVALUATION":
            raise ValueError("Unsupported Stage 10 evaluation schema or method label")
        if not self.output_root.startswith("results/parm_taro/evaluation"):
            raise ValueError("Stage 10 outputs must stay under results/parm_taro/evaluation")
        if self.methods != METHODS:
            raise ValueError("Stage 10 requires the seven frozen evaluation methods")
        if self.alpha_helpfulness_grid != tuple(sorted(set(self.alpha_helpfulness_grid))):
            raise ValueError("Alpha grid must be sorted and unique")
        if self.alpha_helpfulness_grid != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Final Stage 10 alpha grid is frozen")
        if self.fixed_alpha != (0.5, 0.5) or abs(sum(self.fixed_alpha) - 1.0) > 1e-12:
            raise ValueError("Fixed alpha control must be [0.5, 0.5]")
        if self.shuffled_alpha_offset % len(self.alpha_helpfulness_grid) == 0:
            raise ValueError("Shuffled alpha must be a derangement of grid indices")
        if self.guide_alpha_policy_for_router_controls != "hold_requested_alpha":
            raise ValueError("Router controls must keep the PBLORA alpha correct")
        if self.same_average_lambda_policy != "token_weighted_mean_of_full_alpha_in_same_run":
            raise ValueError("Unsupported same-average-lambda protocol")
        if self.evaluation_split != "test_prompt_only" or self.calibration_split != "validation":
            raise ValueError("Final test and validation-only calibration splits are frozen")
        if self.calibration_pool != "both_reference_responses":
            raise ValueError("Calibration must use both validation reference responses")
        if self.normalization != "validation_quantile_minmax_clip":
            raise ValueError("Unsupported objective normalization")
        if not 0.0 <= self.normalization_lower_quantile < self.normalization_upper_quantile <= 1.0:
            raise ValueError("Invalid normalization quantiles")
        if self.hv_reference_point != (0.0, 0.0):
            raise ValueError("Normalized Stage 10 HV reference point is frozen at [0, 0]")
        if self.regret_oracle != "shared_all_method_candidate_pool_per_prompt_alpha_seed":
            raise ValueError("Unsupported preference-regret oracle")
        if self.max_new_tokens != 64 or self.temperature != 1.0:
            raise ValueError("Final generation budget is frozen at greedy 64-token decoding")
        if self.do_sample or not self.stop_on_eos:
            raise ValueError("Final Stage 10 uses greedy decoding with normal EOS termination")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be non-empty and unique")
        if min(self.smoke_prompt_limit, self.scorer_batch_size, self.scorer_max_length) <= 0:
            raise ValueError("Smoke/scorer integer controls must be positive")
        if self.device not in self.DEVICES:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported model dtype")
        if self.bootstrap_samples < 100 or self.permutation_samples < 100:
            raise ValueError("Statistical resample counts are too small")
        if self.multiple_comparison_correction != "holm":
            raise ValueError("Stage 10 requires Holm multiple-comparison correction")
        allowed_metrics = {
            "hypervolume", "mip", "pcs", "preference_regret", "perplexity", "coherence"
        }
        if set(self.primary_metrics) != allowed_metrics:
            raise ValueError("Stage 10 primary metric set is incomplete")
        if any(left != "parm_v2_full_alpha" for left, _ in self.primary_comparisons):
            raise ValueError("Primary comparisons must use full-alpha as treatment")
        if any(left not in self.methods or right not in self.methods for left, right in self.primary_comparisons):
            raise ValueError("Primary comparison references an unknown method")
        if len(self.full_alpha_checkpoint_sha256) != 64:
            raise ValueError("Production checkpoint SHA256 is invalid")

    @property
    def alpha_grid(self) -> tuple[tuple[float, float], ...]:
        return tuple((value, 1.0 - value) for value in self.alpha_helpfulness_grid)

    def shuffled_alpha(self, index: int) -> tuple[float, float]:
        return self.alpha_grid[(index + self.shuffled_alpha_offset) % len(self.alpha_grid)]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("methods", "alpha_helpfulness_grid", "fixed_alpha", "hv_reference_point", "seeds", "primary_metrics"):
            values[name] = list(values[name])
        values["primary_comparisons"] = [list(value) for value in self.primary_comparisons]
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ParmTaroEvaluationConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Stage 10 config fields: {unknown}")
        normalized = dict(values)
        for name in ("methods", "alpha_helpfulness_grid", "fixed_alpha", "hv_reference_point", "seeds", "primary_metrics"):
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        if "primary_comparisons" in normalized:
            normalized["primary_comparisons"] = tuple(
                tuple(value) for value in normalized["primary_comparisons"]
            )
        return cls(**normalized)

    @classmethod
    def load_json(cls, path: str | Path) -> "ParmTaroEvaluationConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Stage 10 config must be a JSON object")
        return cls.from_dict(values)
