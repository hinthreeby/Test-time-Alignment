"""Validation-only calibration and immutable Stage 10 protocol locking."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PARM_TARO.evaluation.config import ParmTaroEvaluationConfig
from PARM_TARO.evaluation.io import canonical_json, write_json_atomic
from PARM_TARO.training.runtime import PROJECT_ROOT, project_path
from router_v2.cache.io import sha256_file
from router_v2.training.audit import hash_tree


PRODUCTION_CHECKPOINT_SHA256 = (
    "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20"
)
PARM_BASELINE_SHA256 = (
    "dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a"
)


def _quantile(values: Sequence[float], probability: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean or any(not math.isfinite(value) for value in clean):
        raise ValueError("Calibration values must be non-empty and finite")
    position = probability * (len(clean) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


@dataclass(frozen=True)
class ObjectiveNormalizer:
    """Shared clipped affine transform fit on validation references only."""

    helpfulness_low: float
    helpfulness_high: float
    harmlessness_low: float
    harmlessness_high: float

    def __post_init__(self) -> None:
        values = (
            self.helpfulness_low,
            self.helpfulness_high,
            self.harmlessness_low,
            self.harmlessness_high,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Normalization anchors must be finite")
        if self.helpfulness_high <= self.helpfulness_low:
            raise ValueError("Degenerate helpfulness normalization anchors")
        if self.harmlessness_high <= self.harmlessness_low:
            raise ValueError("Degenerate harmlessness normalization anchors")

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        return max(0.0, min(1.0, (float(value) - low) / (high - low)))

    def transform(self, helpfulness_raw: float, harmlessness_raw: float) -> tuple[float, float]:
        return (
            self._normalize(helpfulness_raw, self.helpfulness_low, self.helpfulness_high),
            self._normalize(harmlessness_raw, self.harmlessness_low, self.harmlessness_high),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "validation_quantile_minmax_clip",
            "objective_order": ["helpfulness", "harmlessness"],
            "anchors": {
                "helpfulness": {"low": self.helpfulness_low, "high": self.helpfulness_high},
                "harmlessness": {"low": self.harmlessness_low, "high": self.harmlessness_high},
            },
            "clip_range": [0.0, 1.0],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectiveNormalizer":
        if value.get("method") != "validation_quantile_minmax_clip":
            raise ValueError("Protocol lock has an unsupported normalizer")
        anchors = value["anchors"]
        return cls(
            helpfulness_low=float(anchors["helpfulness"]["low"]),
            helpfulness_high=float(anchors["helpfulness"]["high"]),
            harmlessness_low=float(anchors["harmlessness"]["low"]),
            harmlessness_high=float(anchors["harmlessness"]["high"]),
        )


def fit_normalizer(
    records: Sequence[Mapping[str, Any]],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> ObjectiveNormalizer:
    helpfulness = [float(row["helpfulness_raw"]) for row in records]
    harmlessness = [float(row["harmlessness_raw"]) for row in records]
    return ObjectiveNormalizer(
        helpfulness_low=_quantile(helpfulness, lower_quantile),
        helpfulness_high=_quantile(helpfulness, upper_quantile),
        harmlessness_low=_quantile(harmlessness, lower_quantile),
        harmlessness_high=_quantile(harmlessness, upper_quantile),
    )


def _file_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def protected_snapshot(config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    return {
        "router_v1": hash_tree(PROJECT_ROOT / "router"),
        "parm_original": hash_tree(PROJECT_ROOT / "PARM"),
        "method_rad": hash_tree(PROJECT_ROOT / "Method" / "RAD"),
        "stage9_checkpoints": {
            "taro": _file_descriptor(project_path(config.taro_checkpoint)),
            "v2_no_alpha": _file_descriptor(project_path(config.no_alpha_checkpoint)),
            "v2_full_alpha_production": _file_descriptor(project_path(config.full_alpha_checkpoint)),
        },
        "pblora": hash_tree(PROJECT_ROOT / "results" / "parm_taro" / "checkpoints" / "parm_pku_pblora"),
    }


def protocol_specification(config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    """Build the frozen spec without opening the final test payload."""

    manifest_path = project_path(config.data_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["splits"]["test"]["records"] != 1500:
        raise ValueError("Final Stage 10 test split must contain 1,500 prompts")
    if manifest["splits"]["validation"]["records"] != 500:
        raise ValueError("Stage 10 calibration split must contain 500 examples")
    full_checkpoint = project_path(config.full_alpha_checkpoint)
    actual_sha = sha256_file(full_checkpoint)
    if actual_sha != config.full_alpha_checkpoint_sha256 or actual_sha != PRODUCTION_CHECKPOINT_SHA256:
        raise ValueError("Stage 10 full-alpha checkpoint is not the production checkpoint")
    parm_tree = hash_tree(PROJECT_ROOT / "PARM")
    if parm_tree["tree_sha256"] != PARM_BASELINE_SHA256:
        raise ValueError("Original PARM tree differs from the frozen baseline")
    methods = {
        "parm_static": {"router": "fixed_lambda_1", "guide_alpha": "requested", "router_alpha": None},
        "parm_taro": {"router": config.taro_checkpoint, "guide_alpha": "requested", "router_alpha": None},
        "parm_v2_no_alpha": {"router": config.no_alpha_checkpoint, "guide_alpha": "requested", "router_alpha": None},
        "parm_v2_full_alpha": {"router": config.full_alpha_checkpoint, "guide_alpha": "requested", "router_alpha": "requested"},
        "parm_v2_full_alpha_same_average": {"router": config.same_average_lambda_policy, "guide_alpha": "requested", "router_alpha": None},
        "parm_v2_full_alpha_shuffled_alpha": {"router": config.full_alpha_checkpoint, "guide_alpha": "requested", "router_alpha": f"cyclic_grid_offset_{config.shuffled_alpha_offset}"},
        "parm_v2_full_alpha_fixed_alpha": {"router": config.full_alpha_checkpoint, "guide_alpha": "requested", "router_alpha": list(config.fixed_alpha)},
    }
    return {
        "schema_version": 1,
        "stage": 10,
        "status": "PROTOCOL_SPEC_FROZEN_CALIBRATION_PENDING",
        "test_payload_opened": False,
        "methods": methods,
        "method_order": list(config.methods),
        "evaluation": {
            "split": "test_prompt_only",
            "records": 1500,
            "manifest_declared_sha256": manifest["files"]["test_prompt_only.json"]["sha256"],
            "alpha_order": ["helpfulness", "harmlessness"],
            "alpha_grid": [list(value) for value in config.alpha_grid],
            "seeds": list(config.seeds),
        },
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "stop_on_eos": config.stop_on_eos,
            "prompt_format": "raw_processed_prompt_no_added_separator",
        },
        "objectives": {
            "raw_order": ["helpfulness_reward_high_better", "negative_cost_high_better"],
            "helpfulness_model": config.helpfulness_model_path,
            "harmlessness_cost_model": config.harmlessness_cost_model_path,
            "safe_rlhf_source": config.safe_rlhf_source_path,
            "normalization_fit_split": "validation",
            "normalization_fit_pool": "both_reference_responses",
            "normalization_quantiles": [config.normalization_lower_quantile, config.normalization_upper_quantile],
            "normalization_clip": [0.0, 1.0],
        },
        "pareto_hypervolume": {
            "maximize": ["helpfulness", "harmlessness"],
            "front_unit": "method_alpha_mean_over_prompts_and_seeds",
            "reference_point_normalized": list(config.hv_reference_point),
        },
        "preference_regret": {
            "oracle": config.regret_oracle,
            "candidate_pool": "all seven method outputs for identical prompt/requested-alpha/seed",
        },
        "statistics": {
            "paired_unit": "source_prompt_id",
            "bootstrap_samples": config.bootstrap_samples,
            "permutation_samples": config.permutation_samples,
            "seed": config.statistics_seed,
            "effect_size": "paired_cohen_dz",
            "multiple_comparison_correction": config.multiple_comparison_correction,
            "primary_comparisons": [list(value) for value in config.primary_comparisons],
            "primary_metrics": list(config.primary_metrics),
        },
        "provenance": {
            "config": config.to_dict(),
            "data_manifest": _file_descriptor(manifest_path),
            "production_checkpoint": _file_descriptor(full_checkpoint),
            "parm_tree": parm_tree,
        },
    }


def specification_sha256(specification: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(specification).encode("utf-8")).hexdigest()


def build_protocol_lock(
    config: ParmTaroEvaluationConfig,
    calibration_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    specification = protocol_specification(config)
    expected = 2 * int(json.loads((project_path(config.data_root) / "manifest.json").read_text())["splits"]["validation"]["records"])
    keys = {(str(row["sample_id"]), int(row["response_index"])) for row in calibration_records}
    if len(calibration_records) != expected or len(keys) != expected:
        raise ValueError("Calibration must score both responses for every validation example")
    normalizer = fit_normalizer(
        calibration_records,
        lower_quantile=config.normalization_lower_quantile,
        upper_quantile=config.normalization_upper_quantile,
    )
    lock = {
        **specification,
        "status": "PROTOCOL_LOCKED_TEST_NOT_USED",
        "specification_sha256": specification_sha256(specification),
        "normalization": {
            **normalizer.to_dict(),
            "fit_split": "validation",
            "fit_records": expected,
            "lower_quantile": config.normalization_lower_quantile,
            "upper_quantile": config.normalization_upper_quantile,
        },
        "objective_model_hashes": {
            "helpfulness": hash_tree(project_path(config.helpfulness_model_path)),
            "harmlessness_cost": hash_tree(project_path(config.harmlessness_cost_model_path)),
            "safe_rlhf_source": hash_tree(project_path(config.safe_rlhf_source_path)),
        },
    }
    lock["protocol_lock_sha256"] = specification_sha256(lock)
    return lock


def save_protocol_lock(path: str | Path, lock: Mapping[str, Any]) -> None:
    output = Path(path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != lock:
            raise FileExistsError("Refusing to overwrite a different Stage 10 protocol lock")
        return
    write_json_atomic(output, dict(lock))


def load_protocol_lock(path: str | Path, config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = lock.pop("protocol_lock_sha256", None)
    actual_hash = specification_sha256(lock)
    lock["protocol_lock_sha256"] = stored_hash
    if stored_hash != actual_hash:
        raise ValueError("Stage 10 protocol lock hash is invalid")
    if lock.get("status") != "PROTOCOL_LOCKED_TEST_NOT_USED":
        raise ValueError("Stage 10 protocol is not locked")
    current_spec = protocol_specification(config)
    if lock.get("specification_sha256") != specification_sha256(current_spec):
        raise ValueError("Stage 10 config/provenance differs from the protocol lock")
    current_objective_hashes = {
        "helpfulness": hash_tree(project_path(config.helpfulness_model_path)),
        "harmlessness_cost": hash_tree(project_path(config.harmlessness_cost_model_path)),
        "safe_rlhf_source": hash_tree(project_path(config.safe_rlhf_source_path)),
    }
    if lock.get("objective_model_hashes") != current_objective_hashes:
        raise ValueError("Objective scorer checkpoints differ from the protocol lock")
    ObjectiveNormalizer.from_dict(lock["normalization"])
    return lock
