"""Strict Stage 9 training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True)
class ParmRouterTrainingConfig:
    """Configuration for one TARO/no-alpha/alpha training stage."""

    STAGES: ClassVar[tuple[str, ...]] = ("taro", "v2_no_alpha", "v2_alpha")
    DEVICES: ClassVar[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})
    MODEL_DTYPES: ClassVar[frozenset[str]] = frozenset(
        {"float16", "bfloat16", "float32"}
    )

    schema_version: int = 1
    method_label: str = "PARM_TARO_ROUTER_TRAINING"
    stage: str = "v2_alpha"
    router_kind: str = "smart"
    router_config_path: str = "PARM_TARO/configs/router_v2_alpha_tulu2.json"
    data_root: str = "dataset/parm_taro"
    base_model_path: str = "models/tulu-2-7b"
    tokenizer_path: str = "models/tulu-2-7b"
    parm_adapter_path: str = "results/parm_taro/checkpoints/parm_pku_pblora"
    output_dir: str = "results/parm_taro/training/v2_alpha"
    seed: int = 2026
    device: str = "auto"
    allow_cpu_fallback: bool = True
    model_dtype: str = "float16"
    model_placement: str = "accelerate_auto"
    share_base_backbone: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    cuda_memory_per_model_gib: int = 4
    max_length: int = 512
    max_continuation_tokens: int = 128
    include_eos_target: bool = True
    num_epochs: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    alpha_sampling: str = "uniform_simplex_hash"
    scalarization_rule: str = "alpha_weighted_response_mean_nll"
    alpha_order: tuple[str, ...] = ("helpfulness", "harmlessness")
    validation_alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    near_tie_threshold: float = 0.05
    static_reference_scale: float = 1.0
    min_control_lambda_delta: float = 1e-3
    require_shuffled_loss_degradation: bool = True
    enforce_stage_order: bool = True
    log_every_optimizer_steps: int = 10
    checkpoint_every_optimizer_steps: int = 250
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported Stage 9 training schema_version")
        if self.method_label != "PARM_TARO_ROUTER_TRAINING":
            raise ValueError("Invalid Stage 9 method_label")
        if self.stage not in self.STAGES:
            raise ValueError(f"Unsupported Stage 9 stage: {self.stage}")
        expected_kind = "taro" if self.stage == "taro" else "smart"
        if self.router_kind != expected_kind:
            raise ValueError(f"Stage {self.stage} requires router_kind={expected_kind}")
        if self.device not in self.DEVICES:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.model_dtype not in self.MODEL_DTYPES:
            raise ValueError("Unsupported model_dtype")
        if self.model_placement not in {"accelerate_auto", "single_device"}:
            raise ValueError("Unsupported model_placement")
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("bnb_4bit_quant_type must be nf4 or fp4")
        if self.cuda_memory_per_model_gib <= 0:
            raise ValueError("cuda_memory_per_model_gib must be positive")
        positive_ints = {
            "max_length": self.max_length,
            "max_continuation_tokens": self.max_continuation_tokens,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "log_every_optimizer_steps": self.log_every_optimizer_steps,
            "checkpoint_every_optimizer_steps": self.checkpoint_every_optimizer_steps,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Stage 9 integer fields must be positive: {invalid}")
        if self.max_continuation_tokens >= self.max_length:
            raise ValueError("max_continuation_tokens must be below max_length")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.alpha_sampling != "uniform_simplex_hash":
            raise ValueError("Unsupported alpha_sampling")
        if self.scalarization_rule != "alpha_weighted_response_mean_nll":
            raise ValueError("Unsupported scalarization_rule")
        if self.alpha_order != ("helpfulness", "harmlessness"):
            raise ValueError("Canonical alpha_order must be helpfulness, harmlessness")
        if not self.validation_alpha_grid:
            raise ValueError("validation_alpha_grid cannot be empty")
        if any(value < 0.0 or value > 1.0 for value in self.validation_alpha_grid):
            raise ValueError("Validation alpha values must be in [0, 1]")
        if tuple(sorted(set(self.validation_alpha_grid))) != self.validation_alpha_grid:
            raise ValueError("validation_alpha_grid must be sorted and unique")
        if not 0.0 <= self.near_tie_threshold < 0.5:
            raise ValueError("near_tie_threshold must be in [0, 0.5)")
        if self.static_reference_scale < 0.0:
            raise ValueError("static_reference_scale must be non-negative")
        if self.min_control_lambda_delta <= 0.0:
            raise ValueError("min_control_lambda_delta must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name, value in (
            ("max_train_samples", self.max_train_samples),
            ("max_validation_samples", self.max_validation_samples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null")

    @property
    def predecessor_stage(self) -> str | None:
        index = self.STAGES.index(self.stage)
        return None if index == 0 else self.STAGES[index - 1]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["alpha_order"] = list(self.alpha_order)
        values["validation_alpha_grid"] = list(self.validation_alpha_grid)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ParmRouterTrainingConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Stage 9 config fields: {unknown}")
        normalized = dict(values)
        for name in ("alpha_order", "validation_alpha_grid"):
            if name in normalized:
                if not isinstance(normalized[name], (list, tuple)):
                    raise TypeError(f"{name} must be a JSON array")
                normalized[name] = tuple(normalized[name])
        if "validation_alpha_grid" in normalized:
            normalized["validation_alpha_grid"] = tuple(
                float(value) for value in normalized["validation_alpha_grid"]
            )
        return cls(**normalized)

    @classmethod
    def load_json(cls, path: str | Path) -> "ParmRouterTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Stage 9 config must be a JSON object")
        return cls.from_dict(values)
