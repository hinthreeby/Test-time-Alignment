"""Configuration contract for the isolated sentiment guide."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


GUIDE_CONFIG_SCHEMA_VERSION = 1
GUIDE_TASK = "sentiment"
GUIDE_OBJECTIVE = "positive_continuation_causal_nll"


@dataclass(frozen=True)
class SentimentGuideConfig:
    schema_version: int = GUIDE_CONFIG_SCHEMA_VERSION
    task: str = GUIDE_TASK
    objective: str = GUIDE_OBJECTIVE
    base_model_path: str = "models/gpt2-medium"
    train_data_path: str = (
        "dataset/RAD_train/router_amazon_polarity/train.jsonl"
    )
    validation_data_path: str = (
        "dataset/RAD_train/router_amazon_polarity/validation.jsonl"
    )
    negative_audit_data_path: str = (
        "dataset/RAD_train/amazon_polarity/test/data-00000-of-00001.arrow"
    )
    output_dir: str = "results/router_v2/sentiment_guide"
    seed: int = 42
    max_length: int = 160
    include_eos_target: bool = True
    num_train_epochs: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn", "c_proj", "c_fc")
    gradient_checkpointing: bool = True
    precision: str = "fp16"
    save_optimizer_steps: int = 250
    log_optimizer_steps: int = 10
    validation_positive_samples: int = 100
    validation_negative_samples: int = 100
    validation_inspection_samples: int = 20
    minimum_positive_nll_improvement: float = 0.01
    minimum_direction_delta_gap: float = 0.01
    cache_top_k: int = 20
    cache_max_position: int = 79
    cache_shard_size: int = 4096

    def __post_init__(self) -> None:
        if self.schema_version != GUIDE_CONFIG_SCHEMA_VERSION:
            raise ValueError("Unsupported sentiment guide config schema_version")
        if self.task != GUIDE_TASK:
            raise ValueError("Sentiment guide task must be 'sentiment'")
        if self.objective != GUIDE_OBJECTIVE:
            raise ValueError(
                "Sentiment guide objective must be positive continuation NLL"
            )
        if self.seed < 0 or self.max_length < 2:
            raise ValueError("seed and max_length are invalid")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must be non-empty")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        if self.save_optimizer_steps <= 0 or self.log_optimizer_steps <= 0:
            raise ValueError("save/log steps must be positive")
        if self.validation_positive_samples < 20:
            raise ValueError("validation_positive_samples must be at least 20")
        if self.validation_negative_samples < 20:
            raise ValueError("validation_negative_samples must be at least 20")
        if self.validation_inspection_samples < 1:
            raise ValueError("validation_inspection_samples must be positive")
        if self.minimum_positive_nll_improvement <= 0:
            raise ValueError(
                "minimum_positive_nll_improvement must be positive"
            )
        if self.minimum_direction_delta_gap <= 0:
            raise ValueError("minimum_direction_delta_gap must be positive")
        if self.cache_top_k <= 0:
            raise ValueError("cache_top_k must be positive")
        if self.cache_max_position <= 0:
            raise ValueError("cache_max_position must be positive")
        if self.cache_shard_size <= self.cache_max_position:
            raise ValueError(
                "cache_shard_size must hold at least one complete continuation"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SentimentGuideConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown sentiment guide config keys: {unknown}")
        values = dict(payload)
        if "lora_target_modules" in values:
            values["lora_target_modules"] = tuple(
                str(value) for value in values["lora_target_modules"]
            )
        return cls(**values)

    @classmethod
    def load_json(cls, path: str | Path) -> "SentimentGuideConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Sentiment guide config must contain an object")
        return cls.from_dict(payload)
