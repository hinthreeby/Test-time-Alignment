"""Validated Stage 6 RAD evaluation configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True)
class RADEvaluationConfig:
    DEVICES: ClassVar[frozenset[str]] = frozenset({"auto", "cuda", "cpu"})
    PROMPT_CLASSES: ClassVar[frozenset[str]] = frozenset(
        {"negative", "neutral", "positive"}
    )
    V2_ROUTER_METHODS: ClassVar[tuple[str, ...]] = (
        "taro",
        "v2_state",
        "v2_history",
    )

    schema_version: int = 1
    output_dir: str = "results/router_v2/rad/full"
    dataset_dir: str = "dataset/rad_benchmark"
    base_model_path: str = "models/gpt2-medium"
    guide_model_path: str = "results/router_v2/sentiment_guide/final_adapter"
    sentiment_classifier_path: str = "models/sentiment-roberta-large-english"
    taro_checkpoint_path: str = "results/router_v2/training/taro/best.pt"
    state_checkpoint_path: str = "results/router_v2/training/state/best.pt"
    history_checkpoint_path: str = "results/router_v2/training/history/best.pt"
    v1_results_dir: str = "router/evaluation/report_run_32tokens"
    prompt_classes: tuple[str, ...] = ("negative", "neutral", "positive")
    validation_prompt_limit_per_class: int = 50
    test_prompt_limit_per_class: int = 200
    test_prompt_offset_per_class: int = 50
    seeds: tuple[int, ...] = (1,)
    fixed_lambda_sweep: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    heuristic_methods: tuple[str, ...] = (
        "heuristic_entropy_gap",
        "heuristic_js",
    )
    max_new_tokens: int = 32
    top_k: int = 20
    temperature: float = 1.0
    do_sample: bool = False
    stop_on_eos: bool = False
    device: str = "auto"
    allow_cpu_fallback: bool = True
    precision: str = "fp32"
    sentiment_classifier_device: str = "cpu"
    sentiment_classifier_batch_size: int = 16
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 12345
    log_every_jobs: int = 10

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported RAD evaluation schema: {self.schema_version}"
            )
        if not self.output_dir.startswith("results/router_v2/rad/"):
            raise ValueError(
                "Stage 6 output_dir must be under results/router_v2/rad/"
            )
        if not self.prompt_classes:
            raise ValueError("prompt_classes cannot be empty")
        if len(set(self.prompt_classes)) != len(self.prompt_classes):
            raise ValueError("prompt_classes cannot contain duplicates")
        invalid_classes = sorted(set(self.prompt_classes) - self.PROMPT_CLASSES)
        if invalid_classes:
            raise ValueError(f"Unsupported prompt classes: {invalid_classes}")
        positive_ints = {
            "validation_prompt_limit_per_class": (
                self.validation_prompt_limit_per_class
            ),
            "test_prompt_limit_per_class": self.test_prompt_limit_per_class,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "bootstrap_samples": self.bootstrap_samples,
            "sentiment_classifier_batch_size": (
                self.sentiment_classifier_batch_size
            ),
            "log_every_jobs": self.log_every_jobs,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Evaluation integer fields must be positive: {invalid}")
        if self.test_prompt_offset_per_class < 0:
            raise ValueError("test_prompt_offset_per_class must be non-negative")
        if self.max_new_tokens < 32:
            raise ValueError("Scientific Stage 6 requires at least 32 new tokens")
        if self.max_new_tokens > 80:
            raise ValueError(
                "Stage 6 Smart checkpoints support at most 80 positions"
            )
        if self.top_k != 20:
            raise ValueError("Stage 6 checkpoints require top_k=20")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if not self.fixed_lambda_sweep:
            raise ValueError("fixed_lambda_sweep cannot be empty")
        if len(set(self.fixed_lambda_sweep)) != len(self.fixed_lambda_sweep):
            raise ValueError("fixed_lambda_sweep cannot contain duplicates")
        if any(not 0.0 <= value <= 1.0 for value in self.fixed_lambda_sweep):
            raise ValueError("Fixed lambda values must be in [0, 1]")
        allowed_heuristics = {"heuristic_entropy_gap", "heuristic_js"}
        if not self.heuristic_methods or not set(self.heuristic_methods) <= allowed_heuristics:
            raise ValueError("Unsupported or empty heuristic_methods")
        if len(set(self.heuristic_methods)) != len(self.heuristic_methods):
            raise ValueError("heuristic_methods cannot contain duplicates")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.device not in self.DEVICES:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.sentiment_classifier_device not in self.DEVICES:
            raise ValueError(
                f"Unsupported classifier device: {self.sentiment_classifier_device}"
            )
        if self.precision != "fp32":
            raise ValueError("Scientific Stage 6 requires precision='fp32'")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for name in (
            "prompt_classes",
            "seeds",
            "fixed_lambda_sweep",
            "heuristic_methods",
        ):
            values[name] = list(values[name])
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RADEvaluationConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown RAD evaluation config fields: {unknown}")
        converted = dict(values)
        for name in (
            "prompt_classes",
            "seeds",
            "fixed_lambda_sweep",
            "heuristic_methods",
        ):
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted)

    @classmethod
    def load_json(cls, path: str | Path) -> "RADEvaluationConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("RAD evaluation config must be a JSON object")
        return cls.from_dict(value)
