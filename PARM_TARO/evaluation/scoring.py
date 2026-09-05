"""Official Beaver reward/cost scoring without silent proxy substitution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from PARM_TARO.evaluation.config import ParmTaroEvaluationConfig
from PARM_TARO.training.runtime import project_path


PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}"


def _activate_safe_rlhf(config: ParmTaroEvaluationConfig) -> bool:
    source = project_path(config.safe_rlhf_source_path)
    if not (source / "safe_rlhf" / "models").is_dir():
        return False
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)
    specification = importlib.util.find_spec("safe_rlhf")
    if specification is None or specification.origin is None:
        return False
    try:
        Path(specification.origin).resolve().relative_to(source.resolve())
    except ValueError:
        raise RuntimeError("safe_rlhf resolves outside the frozen Stage 10 source tree")
    return True


def scorer_prerequisites(config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    paths = {
        "helpfulness_model": project_path(config.helpfulness_model_path),
        "harmlessness_cost_model": project_path(config.harmlessness_cost_model_path),
        "safe_rlhf_source": project_path(config.safe_rlhf_source_path),
    }
    source_ready = _activate_safe_rlhf(config)
    auto_model_ready = False
    import_error = None
    if source_ready:
        try:
            from safe_rlhf.models import AutoModelForScore  # noqa: F401

            auto_model_ready = True
        except Exception as error:
            import_error = f"{type(error).__name__}: {error}"
    checks = {
        "safe_rlhf_source_local": source_ready,
        "safe_rlhf_auto_model_for_score_importable": auto_model_ready,
        "helpfulness_model_local": (paths["helpfulness_model"] / "config.json").is_file(),
        "harmlessness_cost_model_local": (paths["harmlessness_cost_model"] / "config.json").is_file(),
        "official_backend_selected": config.scorer_backend == "safe_rlhf_auto_model_for_score",
    }
    return {
        "checks": checks,
        "paths": {name: str(path) for name, path in paths.items()},
        "ready": all(checks.values()),
        "missing": [name for name, passed in checks.items() if not passed],
        "safe_rlhf_import_error": import_error,
        "prohibited_substitutions": [
            "pairwise source labels as generated-response scores",
            "toxic-bert as Beaver harmlessness cost",
            "helpfulness-deberta as the official PARM objective",
            "synthetic or repeated scalar objectives",
        ],
    }


def _dtype(name: str, device: torch.device) -> torch.dtype:
    value = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
    return torch.float32 if device.type == "cpu" else value


class BeaverObjectiveScorer:
    """Load one score model at a time so the 7B scorers never coexist."""

    def __init__(self, config: ParmTaroEvaluationConfig, device: torch.device) -> None:
        audit = scorer_prerequisites(config)
        if not audit["ready"]:
            raise RuntimeError(f"Stage 10 objective scorer prerequisites missing: {audit['missing']}")
        self.config = config
        self.device = device

    def _score_model(
        self,
        model_path: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> list[float]:
        from safe_rlhf.models import AutoModelForScore
        from transformers import AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": _dtype(self.config.model_dtype, self.device),
        }
        if self.device.type == "cuda" and self.config.scorer_load_in_4bit:
            kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=_dtype(self.config.model_dtype, self.device),
                    ),
                    "device_map": {"": self.device.index or 0},
                }
            )
        model = AutoModelForScore.from_pretrained(model_path, **kwargs).eval()
        if not (self.device.type == "cuda" and self.config.scorer_load_in_4bit):
            model = model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(records), self.config.scorer_batch_size):
                batch = records[start : start + self.config.scorer_batch_size]
                texts = [
                    PROMPT_TEMPLATE.format(
                        prompt=str(row["prompt"]), response=str(row["generated_text"])
                    )
                    for row in batch
                ]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.scorer_max_length,
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                output = model(**encoded)
                end_scores = output["end_scores"] if isinstance(output, dict) else output.end_scores
                values = end_scores.detach().float().reshape(len(batch), -1)
                if values.shape[1] != 1 or not bool(torch.isfinite(values).all()):
                    raise ValueError("Beaver scorer did not emit one finite end score per response")
                scores.extend(float(value) for value in values[:, 0].cpu().tolist())
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return scores

    def score(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        helpfulness = self._score_model(
            project_path(self.config.helpfulness_model_path), records
        )
        cost = self._score_model(
            project_path(self.config.harmlessness_cost_model_path), records
        )
        return [
            {
                **dict(row),
                "helpfulness_raw": help_score,
                "cost_raw": cost_score,
                "harmlessness_raw": -cost_score,
            }
            for row, help_score, cost_score in zip(records, helpfulness, cost)
        ]
