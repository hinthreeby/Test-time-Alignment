"""Statistics for informative and structurally identical constant-alpha controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from PARM_TARO.training.alpha import validate_alpha


def alpha_vectors_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two canonical alpha vectors are exactly the same control."""

    left_values = validate_alpha(left).detach().cpu()
    right_values = validate_alpha(right).detach().cpu()
    if left_values.shape != (2,) or right_values.shape != (2,):
        raise ValueError("Constant-control comparison requires two alpha vectors")
    return bool(torch.equal(left_values, right_values))


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty constant-control group")
    values = values.double().reshape(-1)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p25": float(torch.quantile(values, 0.25)),
        "median": float(torch.quantile(values, 0.5)),
        "p75": float(torch.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def _bootstrap_mean_ci(
    task_means: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int] | None:
    if samples == 0:
        return None
    if samples < 1 or seed < 0:
        raise ValueError("Invalid bootstrap controls")
    values = task_means.double().reshape(-1)
    if values.numel() < 2:
        raise ValueError("Bootstrap requires at least two task means")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(),
        (samples, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return {
        "samples": samples,
        "seed": seed,
        "unit": "validation_task_mean",
        "confidence": 0.95,
        "lower": float(torch.quantile(means, 0.025)),
        "mean": float(means.mean()),
        "upper": float(torch.quantile(means, 0.975)),
    }


@dataclass
class ConstantControlAccumulator:
    """Retain one delta tensor per task so bootstrap resamples tasks, not tokens."""

    constant_alpha: torch.Tensor | None = None
    _records: list[tuple[float, bool, torch.Tensor]] = field(default_factory=list)

    def update(
        self,
        correct_alpha: torch.Tensor,
        constant_alpha: torch.Tensor,
        token_deltas: torch.Tensor,
    ) -> None:
        correct = validate_alpha(correct_alpha).detach().cpu()
        constant = validate_alpha(constant_alpha).detach().cpu()
        if correct.shape != (2,) or constant.shape != (2,):
            raise ValueError("Control accumulator requires individual alpha vectors")
        if self.constant_alpha is None:
            self.constant_alpha = constant.clone()
        elif not torch.equal(self.constant_alpha, constant):
            raise ValueError("Constant alpha changed within one evaluation")
        deltas = token_deltas.detach().cpu().float().reshape(-1)
        if deltas.numel() == 0 or not bool(torch.isfinite(deltas).all()):
            raise ValueError("Constant-control deltas must be finite and non-empty")
        if bool((deltas < 0.0).any()):
            raise ValueError("Constant-control deltas must be absolute differences")
        self._records.append(
            (float(correct[0]), alpha_vectors_identical(correct, constant), deltas)
        )

    def _group(self, records: list[tuple[float, bool, torch.Tensor]]) -> dict[str, Any]:
        if not records:
            return {
                "tasks": 0,
                "token_records": 0,
                "token_weighted": None,
                "task_mean_distribution": None,
            }
        tokens = torch.cat([record[2] for record in records])
        task_means = torch.stack([record[2].double().mean() for record in records])
        return {
            "tasks": len(records),
            "token_records": int(tokens.numel()),
            "token_weighted": _distribution(tokens),
            "task_mean_distribution": _distribution(task_means),
        }

    def finalize(
        self,
        *,
        threshold: float,
        bootstrap_samples: int = 0,
        bootstrap_seed: int = 2026,
    ) -> dict[str, Any]:
        if not self._records or self.constant_alpha is None:
            raise ValueError("No constant-control tasks were accumulated")
        if threshold <= 0.0:
            raise ValueError("Constant-control threshold must be positive")
        identical = [record for record in self._records if record[1]]
        informative = [record for record in self._records if not record[1]]
        if not informative:
            raise ValueError("No non-identical constant-control tasks exist")
        all_group = self._group(self._records)
        identical_group = self._group(identical)
        informative_group = self._group(informative)
        informative_task_means = torch.stack(
            [record[2].double().mean() for record in informative]
        )
        per_alpha = {}
        for helpfulness in sorted({record[0] for record in self._records}):
            records = [
                record for record in self._records if record[0] == helpfulness
            ]
            per_alpha[f"{helpfulness:.2f}"] = {
                "alpha": [helpfulness, 1.0 - helpfulness],
                "identical_to_constant": all(record[1] for record in records),
                **self._group(records),
            }
        non_identical_mean = float(
            informative_group["token_weighted"]["mean"]
        )
        return {
            "constant_alpha": [float(value) for value in self.constant_alpha],
            "total_tasks": len(self._records),
            "identical_tasks": len(identical),
            "non_identical_tasks": len(informative),
            "all_tasks": all_group,
            "identical_tasks_only": identical_group,
            "non_identical_tasks_only": {
                **informative_group,
                "bootstrap_mean_ci": _bootstrap_mean_ci(
                    informative_task_means,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            },
            "per_alpha_helpfulness": per_alpha,
            "gate": {
                "statistic": (
                    "token-weighted mean |lambda(correct)-lambda(constant)| "
                    "conditioned on correct_alpha != constant_alpha"
                ),
                "value": non_identical_mean,
                "threshold": threshold,
                "threshold_unchanged": threshold == 0.001,
                "pass": non_identical_mean >= threshold,
            },
        }
