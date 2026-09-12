"""Streaming lambda and objective diagnostics for Stage 5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from router_v2.training.objective import RouterTrainingObjective


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2 or right.numel() != left.numel():
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(
        left_centered.square().sum() * right_centered.square().sum()
    )
    if float(denominator) == 0.0:
        return None
    return float((left_centered * right_centered).sum() / denominator)


@dataclass
class LambdaDiagnostics:
    lambda_max: float
    near_boundary_fraction: float = 0.05
    alpha_bins: int = 5
    lambda_values: list[torch.Tensor] = field(default_factory=list)
    positions: list[torch.Tensor] = field(default_factory=list)
    base_entropies: list[torch.Tensor] = field(default_factory=list)
    js_values: list[torch.Tensor] = field(default_factory=list)
    preferences: list[torch.Tensor] = field(default_factory=list)
    nll_sum: float = 0.0
    correct_count: int = 0
    token_count: int = 0
    total_loss_weighted: float = 0.0
    entropy_weighted: float = 0.0
    smoothness_weighted: float = 0.0
    strength_weighted: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive")
        if not 0.0 < self.near_boundary_fraction < 0.5:
            raise ValueError("near_boundary_fraction must be in (0, 0.5)")
        if self.alpha_bins <= 0:
            raise ValueError("alpha_bins must be positive")

    def update(
        self,
        objective: RouterTrainingObjective,
        lambda_t: torch.Tensor,
        valid_mask: torch.Tensor,
        position: torch.Tensor,
        base_entropy: torch.Tensor,
        js_divergence: torch.Tensor,
        preference: torch.Tensor | None = None,
    ) -> None:
        leading_shape = valid_mask.shape
        expected = leading_shape + (1,)
        if lambda_t.shape != expected:
            raise ValueError("lambda_t must align with valid_mask")
        for name, value in (
            ("position", position),
            ("base_entropy", base_entropy),
            ("js_divergence", js_divergence),
        ):
            if value.shape != leading_shape:
                raise ValueError(f"{name} must align with valid_mask")
        mask = valid_mask.detach().cpu().bool()
        self.lambda_values.append(lambda_t.detach().cpu()[..., 0][mask].float())
        self.positions.append(position.detach().cpu()[mask].long())
        self.base_entropies.append(
            base_entropy.detach().cpu()[mask].float()
        )
        self.js_values.append(js_divergence.detach().cpu()[mask].float())
        if preference is not None:
            if preference.shape[:-1] == leading_shape:
                expanded = preference
            elif preference.ndim == 2 and preference.shape[0] == leading_shape[0]:
                expanded = preference[:, None, :].expand(
                    leading_shape + (preference.shape[-1],)
                )
            else:
                raise ValueError("preference shape is incompatible with diagnostics")
            self.preferences.append(
                expanded.detach().cpu()[mask].float()
            )
        count = objective.token_count
        self.nll_sum += float(objective.nll_sum.detach())
        self.correct_count += int(objective.correct_count.detach())
        self.token_count += count
        self.total_loss_weighted += float(objective.total_loss.detach()) * count
        self.entropy_weighted += float(objective.entropy.detach()) * count
        self.smoothness_weighted += float(objective.smoothness.detach()) * count
        self.strength_weighted += float(objective.strength.detach()) * count

    def state_dict(self) -> dict[str, Any]:
        return {
            "lambda_max": self.lambda_max,
            "near_boundary_fraction": self.near_boundary_fraction,
            "alpha_bins": self.alpha_bins,
            "lambda_values": [value.clone() for value in self.lambda_values],
            "positions": [value.clone() for value in self.positions],
            "base_entropies": [value.clone() for value in self.base_entropies],
            "js_values": [value.clone() for value in self.js_values],
            "preferences": [value.clone() for value in self.preferences],
            "nll_sum": self.nll_sum,
            "correct_count": self.correct_count,
            "token_count": self.token_count,
            "total_loss_weighted": self.total_loss_weighted,
            "entropy_weighted": self.entropy_weighted,
            "smoothness_weighted": self.smoothness_weighted,
            "strength_weighted": self.strength_weighted,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LambdaDiagnostics":
        accumulator = cls(
            lambda_max=float(state["lambda_max"]),
            near_boundary_fraction=float(state["near_boundary_fraction"]),
            alpha_bins=int(state["alpha_bins"]),
        )
        for name in (
            "lambda_values",
            "positions",
            "base_entropies",
            "js_values",
            "preferences",
        ):
            setattr(
                accumulator,
                name,
                [value.detach().cpu().clone() for value in state[name]],
            )
        accumulator.nll_sum = float(state["nll_sum"])
        accumulator.correct_count = int(state["correct_count"])
        accumulator.token_count = int(state["token_count"])
        accumulator.total_loss_weighted = float(
            state["total_loss_weighted"]
        )
        accumulator.entropy_weighted = float(state["entropy_weighted"])
        accumulator.smoothness_weighted = float(state["smoothness_weighted"])
        accumulator.strength_weighted = float(state["strength_weighted"])
        return accumulator

    def finalize(self) -> dict[str, Any]:
        if self.token_count <= 0 or not self.lambda_values:
            raise ValueError("Cannot finalize empty diagnostics")
        values = torch.cat(self.lambda_values)
        positions = torch.cat(self.positions)
        base_entropy = torch.cat(self.base_entropies)
        js_divergence = torch.cat(self.js_values)
        quantiles = torch.quantile(
            values,
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]),
        )
        position_summary = []
        for position in sorted(set(positions.tolist())):
            selected = values[positions == position]
            position_summary.append(
                {
                    "position": int(position),
                    "count": int(selected.numel()),
                    "lambda_mean": float(selected.mean()),
                }
            )
        alpha_summary: dict[str, Any]
        if self.preferences:
            preference = torch.cat(self.preferences)
            dimensions = []
            for dimension in range(preference.shape[-1]):
                alpha = preference[:, dimension].clamp(0.0, 1.0)
                bucket_ids = torch.clamp(
                    (alpha * self.alpha_bins).long(),
                    max=self.alpha_bins - 1,
                )
                buckets = []
                for bucket in range(self.alpha_bins):
                    selected = values[bucket_ids == bucket]
                    buckets.append(
                        {
                            "bucket": bucket,
                            "lower": bucket / self.alpha_bins,
                            "upper": (bucket + 1) / self.alpha_bins,
                            "count": int(selected.numel()),
                            "lambda_mean": (
                                float(selected.mean())
                                if selected.numel()
                                else None
                            ),
                        }
                    )
                dimensions.append(
                    {"alpha_dimension": dimension, "buckets": buckets}
                )
            alpha_summary = {"status": "available", "dimensions": dimensions}
        else:
            alpha_summary = {"status": "not_applicable"}

        standard_deviation = (
            float(values.std(unbiased=False)) if values.numel() > 1 else 0.0
        )
        return {
            "token_count": self.token_count,
            "nll": self.nll_sum / self.token_count,
            "token_accuracy": self.correct_count / self.token_count,
            "total_loss": self.total_loss_weighted / self.token_count,
            "entropy": self.entropy_weighted / self.token_count,
            "smoothness": self.smoothness_weighted / self.token_count,
            "strength": self.strength_weighted / self.token_count,
            "lambda": {
                "mean": float(values.mean()),
                "std": standard_deviation,
                "min": float(values.min()),
                "max": float(values.max()),
                "p05": float(quantiles[0]),
                "p25": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p75": float(quantiles[3]),
                "p95": float(quantiles[4]),
                "fraction_near_zero": float(
                    (values <= self.near_boundary_fraction * self.lambda_max)
                    .float()
                    .mean()
                ),
                "fraction_near_max": float(
                    (
                        values
                        >= (1.0 - self.near_boundary_fraction)
                        * self.lambda_max
                    )
                    .float()
                    .mean()
                ),
            },
            "lambda_by_position": position_summary,
            "lambda_vs_base_entropy_pearson": _pearson(
                values,
                base_entropy,
            ),
            "lambda_vs_js_pearson": _pearson(values, js_divergence),
            "lambda_by_alpha_bucket": alpha_summary,
            "finite": all(
                math.isfinite(value)
                for value in (
                    self.nll_sum,
                    self.total_loss_weighted,
                    float(values.sum()),
                )
            ),
        }
