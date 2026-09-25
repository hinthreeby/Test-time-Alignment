"""
DynamicPreferenceController — MATO-style entropic mirror descent.

Updates alpha_t based on accumulated per-objective rewards.
Applies smoothing, trust-region (KL budget), and simplex projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class AlphaUpdateInfo:
    alpha_candidate: List[float]
    alpha_smoothed: List[float]
    alpha_t: List[float]
    kl_to_user: float
    l1_shift: float
    dominant_objective: int
    deficit_vector: List[float]
    clipped_by_trust_region: bool


def _to_tensor(x: List[float], device: str = "cpu") -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64, device=device)


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """KL(p || q)"""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p / q).log()).sum().item()


def _project_to_kl_ball(
    alpha: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    eps: float = 1e-8,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Project alpha onto the intersection of:
      - K-simplex
      - KL(alpha || center) <= radius
    via binary search on the KL constraint.
    """
    if _kl_div(alpha, center, eps) <= radius:
        return alpha

    # Binary search on temperature t: alpha_proj = softmax(log(alpha) + t*log(center))
    lo, hi = 0.0, 1.0
    # Expand hi until KL is within budget
    while True:
        blended = (1 - hi) * alpha + hi * center
        blended = blended / blended.sum()
        if _kl_div(blended, center, eps) <= radius:
            break
        hi = (hi + 1.0) / 2.0
        if hi > 1.0 - 1e-9:
            return center.clone()

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        blended = (1 - mid) * alpha + mid * center
        blended = blended / blended.sum()
        if _kl_div(blended, center, eps) <= radius:
            hi = mid
        else:
            lo = mid

    blended = (1 - hi) * alpha + hi * center
    return (blended / blended.sum()).clamp(min=eps)


def normalize_rewards(
    cumulative_rewards: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize per-objective accumulated rewards for scale comparability."""
    if running_mean is None or running_var is None:
        return cumulative_rewards
    std = (running_var + eps).sqrt()
    return (cumulative_rewards - running_mean) / std


class DynamicPreferenceController:
    """
    Updates alpha_t via MATO-inspired entropic mirror descent.

    Closed-form update:
        alpha_candidate_k ∝ alpha_user_k * exp(-R_k / tau)

    Applies EMA smoothing, then projects onto KL trust-region.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        smoothing: float = 0.25,
        kl_budget: float = 0.1,
        eps: float = 1e-8,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if kl_budget < 0:
            raise ValueError("kl_budget must be non-negative")
        self.temperature = temperature
        self.smoothing = smoothing
        self.kl_budget = kl_budget
        self.eps = eps

    def update(
        self,
        alpha_user: List[float],
        alpha_previous: List[float],
        accumulated_rewards: List[float],
        running_mean: Optional[List[float]] = None,
        running_var: Optional[List[float]] = None,
        uncertainty: Optional[float] = None,
    ) -> tuple[List[float], AlphaUpdateInfo]:
        """
        Return updated alpha_t and diagnostic info.

        Args:
            alpha_user: Original user preference (prior).
            alpha_previous: Last alpha_t.
            accumulated_rewards: Per-objective cumulative reward so far.
            running_mean / running_var: Normalization stats (from train/val).
            uncertainty: Tracker uncertainty; if too high, stay close to alpha_previous.
        """
        if not alpha_user or len(alpha_user) != len(alpha_previous) or len(alpha_user) != len(accumulated_rewards):
            raise ValueError("alpha and reward vectors must have the same non-zero length")
        if any(value < 0 or not math.isfinite(value) for value in alpha_user):
            raise ValueError("alpha_user must contain finite non-negative values")
        if any(value < 0 or not math.isfinite(value) for value in alpha_previous):
            raise ValueError("alpha_previous must contain finite non-negative values")
        if not math.isclose(sum(alpha_user), 1.0, abs_tol=1e-6):
            raise ValueError("alpha_user must sum to 1")
        if not math.isclose(sum(alpha_previous), 1.0, abs_tol=1e-6):
            raise ValueError("alpha_previous must sum to 1")
        if any(not math.isfinite(value) for value in accumulated_rewards):
            raise ValueError("accumulated_rewards must be finite")
        alpha_u = _to_tensor(alpha_user)
        alpha_prev = _to_tensor(alpha_previous)
        R = _to_tensor(accumulated_rewards)

        # Normalize reward scale
        mean_t = _to_tensor(running_mean) if running_mean else None
        var_t = _to_tensor(running_var) if running_var else None
        R_norm = normalize_rewards(R, mean_t, var_t, self.eps)

        # Entropic mirror descent: alpha_candidate_k ∝ alpha_user_k * exp(-R_k / tau)
        log_alpha_user = (alpha_u + self.eps).log()
        logits = log_alpha_user - R_norm / self.temperature
        alpha_candidate = F.softmax(logits.float(), dim=-1).double()

        # EMA smoothing
        eff_smoothing = self.smoothing
        if uncertainty is not None:
            # High uncertainty → conservative update
            eff_smoothing = self.smoothing * max(0.0, 1.0 - float(uncertainty))

        alpha_smoothed = (1.0 - eff_smoothing) * alpha_prev + eff_smoothing * alpha_candidate
        alpha_smoothed = (alpha_smoothed / alpha_smoothed.sum()).clamp(min=self.eps)

        # Trust-region projection
        alpha_clipped = _project_to_kl_ball(alpha_smoothed, alpha_u, self.kl_budget, self.eps)
        clipped = not torch.allclose(alpha_clipped, alpha_smoothed, atol=1e-6)

        kl = _kl_div(alpha_clipped, alpha_u, self.eps)
        l1_shift = (alpha_clipped - alpha_prev).abs().sum().item()
        deficit_vector = (alpha_u - alpha_clipped).tolist()
        dominant = int(alpha_clipped.argmax().item())

        info = AlphaUpdateInfo(
            alpha_candidate=alpha_candidate.tolist(),
            alpha_smoothed=alpha_smoothed.tolist(),
            alpha_t=alpha_clipped.tolist(),
            kl_to_user=kl,
            l1_shift=l1_shift,
            dominant_objective=dominant,
            deficit_vector=deficit_vector,
            clipped_by_trust_region=clipped,
        )

        return alpha_clipped.tolist(), info
