"""Scalarizations for the optional, authorization-gated matched tiny A/B."""
from __future__ import annotations
import torch

def validate_alpha(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.shape[-1] != 2 or bool((alpha < 0).any()) or not torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6):
        raise ValueError("alpha must be a two-objective simplex vector")
    return alpha

def linear_scalarization(losses: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Original PARM scalarization, with [helpfulness, harmlessness] order."""
    return (losses * validate_alpha(alpha)).sum(-1)

def augmented_tchebycheff(losses: torch.Tensor, alpha: torch.Tensor, *, reference: torch.Tensor,
                          scales: torch.Tensor, augmentation: float = 0.05) -> torch.Tensor:
    """Smooth augmented Tchebycheff minimization using logsumexp as smooth max.

    User alpha maps identically to objective weights: alpha=(1,0) retains only
    helpfulness and alpha=(0,1) only harmlessness. This is the unique simple
    mapping that preserves endpoints, the midpoint, objective order, and
    alpha's stated meaning without introducing a tuned preference warp.
    Reference and scales must be frozen from training-only pilot statistics.
    """
    weights = validate_alpha(alpha); normalized = (losses-reference) / scales.clamp_min(1e-8)
    weighted = weights * normalized
    smooth_max = torch.logsumexp(20.0 * weighted, dim=-1) / 20.0
    return smooth_max + augmentation * weighted.sum(-1)
