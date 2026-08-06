"""Feature normalization and guided-score utilities for the RAD token router."""

from __future__ import annotations

import torch


def _assert_candidate_tensor(name: str, values: torch.Tensor, top_k: int) -> None:
    assert isinstance(values, torch.Tensor), f"{name} must be a torch.Tensor"
    assert values.ndim >= 2, f"{name} must have shape [..., {top_k}], got {tuple(values.shape)}"
    assert values.shape[-1] == top_k, f"{name} last dimension must be {top_k}, got {values.shape[-1]}"
    assert torch.is_floating_point(values), f"{name} must be floating point"
    assert torch.isfinite(values).all(), f"{name} contains NaN or infinity"


def standardize_candidate_scores(values: torch.Tensor, top_k: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Standardize candidate scores along the candidate dimension."""
    _assert_candidate_tensor("values", values, top_k)
    detached = values.detach().float()
    mean = detached.mean(dim=-1, keepdim=True)
    std = detached.std(dim=-1, keepdim=True, unbiased=False)
    normalized = (detached - mean) / (std + eps)
    return torch.where(std > eps, normalized, torch.zeros_like(normalized))


def build_router_features(
    base_logits: torch.Tensor,
    reward_scores: torch.Tensor,
    top_k: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build normalized 40-dim router features from raw 20-dim logits and rewards."""
    _assert_candidate_tensor("base_logits", base_logits, top_k)
    _assert_candidate_tensor("reward_scores", reward_scores, top_k)
    assert base_logits.shape == reward_scores.shape, (
        f"base_logits and reward_scores shapes must match, got {tuple(base_logits.shape)} "
        f"and {tuple(reward_scores.shape)}"
    )
    base_features = standardize_candidate_scores(base_logits, top_k=top_k, eps=eps)
    reward_features = standardize_candidate_scores(reward_scores, top_k=top_k, eps=eps)
    return torch.cat([base_features, reward_features], dim=-1)


def compute_guided_scores(
    base_logits: torch.Tensor,
    reward_scores: torch.Tensor,
    beta: torch.Tensor,
    top_k: int = 20,
) -> torch.Tensor:
    """Compute RAD guided scores from raw cached values, separate from normalization."""
    _assert_candidate_tensor("base_logits", base_logits, top_k)
    _assert_candidate_tensor("reward_scores", reward_scores, top_k)
    assert base_logits.shape == reward_scores.shape, (
        f"base_logits and reward_scores shapes must match, got {tuple(base_logits.shape)} "
        f"and {tuple(reward_scores.shape)}"
    )
    assert isinstance(beta, torch.Tensor), "beta must be a torch.Tensor"
    assert beta.shape[-1:] == (1,), f"beta must have trailing singleton dim, got {tuple(beta.shape)}"
    assert beta.shape[:-1] == base_logits.shape[:-1], (
        f"beta leading shape {tuple(beta.shape[:-1])} must match candidate leading shape "
        f"{tuple(base_logits.shape[:-1])}"
    )
    assert torch.isfinite(beta).all(), "beta contains NaN or infinity"
    return base_logits.detach().float() + beta.float() * reward_scores.detach().float()
