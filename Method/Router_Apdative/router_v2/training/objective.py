"""Gold-token Stage 5 objective and optional router regularizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RouterTrainingObjective:
    total_loss: torch.Tensor
    nll_loss: torch.Tensor
    entropy: torch.Tensor
    smoothness: torch.Tensor
    strength: torch.Tensor
    entropy_contribution: torch.Tensor
    smoothness_contribution: torch.Tensor
    strength_contribution: torch.Tensor
    nll_sum: torch.Tensor
    correct_count: torch.Tensor
    token_count: int


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    if int(count) <= 0:
        raise ValueError("Objective mask must contain at least one token")
    return (values * mask.to(dtype=values.dtype)).sum() / count


def compute_router_training_objective(
    guided_logits: torch.Tensor,
    gold_token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    gate: torch.Tensor,
    lambda_t: torch.Tensor,
    *,
    lambda_max: float,
    entropy_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    strength_weight: float = 0.0,
    eps: float = 1e-6,
) -> RouterTrainingObjective:
    if guided_logits.ndim != 3:
        raise ValueError("guided_logits must have shape [B, T, V]")
    leading_shape = guided_logits.shape[:-1]
    if gold_token_ids.shape != leading_shape or valid_mask.shape != leading_shape:
        raise ValueError("Gold IDs and valid_mask must have shape [B, T]")
    if gate.shape != leading_shape + (1,) or lambda_t.shape != gate.shape:
        raise ValueError("gate and lambda_t must have shape [B, T, 1]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if lambda_max <= 0.0:
        raise ValueError("lambda_max must be positive")
    if any(
        weight < 0.0
        for weight in (entropy_weight, smoothness_weight, strength_weight)
    ):
        raise ValueError("Objective weights must be non-negative")

    per_token_nll = functional.cross_entropy(
        guided_logits.reshape(-1, guided_logits.shape[-1]),
        gold_token_ids.reshape(-1),
        reduction="none",
    ).reshape(leading_shape)
    nll_loss = _masked_mean(per_token_nll, valid_mask)
    nll_sum = (per_token_nll * valid_mask).sum()
    predictions = guided_logits.argmax(dim=-1)
    correct_count = (
        predictions.eq(gold_token_ids) & valid_mask
    ).sum()

    clipped_gate = gate.squeeze(-1).clamp(eps, 1.0 - eps)
    entropy_values = -(
        clipped_gate * clipped_gate.log()
        + (1.0 - clipped_gate) * (1.0 - clipped_gate).log()
    )
    entropy = _masked_mean(entropy_values, valid_mask)
    normalized_strength = lambda_t.squeeze(-1) / float(lambda_max)
    strength = _masked_mean(normalized_strength.square(), valid_mask)

    if guided_logits.shape[1] < 2:
        smoothness = lambda_t.sum() * 0.0
    else:
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if bool(pair_mask.any()):
            differences = (
                lambda_t[:, 1:, 0] - lambda_t[:, :-1, 0]
            ).square()
            smoothness = _masked_mean(differences, pair_mask)
        else:
            smoothness = lambda_t.sum() * 0.0

    entropy_contribution = entropy_weight * entropy
    smoothness_contribution = smoothness_weight * smoothness
    strength_contribution = strength_weight * strength
    total_loss = (
        nll_loss
        + entropy_contribution
        + smoothness_contribution
        + strength_contribution
    )
    return RouterTrainingObjective(
        total_loss=total_loss,
        nll_loss=nll_loss,
        entropy=entropy,
        smoothness=smoothness,
        strength=strength,
        entropy_contribution=entropy_contribution,
        smoothness_contribution=smoothness_contribution,
        strength_contribution=strength_contribution,
        nll_sum=nll_sum,
        correct_count=correct_count,
        token_count=int(valid_mask.sum()),
    )
