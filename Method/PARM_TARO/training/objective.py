"""Alpha-weighted dual-response token-level guided objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DualResponseObjective:
    total_loss: torch.Tensor
    response_nll: torch.Tensor
    response_weights: torch.Tensor
    weighted_token_nll_sum: torch.Tensor
    weighted_token_count: torch.Tensor


@dataclass(frozen=True)
class PreferenceSensitiveObjective:
    total_loss: torch.Tensor
    preference_loss: torch.Tensor
    quality_nll: DualResponseObjective
    strength_loss: torch.Tensor
    response_mean_log_likelihoods: torch.Tensor
    response_weight_difference: torch.Tensor
    weighted_preference_margin: torch.Tensor
    mean_lambda: torch.Tensor


def response_token_nll(
    guided_logprobs: torch.Tensor,
    gold_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if guided_logprobs.ndim != 3 or guided_logprobs.shape[0] != 1:
        raise ValueError("guided_logprobs must have shape [1, T, V]")
    if gold_token_ids.shape != (guided_logprobs.shape[1],):
        raise ValueError("gold_token_ids must have shape [T]")
    losses = F.nll_loss(
        guided_logprobs[0],
        gold_token_ids.to(guided_logprobs.device),
        reduction="none",
    )
    return losses.mean(), losses.sum()


def dual_response_objective(
    guided_logprobs: tuple[torch.Tensor, torch.Tensor],
    gold_token_ids: tuple[torch.Tensor, torch.Tensor],
    response_weights: torch.Tensor,
) -> DualResponseObjective:
    if response_weights.shape != (2,):
        raise ValueError("response_weights must have shape [2]")
    weights = response_weights.to(guided_logprobs[0].device)
    if bool((weights < 0).any()) or not torch.allclose(
        weights.sum(), torch.tensor(1.0, device=weights.device), atol=1e-6
    ):
        raise ValueError("response_weights must be a probability vector")
    means = []
    sums = []
    counts = []
    for logprobs, gold in zip(guided_logprobs, gold_token_ids):
        mean, total = response_token_nll(logprobs, gold)
        means.append(mean)
        sums.append(total)
        counts.append(gold.numel())
    response_nll = torch.stack(means)
    total_loss = (weights * response_nll).sum()
    count_tensor = torch.tensor(counts, dtype=weights.dtype, device=weights.device)
    weighted_sum = (
        weights * torch.stack(sums).to(dtype=weights.dtype)
    ).sum()
    weighted_count = (weights * count_tensor).sum()
    return DualResponseObjective(
        total_loss=total_loss,
        response_nll=response_nll,
        response_weights=weights,
        weighted_token_nll_sum=weighted_sum,
        weighted_token_count=weighted_count,
    )


def preference_sensitive_objective(
    guided_logprobs: tuple[torch.Tensor, torch.Tensor],
    gold_token_ids: tuple[torch.Tensor, torch.Tensor],
    response_weights: torch.Tensor,
    lambda_values: tuple[torch.Tensor, torch.Tensor],
    *,
    preference_loss_weight: float,
    nll_weight: float,
    strength_weight: float,
    strength_target: float,
    pairwise_logit_scale: float,
) -> PreferenceSensitiveObjective:
    """Combine alpha-sensitive pairwise ranking with quality preservation.

    Let ``m_alpha = q_0(alpha) - q_1(alpha)`` and let each response score be
    its mean guided gold-token log likelihood. The pairwise term is
    ``-log sigmoid(scale * m_alpha * (S_0 - S_1))``. At an exact preference
    tie ``m_alpha=0``, this term is constant and contributes no score gradient.
    """

    coefficients = (
        preference_loss_weight,
        nll_weight,
        strength_weight,
        strength_target,
        pairwise_logit_scale,
    )
    if any(value < 0.0 for value in coefficients[:-1]):
        raise ValueError("Preference objective weights/target must be non-negative")
    if pairwise_logit_scale <= 0.0:
        raise ValueError("pairwise_logit_scale must be positive")
    if preference_loss_weight == 0.0 and nll_weight == 0.0 and strength_weight == 0.0:
        raise ValueError("Preference objective must enable at least one term")
    quality = dual_response_objective(
        guided_logprobs,
        gold_token_ids,
        response_weights,
    )
    response_scores = -quality.response_nll
    weight_difference = quality.response_weights[0] - quality.response_weights[1]
    weighted_margin = weight_difference * (response_scores[0] - response_scores[1])
    preference_loss = F.softplus(-pairwise_logit_scale * weighted_margin)
    flattened_lambdas = torch.cat(
        [value.reshape(-1) for value in lambda_values]
    )
    if not bool(torch.isfinite(flattened_lambdas).all()):
        raise FloatingPointError("Router lambda values must be finite")
    mean_lambda = flattened_lambdas.mean()
    strength_loss = F.relu(
        mean_lambda.new_tensor(strength_target) - mean_lambda
    ).square()
    total = (
        preference_loss_weight * preference_loss
        + nll_weight * quality.total_loss
        + strength_weight * strength_loss
    )
    return PreferenceSensitiveObjective(
        total_loss=total,
        preference_loss=preference_loss,
        quality_nll=quality,
        strength_loss=strength_loss,
        response_mean_log_likelihoods=response_scores,
        response_weight_difference=weight_difference,
        weighted_preference_margin=weighted_margin,
        mean_lambda=mean_lambda,
    )
