"""Tensor-level diagnostics required before PARM-TARO recovery training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StaticEquivalenceResult:
    states: int
    max_abs_logprob_diff_lambda_0: float
    max_abs_prob_diff_lambda_0: float
    max_abs_logprob_diff_lambda_1: float
    max_abs_prob_diff_lambda_1: float
    selected_token_match_rate_lambda_0: float
    selected_token_match_rate_lambda_1: float


def _combined(base_logits: torch.Tensor, guide_logits: torch.Tensor, scale: float) -> torch.Tensor:
    base = F.log_softmax(base_logits.float(), dim=-1)
    guide = F.log_softmax(guide_logits.float(), dim=-1)
    return F.log_softmax(base + scale * guide, dim=-1)


def static_equivalence(
    base_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    adaptive_lambda_0: torch.Tensor,
    adaptive_lambda_1: torch.Tensor,
) -> StaticEquivalenceResult:
    """Compare adaptive endpoints with base and the static PARM equation."""

    if base_logits.shape != guide_logits.shape:
        raise ValueError("base and guide logits must have identical shapes")
    expected_lambda_shape = base_logits.shape[:-1] + (1,)
    if adaptive_lambda_0.shape != expected_lambda_shape or adaptive_lambda_1.shape != expected_lambda_shape:
        raise ValueError(f"adaptive endpoint tensors must have shape {expected_lambda_shape}")
    base_reference = F.log_softmax(base_logits.float(), dim=-1)
    static_reference = _combined(base_logits, guide_logits, 1.0)
    adaptive_zero = F.log_softmax(
        F.log_softmax(base_logits.float(), dim=-1)
        + adaptive_lambda_0.float() * F.log_softmax(guide_logits.float(), dim=-1),
        dim=-1,
    )
    adaptive_one = F.log_softmax(
        F.log_softmax(base_logits.float(), dim=-1)
        + adaptive_lambda_1.float() * F.log_softmax(guide_logits.float(), dim=-1),
        dim=-1,
    )

    def maximum(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).abs().max().item())

    states = base_logits.numel() // base_logits.shape[-1]
    match_zero = adaptive_zero.argmax(-1).eq(base_reference.argmax(-1)).float().mean()
    match_one = adaptive_one.argmax(-1).eq(static_reference.argmax(-1)).float().mean()
    return StaticEquivalenceResult(
        states=states,
        max_abs_logprob_diff_lambda_0=maximum(adaptive_zero, base_reference),
        max_abs_prob_diff_lambda_0=maximum(adaptive_zero.exp(), base_reference.exp()),
        max_abs_logprob_diff_lambda_1=maximum(adaptive_one, static_reference),
        max_abs_prob_diff_lambda_1=maximum(adaptive_one.exp(), static_reference.exp()),
        selected_token_match_rate_lambda_0=float(match_zero.item()),
        selected_token_match_rate_lambda_1=float(match_one.item()),
    )


def gold_token_utility(
    base_logprobs: torch.Tensor,
    guide_logprobs: torch.Tensor,
    gold_token_ids: torch.Tensor,
    lambdas: Iterable[float],
) -> dict[str, torch.Tensor]:
    """Return exact token NLL, ranks, and dL/dlambda at lambda zero.

    The analytic derivative is ``E_base[guide_logprob] - guide_gold``.
    Positive values mean gradient descent pushes lambda toward zero.
    """

    if base_logprobs.shape != guide_logprobs.shape:
        raise ValueError("base and guide log-probabilities must match")
    if gold_token_ids.shape != base_logprobs.shape[:-1]:
        raise ValueError("gold_token_ids must match the token-state dimensions")
    gold = gold_token_ids.long().unsqueeze(-1)
    base_gold = base_logprobs.gather(-1, gold).squeeze(-1)
    guide_gold = guide_logprobs.gather(-1, gold).squeeze(-1)
    base_rank = base_logprobs.gt(base_gold.unsqueeze(-1)).sum(-1) + 1
    guide_rank = guide_logprobs.gt(guide_gold.unsqueeze(-1)).sum(-1) + 1
    derivative = (base_logprobs.exp() * guide_logprobs).sum(-1) - guide_gold
    result = {
        "base_gold_logprob": base_gold,
        "guide_gold_logprob": guide_gold,
        "base_gold_rank": base_rank,
        "guide_gold_rank": guide_rank,
        "dL_dlambda_at_0": derivative,
    }
    for value in lambdas:
        scale = float(value)
        combined = F.log_softmax(base_logprobs + scale * guide_logprobs, dim=-1)
        result[f"nll_lambda_{scale:g}"] = -combined.gather(-1, gold).squeeze(-1)
    return result

