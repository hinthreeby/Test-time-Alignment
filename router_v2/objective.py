"""Gold-token NLL and optional Bernoulli entropy for TARO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from router_v2.config import TARORouterConfig
from router_v2.model import TARORouterOutput


@dataclass(frozen=True)
class TAROObjective:
    total_loss: torch.Tensor
    nll_loss: torch.Tensor
    entropy: torch.Tensor
    entropy_contribution: torch.Tensor
    token_count: int


def bernoulli_entropy(alpha: torch.Tensor, *, eps: float) -> torch.Tensor:
    bounded_alpha = alpha.clamp(min=eps, max=1.0 - eps)
    return -(
        bounded_alpha * bounded_alpha.log()
        + (1.0 - bounded_alpha) * (1.0 - bounded_alpha).log()
    )


def compute_taro_objective(
    output: TARORouterOutput,
    gold_token_ids: torch.Tensor,
    config: TARORouterConfig,
) -> TAROObjective:
    """Compute exact gold NLL over the logits emitted by the TARO router."""

    if gold_token_ids.shape != output.guided_logits.shape[:-1]:
        raise ValueError("gold_token_ids must match the leading guided-logit dimensions")
    if output.alpha.shape != gold_token_ids.shape + (1,):
        raise ValueError("alpha and gold_token_ids have incompatible shapes")
    flat_logits = output.guided_logits.reshape(-1, output.guided_logits.shape[-1])
    flat_targets = gold_token_ids.to(device=flat_logits.device, dtype=torch.long).reshape(-1)
    nll_loss = functional.cross_entropy(
        flat_logits,
        flat_targets,
        reduction=config.loss_reduction,
    )
    entropy_values = bernoulli_entropy(output.alpha, eps=config.alpha_eps)
    if config.loss_reduction == "sum":
        entropy = entropy_values.sum()
    else:
        entropy = entropy_values.mean()
    entropy_term = (
        config.entropy_weight * entropy
        if config.mode == "taro_topk_nll_entropy"
        else entropy.new_zeros(())
    )
    return TAROObjective(
        total_loss=nll_loss + entropy_term,
        nll_loss=nll_loss,
        entropy=entropy,
        entropy_contribution=entropy_term,
        token_count=flat_targets.numel(),
    )
