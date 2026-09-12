"""Independent Top-K candidate selection for faithful TARO routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TAROTopKBatch:
    """Separate base and reward Top-K token/logit pairs."""

    base_token_ids: torch.Tensor
    base_logits: torch.Tensor
    reward_token_ids: torch.Tensor
    reward_logits: torch.Tensor


def select_taro_topk(
    base_logits: torch.Tensor,
    reward_logits: torch.Tensor,
    *,
    top_k: int,
) -> TAROTopKBatch:
    """Select base and reward Top-K sets independently, without gold targets."""

    if base_logits.shape != reward_logits.shape:
        raise ValueError("base_logits and reward_logits must have identical shapes")
    if base_logits.ndim < 1:
        raise ValueError("logits must have shape (..., vocab_size)")
    if base_logits.device != reward_logits.device:
        raise ValueError("base_logits and reward_logits must be on the same device")
    vocab_size = base_logits.shape[-1]
    if not 1 <= top_k <= vocab_size:
        raise ValueError("top_k must be in [1, vocab_size]")

    detached_base = base_logits.detach()
    detached_reward = reward_logits.detach()
    base_topk_logits, base_token_ids = torch.topk(
        detached_base,
        k=top_k,
        dim=-1,
    )
    reward_topk_logits, reward_token_ids = torch.topk(
        detached_reward,
        k=top_k,
        dim=-1,
    )
    return TAROTopKBatch(
        base_token_ids=base_token_ids,
        base_logits=base_topk_logits,
        reward_token_ids=reward_token_ids,
        reward_logits=reward_topk_logits,
    )
