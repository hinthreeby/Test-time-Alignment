"""Frozen teacher-forced base/PBLORA full-vocabulary log probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from PARM_TARO.adapters.parm_adapter import (
    named_preference_to_parm,
    set_parm_preference,
)
from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.training.alpha import validate_alpha
from PARM_TARO.training.data import TokenizedResponse


@dataclass(frozen=True)
class FrozenSequenceDistributions:
    base_logprobs: torch.Tensor
    guide_logprobs: torch.Tensor
    gold_token_ids: torch.Tensor
    base_selected_logprobs: torch.Tensor
    position: torch.Tensor


def _selected_logits(
    model: Any,
    tokenized: TokenizedResponse,
    *,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=tokenized.input_ids.to(device),
            attention_mask=tokenized.attention_mask.to(device),
            position_ids=tokenized.position_ids.to(device),
            use_cache=False,
        )
        logits = output.logits
        selected = logits[0].index_select(
            0, tokenized.logit_positions.to(logits.device)
        ).float().unsqueeze(0)
        del output, logits
    if selected.ndim != 3 or not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("Frozen causal model emitted invalid logits")
    return selected.detach().to(device=device, dtype=torch.float32)


def compute_frozen_sequence_distributions(
    base_model: Any,
    guide_model: Any,
    tokenized: TokenizedResponse,
    alignment: PARMTokenAlignment,
    named_alpha: torch.Tensor,
    *,
    device: torch.device,
) -> FrozenSequenceDistributions:
    """Compute exact full-vocabulary inputs with no model autograd graph."""

    alpha = validate_alpha(named_alpha)
    if alpha.shape != (2,):
        raise ValueError("Online extraction requires one alpha vector")
    base_logits = _selected_logits(base_model, tokenized, device=device)
    alignment.validate_base_logits(base_logits)
    base_logprobs = F.log_softmax(base_logits, dim=-1).detach()
    guide_logprobs = compute_frozen_guide_logprobs(
        guide_model,
        tokenized,
        alignment,
        alpha,
        device=device,
    )
    gold = tokenized.gold_token_ids.to(device)
    selected = base_logprobs[0].gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    length = gold.shape[0]
    return FrozenSequenceDistributions(
        base_logprobs=base_logprobs,
        guide_logprobs=guide_logprobs,
        gold_token_ids=gold,
        base_selected_logprobs=selected.unsqueeze(0).detach(),
        position=torch.arange(length, device=device).unsqueeze(0),
    )


def compute_frozen_guide_logprobs(
    guide_model: Any,
    tokenized: TokenizedResponse,
    alignment: PARMTokenAlignment,
    named_alpha: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Compute one frozen PBLORA distribution without repeating the base pass."""

    alpha = validate_alpha(named_alpha)
    if alpha.shape != (2,):
        raise ValueError("Online extraction requires one alpha vector")
    set_parm_preference(
        guide_model,
        named_preference_to_parm(alpha),
    )
    guide_logits = _selected_logits(guide_model, tokenized, device=device)
    guide_logits = alignment.align_guide_logits(guide_logits)
    return F.log_softmax(guide_logits, dim=-1).detach()
