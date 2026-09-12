"""Frozen online full-logit computation for exact Stage 5 NLL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from router_v2.guide_model.extraction import (
    ExtractionBatch,
    prepare_extraction_batch,
)
from router_v2.training.data import CachedRouterBatch


@dataclass(frozen=True)
class OnlineLogitBatch:
    base_logits: torch.Tensor
    guide_logits: torch.Tensor
    base_selected_logprob: torch.Tensor
    extraction_batch: ExtractionBatch


def _gather_token_positions(
    sequence_logits: torch.Tensor,
    extraction_batch: ExtractionBatch,
    *,
    output_length: int,
) -> torch.Tensor:
    if sequence_logits.ndim != 3:
        raise ValueError("Causal model logits must have shape [B, S, V]")
    output = torch.zeros(
        sequence_logits.shape[0],
        output_length,
        sequence_logits.shape[-1],
        dtype=sequence_logits.dtype,
        device=sequence_logits.device,
    )
    for row, positions in enumerate(extraction_batch.logit_positions):
        if len(positions) > output_length:
            raise ValueError("Online continuation exceeds cached sequence length")
        index = torch.tensor(
            positions,
            dtype=torch.long,
            device=sequence_logits.device,
        )
        output[row, : len(positions)] = sequence_logits[row].index_select(
            0,
            index,
        )
    return output


def _model_logits(
    model: Any,
    extraction_batch: ExtractionBatch,
    *,
    device: torch.device,
    output_length: int,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=extraction_batch.input_ids.to(device),
            attention_mask=extraction_batch.attention_mask.to(device),
            position_ids=extraction_batch.position_ids.to(device),
            use_cache=False,
        ).logits
        if logits.dtype != torch.float32:
            raise TypeError(
                f"Scientific Stage 5 requires FP32 model logits, got {logits.dtype}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Frozen causal model emitted NaN or infinity")
        selected = _gather_token_positions(
            logits,
            extraction_batch,
            output_length=output_length,
        )
    return selected.detach()


def compute_online_logit_batch(
    base_model: Any,
    guide_model: Any,
    tokenizer: Any,
    cached_batch: CachedRouterBatch,
    *,
    device: torch.device,
    max_length: int,
    max_continuation_tokens: int,
) -> OnlineLogitBatch:
    extraction_batch = prepare_extraction_batch(
        tokenizer,
        cached_batch.samples,
        max_length=max_length,
        max_continuation_tokens=max_continuation_tokens,
    )
    for row, continuation_ids in enumerate(extraction_batch.continuation_ids):
        cached_length = int(cached_batch.valid_mask[row].sum())
        if len(continuation_ids) != cached_length:
            raise ValueError(
                f"Online/cache continuation length mismatch for "
                f"{cached_batch.sample_ids[row]}"
            )
        cached_gold = cached_batch.gold_token_ids[row, :cached_length].tolist()
        if cached_gold != list(continuation_ids):
            raise ValueError(
                f"Online/cache gold mismatch for {cached_batch.sample_ids[row]}"
            )
    output_length = cached_batch.gold_token_ids.shape[1]
    base_logits = _model_logits(
        base_model,
        extraction_batch,
        device=device,
        output_length=output_length,
    )
    guide_logits = _model_logits(
        guide_model,
        extraction_batch,
        device=device,
        output_length=output_length,
    )
    gold = cached_batch.gold_token_ids.to(device)
    gold_logits = base_logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    base_selected_logprob = gold_logits - torch.logsumexp(base_logits, dim=-1)
    base_selected_logprob = torch.where(
        cached_batch.valid_mask.to(device),
        base_selected_logprob,
        torch.zeros_like(base_selected_logprob),
    )
    return OnlineLogitBatch(
        base_logits=base_logits,
        guide_logits=guide_logits,
        base_selected_logprob=base_selected_logprob.detach(),
        extraction_batch=extraction_batch,
    )
