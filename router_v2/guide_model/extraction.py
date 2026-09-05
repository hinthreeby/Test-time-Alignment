"""Teacher-forced streaming Top-K extraction primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from router_v2.guide_model.data import RADSample


@dataclass(frozen=True)
class ExtractionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logit_positions: tuple[tuple[int, ...], ...]
    continuation_ids: tuple[tuple[int, ...], ...]
    samples: tuple[RADSample, ...]


@dataclass(frozen=True)
class CandidateBatch:
    token_ids: torch.Tensor
    logits: torch.Tensor
    boundary_margin: torch.Tensor
    source_dtype: str


@dataclass(frozen=True)
class ExtractedSampleRecords:
    sample: RADSample
    position: torch.Tensor
    base_token_ids: torch.Tensor
    base_logits: torch.Tensor
    guide_token_ids: torch.Tensor
    guide_logits: torch.Tensor
    gold_token_id: torch.Tensor

    @property
    def num_records(self) -> int:
        return int(self.position.shape[0])


def should_flush_before_extraction_batch(
    *,
    buffered_records: int,
    incoming_batch_records: int,
    shard_size: int,
) -> bool:
    """Keep extraction batches atomic so resume preserves padding context."""

    if buffered_records < 0 or incoming_batch_records <= 0:
        raise ValueError("Extraction record counts are invalid")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return (
        buffered_records > 0
        and buffered_records + incoming_batch_records > shard_size
    )


def prepare_extraction_batch(
    tokenizer: Any,
    samples: Sequence[RADSample],
    *,
    max_length: int,
    max_continuation_tokens: int,
) -> ExtractionBatch:
    if not samples:
        raise ValueError("Extraction batch cannot be empty")
    if max_continuation_tokens <= 0:
        raise ValueError("max_continuation_tokens must be positive")
    encoded = []
    max_input_length = 0
    for sample in samples:
        prompt_ids = tokenizer.encode(
            sample.prompt,
            add_special_tokens=False,
        )
        continuation_ids = tokenizer.encode(
            sample.continuation,
            add_special_tokens=False,
        )
        full_ids = tokenizer.encode(
            sample.prompt + sample.continuation,
            add_special_tokens=False,
        )
        if prompt_ids + continuation_ids != full_ids:
            raise ValueError(
                f"Unstable token boundary for {sample.sample_id}"
            )
        continuation_ids = continuation_ids[:max_continuation_tokens]
        if not prompt_ids or not continuation_ids:
            raise ValueError(
                f"Empty teacher-forcing sequence for {sample.sample_id}"
            )
        input_ids = prompt_ids + continuation_ids
        if len(input_ids) > max_length:
            raise ValueError(
                f"Teacher-forcing input exceeds max_length for "
                f"{sample.sample_id}"
            )
        positions = tuple(
            len(prompt_ids) + offset - 1
            for offset in range(len(continuation_ids))
        )
        encoded.append((input_ids, tuple(continuation_ids), positions))
        max_input_length = max(max_input_length, len(input_ids))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Extraction tokenizer must define pad_token_id")
    input_tensor = torch.full(
        (len(samples), max_input_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_tensor)
    for row, (input_ids, _, _) in enumerate(encoded):
        length = len(input_ids)
        input_tensor[row, :length] = torch.tensor(input_ids)
        attention_mask[row, :length] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    return ExtractionBatch(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logit_positions=tuple(item[2] for item in encoded),
        continuation_ids=tuple(item[1] for item in encoded),
        samples=tuple(samples),
    )


def select_topk_at_positions(
    sequence_logits: torch.Tensor,
    logit_positions: Sequence[Sequence[int]],
    *,
    top_k: int,
) -> tuple[CandidateBatch, ...]:
    """Select Top-K from logits without accepting or inspecting gold IDs."""

    if sequence_logits.ndim != 3:
        raise ValueError("sequence_logits must have shape [B, T, V]")
    if len(logit_positions) != sequence_logits.shape[0]:
        raise ValueError("logit_positions must align with batch dimension")
    if not 1 <= top_k <= sequence_logits.shape[-1]:
        raise ValueError("top_k is outside the vocabulary range")
    outputs = []
    for row, positions in enumerate(logit_positions):
        position_tensor = torch.tensor(
            positions,
            dtype=torch.long,
            device=sequence_logits.device,
        )
        selected = sequence_logits[row].index_select(0, position_tensor)
        diagnostic_k = min(top_k + 1, selected.shape[-1])
        values, token_ids = torch.topk(
            selected,
            k=diagnostic_k,
            dim=-1,
        )
        if diagnostic_k > top_k:
            boundary_margin = values[:, top_k - 1] - values[:, top_k]
        else:
            boundary_margin = torch.full(
                values.shape[:-1],
                float("inf"),
                device=values.device,
                dtype=values.dtype,
            )
        outputs.append(
            CandidateBatch(
                token_ids=(
                    token_ids[:, :top_k]
                    .detach()
                    .cpu()
                    .long()
                    .contiguous()
                ),
                logits=(
                    values[:, :top_k]
                    .detach()
                    .cpu()
                    .float()
                    .contiguous()
                ),
                boundary_margin=(
                    boundary_margin.detach().cpu().float().contiguous()
                ),
                source_dtype=str(sequence_logits.dtype),
            )
        )
    return tuple(outputs)


def model_topk(
    model: Any,
    batch: ExtractionBatch,
    *,
    top_k: int,
    device: torch.device,
    use_fp16: bool,
) -> tuple[CandidateBatch, ...]:
    model.eval()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                position_ids=batch.position_ids.to(device),
                use_cache=False,
            ).logits
        expected_dtype = torch.float16 if use_fp16 else torch.float32
        if logits.dtype != expected_dtype:
            raise TypeError(
                f"Expected {expected_dtype} model logits, got {logits.dtype}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Model emitted NaN or infinity")
        return select_topk_at_positions(
            logits,
            batch.logit_positions,
            top_k=top_k,
        )


def combine_extracted_records(
    batch: ExtractionBatch,
    base_candidates: Sequence[CandidateBatch],
    guide_candidates: Sequence[CandidateBatch],
) -> tuple[ExtractedSampleRecords, ...]:
    if not (
        len(batch.samples)
        == len(base_candidates)
        == len(guide_candidates)
    ):
        raise ValueError("Extracted candidate streams do not align")
    records = []
    for sample, gold_ids, base, guide in zip(
        batch.samples,
        batch.continuation_ids,
        base_candidates,
        guide_candidates,
    ):
        expected = len(gold_ids)
        if (
            base.token_ids.shape[0] != expected
            or guide.token_ids.shape[0] != expected
        ):
            raise ValueError("Candidate record counts do not match targets")
        records.append(
            ExtractedSampleRecords(
                sample=sample,
                position=torch.arange(expected, dtype=torch.long),
                base_token_ids=base.token_ids,
                base_logits=base.logits,
                guide_token_ids=guide.token_ids,
                guide_logits=guide.logits,
                gold_token_id=torch.tensor(gold_ids, dtype=torch.long),
            )
        )
    return tuple(records)
