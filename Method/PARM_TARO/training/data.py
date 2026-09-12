"""Processed Stage 8 examples and teacher-forced token spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from PARM_TARO.data.multi_objective import validate_processed_dataset


PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


@dataclass(frozen=True)
class MultiObjectiveExample:
    sample_id: str
    source_index: int
    prompt: str
    responses: tuple[str, str]
    better_response_id: int
    safer_response_id: int


@dataclass(frozen=True)
class CrossingTokenDiagnostic:
    token_index: int
    token_id: int
    tokenizer_token: str
    text: str
    span: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "tokenizer_token": self.tokenizer_token,
            "text": self.text,
            "span": list(self.span),
        }


@dataclass(frozen=True)
class ResponseBoundaryDiagnostic:
    sample_id: str
    response_index: int
    boundary_character_index: int
    crossing_tokens: tuple[CrossingTokenDiagnostic, ...]
    first_response_loss_token_index: int
    boundary_tokens_excluded: int
    zero_length_token_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "response_index": self.response_index,
            "boundary_character_index": self.boundary_character_index,
            "crossing_tokens": [token.to_dict() for token in self.crossing_tokens],
            "first_response_loss_token_index": (
                self.first_response_loss_token_index
            ),
            "boundary_tokens_excluded": self.boundary_tokens_excluded,
            "zero_length_token_indices": list(self.zero_length_token_indices),
        }


@dataclass(frozen=True)
class TokenizedResponse:
    sample_id: str
    response_index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logit_positions: torch.Tensor
    gold_token_ids: torch.Tensor
    truncated: bool
    boundary_diagnostic: ResponseBoundaryDiagnostic


class SequenceTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class _ConcatenatedEncoding:
    input_ids: tuple[int, ...]
    continuation_indices: tuple[int, ...]
    diagnostic: ResponseBoundaryDiagnostic


def load_examples(
    data_root: str | Path,
    split: str,
    *,
    max_samples: int | None = None,
) -> list[MultiObjectiveExample]:
    if split not in {"train", "validation"}:
        raise ValueError("Stage 9 may load only train or validation")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or null")
    root = Path(data_root).resolve()
    validate_processed_dataset(
        root,
        source_path=Path(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
                "source"
            ]["path"]
        ),
    )
    rows = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    examples = [
        MultiObjectiveExample(
            sample_id=str(row["sample_id"]),
            source_index=int(row["source_index"]),
            prompt=str(row["prompt"]),
            responses=(str(row["response_0"]), str(row["response_1"])),
            better_response_id=int(row["better_response_id"]),
            safer_response_id=int(row["safer_response_id"]),
        )
        for row in rows
    ]
    identifiers = [example.sample_id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate sample IDs in {split}")
    return examples[:max_samples] if max_samples is not None else examples


def deterministic_example_order(
    examples: Sequence[MultiObjectiveExample],
    *,
    epoch: int,
    seed: int,
) -> list[MultiObjectiveExample]:
    if epoch < 0 or seed < 0:
        raise ValueError("epoch and seed must be non-negative")

    def key(example: MultiObjectiveExample) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{seed}\0{epoch}\0{example.sample_id}".encode("utf-8")
        ).hexdigest()
        return digest, example.sample_id

    return sorted(examples, key=key)


def iter_validation_tasks(
    examples: Sequence[MultiObjectiveExample],
    helpfulness_grid: Sequence[float],
) -> Iterator[tuple[MultiObjectiveExample, torch.Tensor]]:
    for example in examples:
        for helpfulness in helpfulness_grid:
            yield example, torch.tensor(
                [helpfulness, 1.0 - helpfulness], dtype=torch.float32
            )


def _tokenizer_token_text(tokenizer: Any, token_id: int) -> str:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        return str(token_id)
    value = converter(token_id)
    return str(value)


def _encode_concatenated_response(
    tokenizer: Any,
    example: MultiObjectiveExample,
    response_index: int,
) -> _ConcatenatedEncoding:
    if response_index not in (0, 1):
        raise ValueError("response_index must be 0 or 1")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Stage 9 requires a fast tokenizer with offset mappings")
    prefix = PROMPT_TEMPLATE.format(prompt=example.prompt)
    full_text = prefix + example.responses[response_index]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple(
        tuple(int(item) for item in value)
        for value in encoded["offset_mapping"]
    )
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer input IDs and offset mappings differ in length")
    invalid_offsets = [
        (index, span)
        for index, span in enumerate(offsets)
        if span[0] < 0
        or span[1] < span[0]
        or span[1] > len(full_text)
    ]
    if invalid_offsets:
        raise ValueError(f"Tokenizer emitted invalid offsets: {invalid_offsets[:3]}")

    boundary = len(prefix)
    crossing_indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start < boundary < end
    )
    continuation_indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start >= boundary and end > start
    )
    if not continuation_indices or continuation_indices[0] == 0:
        raise ValueError(
            f"No response-attributable tokens for {example.sample_id}/"
            f"response_{response_index}"
        )
    crossing_tokens = tuple(
        CrossingTokenDiagnostic(
            token_index=index,
            token_id=input_ids[index],
            tokenizer_token=_tokenizer_token_text(tokenizer, input_ids[index]),
            text=full_text[offsets[index][0] : offsets[index][1]],
            span=offsets[index],
        )
        for index in crossing_indices
    )
    diagnostic = ResponseBoundaryDiagnostic(
        sample_id=example.sample_id,
        response_index=response_index,
        boundary_character_index=boundary,
        crossing_tokens=crossing_tokens,
        first_response_loss_token_index=continuation_indices[0],
        boundary_tokens_excluded=len(crossing_tokens),
        zero_length_token_indices=tuple(
            index
            for index, (start, end) in enumerate(offsets)
            if start == end
        ),
    )
    return _ConcatenatedEncoding(
        input_ids=input_ids,
        continuation_indices=continuation_indices,
        diagnostic=diagnostic,
    )


def inspect_response_boundary(
    tokenizer: Any,
    example: MultiObjectiveExample,
    response_index: int,
) -> ResponseBoundaryDiagnostic:
    """Inspect exact full-string token spans without constructing model tensors."""

    return _encode_concatenated_response(
        tokenizer, example, response_index
    ).diagnostic


def tokenize_response(
    tokenizer: Any,
    example: MultiObjectiveExample,
    response_index: int,
    *,
    max_length: int,
    max_continuation_tokens: int,
    include_eos_target: bool,
) -> TokenizedResponse:
    encoded = _encode_concatenated_response(tokenizer, example, response_index)
    input_ids = encoded.input_ids
    continuation_indices = encoded.continuation_indices
    selected_indices = continuation_indices[:max_continuation_tokens]
    truncated = len(selected_indices) < len(continuation_indices)
    last_index = selected_indices[-1]
    bos_token_id = tokenizer.bos_token_id
    prefix_shift = int(bos_token_id is not None)
    sequence_ids = (
        ([int(bos_token_id)] if bos_token_id is not None else [])
        + list(input_ids[: last_index + 1])
    )
    gold = [input_ids[index] for index in selected_indices]
    logit_positions = [
        index + prefix_shift - 1 for index in selected_indices
    ]
    if include_eos_target:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        logit_positions.append(len(sequence_ids) - 1)
        gold.append(int(eos_token_id))
        sequence_ids.append(int(eos_token_id))
    if len(sequence_ids) > max_length:
        raise SequenceTooLongError(
            f"{example.sample_id}/response_{response_index} has "
            f"{len(sequence_ids)} tokens, max_length={max_length}"
        )
    input_tensor = torch.tensor([sequence_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_tensor)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    return TokenizedResponse(
        sample_id=example.sample_id,
        response_index=response_index,
        input_ids=input_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logit_positions=torch.tensor(logit_positions, dtype=torch.long),
        gold_token_ids=torch.tensor(gold, dtype=torch.long),
        truncated=truncated,
        boundary_diagnostic=encoded.diagnostic,
    )
