"""RAD positive-continuation data contract for guide training/extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RADSample:
    sample_id: str
    prompt: str
    continuation: str
    label: int
    source: str
    source_split: str
    source_index: int


@dataclass(frozen=True)
class TokenizedRADSample:
    sample: RADSample
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    continuation_ids: tuple[int, ...]


def load_rad_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[RADSample]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None")
    samples: list[RADSample] = []
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            required = {
                "id",
                "prompt",
                "continuation",
                "label",
                "source",
                "source_split",
                "source_index",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"RAD row at {path}:{line_number} is missing {missing}"
                )
            sample_id = str(row["id"])
            if not sample_id or sample_id in identifiers:
                raise ValueError(f"Duplicate or empty RAD ID: {sample_id!r}")
            prompt = str(row["prompt"])
            continuation = str(row["continuation"])
            if not prompt or not continuation:
                raise ValueError(f"Empty prompt/continuation for {sample_id}")
            if type(row["label"]) is not int or row["label"] != 1:
                raise ValueError(
                    f"Sentiment guide source must be positive-only: {sample_id}"
                )
            identifiers.add(sample_id)
            samples.append(
                RADSample(
                    sample_id=sample_id,
                    prompt=prompt,
                    continuation=continuation,
                    label=1,
                    source=str(row["source"]),
                    source_split=str(row["source_split"]),
                    source_index=int(row["source_index"]),
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"No RAD samples loaded from {path}")
    return samples


def tokenize_sample(
    tokenizer: Any,
    sample: RADSample,
    *,
    max_length: int,
    include_eos_target: bool,
) -> TokenizedRADSample:
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
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
            f"Tokenizer boundary is not stable for RAD sample {sample.sample_id}"
        )
    if not prompt_ids or not continuation_ids:
        raise ValueError(
            f"Tokenized prompt/continuation is empty for {sample.sample_id}"
        )
    input_ids = list(full_ids)
    labels = [-100] * len(prompt_ids) + list(continuation_ids)
    if include_eos_target:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token for EOS supervision")
        input_ids.append(int(tokenizer.eos_token_id))
        labels.append(int(tokenizer.eos_token_id))
    if len(input_ids) > max_length:
        raise ValueError(
            f"Sample {sample.sample_id} has {len(input_ids)} tokens, "
            f"exceeding configured max_length={max_length}"
        )
    return TokenizedRADSample(
        sample=sample,
        input_ids=tuple(int(value) for value in input_ids),
        labels=tuple(int(value) for value in labels),
        prompt_length=len(prompt_ids),
        continuation_ids=tuple(int(value) for value in continuation_ids),
    )


class PositiveContinuationDataset(Dataset[TokenizedRADSample]):
    """Pre-tokenized positive continuations with prompt labels masked."""

    def __init__(
        self,
        samples: Sequence[RADSample],
        tokenizer: Any,
        *,
        max_length: int,
        include_eos_target: bool,
    ) -> None:
        self.items = [
            tokenize_sample(
                tokenizer,
                sample,
                max_length=max_length,
                include_eos_target=include_eos_target,
            )
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TokenizedRADSample:
        return self.items[index]


def collate_tokenized_samples(
    samples: Sequence[TokenizedRADSample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample batch")
    max_length = max(len(sample.input_ids) for sample in samples)
    input_ids = torch.full(
        (len(samples), max_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[row, :length] = torch.tensor(sample.input_ids)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(sample.labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "samples": list(samples),
    }

