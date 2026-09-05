"""Continuation likelihood evaluation for base and sentiment guide models."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import pyarrow as pa
from pyarrow import ipc

from router_v2.guide_model.data import (
    RADSample,
    TokenizedRADSample,
    collate_tokenized_samples,
    tokenize_sample,
)


@dataclass(frozen=True)
class ContinuationScore:
    sample_id: str
    mean_log_probability: float
    total_log_probability: float
    token_count: int


def load_negative_audit_samples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_samples: int,
    max_length: int,
) -> list[RADSample]:
    """Load held-out negative Amazon reviews from the official test Arrow."""

    source_path = Path(path)
    if source_path.suffix != ".arrow":
        raise ValueError("Negative sentiment audit source must be Arrow")

    def normalize_text(value: str) -> str:
        value = unicodedata.normalize("NFKC", value)
        return re.sub(r"\s+", " ", value).strip()

    def read_table() -> pa.Table:
        with pa.memory_map(str(source_path), "r") as source:
            try:
                return ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                return ipc.open_file(source).read_all()

    samples: list[RADSample] = []
    table = read_table().select(["label", "title", "content"])
    raw_source_index = 0
    for batch in table.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            row_source_index = raw_source_index
            raw_source_index += 1
            if int(row["label"]) != 0:
                continue
            title = normalize_text(str(row.get("title") or ""))
            content = normalize_text(str(row.get("content") or ""))
            separator = (
                " " if title.endswith((".", "!", "?", '"', "'")) else ". "
            )
            text = normalize_text(
                f"{title}{separator}{content}" if title else content
            )
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not 16 <= len(token_ids) <= max_length - 1:
                continue
            continuation_length = min(80, max(8, len(token_ids) // 2))
            split_index = len(token_ids) - continuation_length
            if split_index <= 0:
                continue
            prompt_text = tokenizer.decode(token_ids[:split_index])
            continuation_text = tokenizer.decode(token_ids[split_index:])
            if prompt_text + continuation_text != text:
                continue
            if (
                tokenizer.encode(prompt_text, add_special_tokens=False)
                + tokenizer.encode(continuation_text, add_special_tokens=False)
                != token_ids
            ):
                continue
            sample = RADSample(
                sample_id=(
                    f"amazon-test-negative-{row_source_index:06d}"
                ),
                prompt=prompt_text,
                continuation=continuation_text,
                label=0,
                source="amazon_polarity",
                source_split="test",
                source_index=row_source_index,
            )
            try:
                tokenize_sample(
                    tokenizer,
                    sample,
                    max_length=max_length,
                    include_eos_target=True,
                )
            except ValueError:
                continue
            samples.append(sample)
            if len(samples) >= max_samples:
                return samples
    if len(samples) < max_samples:
        raise ValueError(
            f"Only {len(samples)} tokenizer-stable negative audit samples "
            f"were available; required {max_samples}"
        )
    return samples


def score_tokenized_samples(
    model: Any,
    samples: Sequence[TokenizedRADSample],
    *,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    use_fp16: bool,
) -> list[ContinuationScore]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            items = samples[start : start + batch_size]
            batch = collate_tokenized_samples(
                items,
                pad_token_id=pad_token_id,
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~mask, 0)
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                -1,
                safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_log_probs = token_log_probs * mask
            totals = token_log_probs.sum(dim=-1)
            counts = mask.sum(dim=-1)
            if bool((counts <= 0).any()):
                raise ValueError("A validation sample has no supervised tokens")
            if not bool(torch.isfinite(totals).all()):
                raise FloatingPointError(
                    "Non-finite continuation log probability"
                )
            for item, total, count in zip(items, totals, counts):
                total_value = float(total.cpu())
                count_value = int(count.cpu())
                scores.append(
                    ContinuationScore(
                        sample_id=item.sample.sample_id,
                        mean_log_probability=total_value / count_value,
                        total_log_probability=total_value,
                        token_count=count_value,
                    )
                )
    return scores


def paired_score_summary(
    base_scores: Sequence[ContinuationScore],
    guide_scores: Sequence[ContinuationScore],
) -> dict[str, Any]:
    if len(base_scores) != len(guide_scores) or not base_scores:
        raise ValueError("Base/guide validation scores must align and be non-empty")
    rows = []
    for base, guide in zip(base_scores, guide_scores):
        if base.sample_id != guide.sample_id:
            raise ValueError("Base/guide score sample IDs do not align")
        rows.append(
            {
                "sample_id": base.sample_id,
                "base_mean_log_probability": base.mean_log_probability,
                "guide_mean_log_probability": guide.mean_log_probability,
                "guide_minus_base": (
                    guide.mean_log_probability - base.mean_log_probability
                ),
                "token_count": base.token_count,
            }
        )
    count = len(rows)
    return {
        "num_samples": count,
        "base_mean_nll": -sum(
            row["base_mean_log_probability"] for row in rows
        )
        / count,
        "guide_mean_nll": -sum(
            row["guide_mean_log_probability"] for row in rows
        )
        / count,
        "mean_guide_minus_base": sum(
            row["guide_minus_base"] for row in rows
        )
        / count,
        "rows": rows,
    }
