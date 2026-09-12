"""Read-only RAD benchmark loading with stable, collision-free prompt IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from router_v2.evaluation.rad.config import RADEvaluationConfig


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    prompt_sentiment_class: str
    source_prompt_id: str | None = None
    source_index: int | None = None


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _stable_prompt_id(split: str, prompt_class: str, source_index: int) -> str:
    return f"rad:{split}:{prompt_class}:{source_index:08d}"


def _v1_source_prompt_id(
    value: dict[str, object],
    prompt_class: str,
    line_number: int,
) -> str:
    # V1 used the source md5 directly, with class/line as its fallback.
    return str(value.get("md5_hash") or f"{prompt_class}:{line_number}")


def read_prompt_records(
    path: str | Path,
    prompt_class: str,
    *,
    split: str,
    limit: int,
    offset: int,
) -> list[PromptRecord]:
    source = Path(path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if limit <= 0 or offset < 0:
        raise ValueError("Prompt limit/offset are invalid")
    records: list[PromptRecord] = []
    seen = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            prompt_value = value.get("prompt")
            prompt = (
                str(prompt_value.get("text", ""))
                if isinstance(prompt_value, dict)
                else str(prompt_value or "")
            )
            if not prompt:
                continue
            if seen < offset:
                seen += 1
                continue
            seen += 1
            source_index = line_number - 1
            records.append(
                PromptRecord(
                    prompt_id=_stable_prompt_id(
                        split,
                        prompt_class,
                        source_index,
                    ),
                    prompt=prompt,
                    prompt_sentiment_class=prompt_class,
                    source_prompt_id=_v1_source_prompt_id(
                        value,
                        prompt_class,
                        line_number,
                    ),
                    source_index=source_index,
                )
            )
            if len(records) >= limit:
                break
    if len(records) != limit:
        raise ValueError(
            f"Requested {limit} prompts from {source} at offset {offset}, "
            f"found {len(records)}"
        )
    return records


def load_prompt_split(
    config: RADEvaluationConfig,
    split: str,
    *,
    project_root: Path,
) -> list[PromptRecord]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    dataset_dir = project_path(project_root, config.dataset_dir)
    limit = (
        config.validation_prompt_limit_per_class
        if split == "validation"
        else config.test_prompt_limit_per_class
    )
    offset = 0 if split == "validation" else config.test_prompt_offset_per_class
    records: list[PromptRecord] = []
    for prompt_class in config.prompt_classes:
        records.extend(
            read_prompt_records(
                dataset_dir / f"{prompt_class}_prompts.jsonl",
                prompt_class,
                split=split,
                limit=limit,
                offset=offset,
            )
        )
    identifiers = [record.prompt_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate prompt IDs in {split} split")
    return records


def prompt_keys(
    prompts: Iterable[PromptRecord],
    seeds: Iterable[int],
) -> set[tuple[str, int]]:
    return {
        (prompt.prompt_id, int(seed))
        for prompt in prompts
        for seed in seeds
    }
