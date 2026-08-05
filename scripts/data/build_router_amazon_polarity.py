#!/usr/bin/env python3
"""Build prompt-continuation router data from local Amazon Polarity Arrow files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import shutil
import string
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
from pyarrow import ipc
from transformers import AutoTokenizer, PreTrainedTokenizerBase

LOGGER = logging.getLogger("router_amazon_polarity")


@dataclass(frozen=True)
class SourceInspection:
    source_dir: str
    format: str
    splits: list[str]
    data_files: dict[str, list[str]]
    row_counts: dict[str, int]
    fields: dict[str, str]
    label_names: list[str]
    positive_label: int
    title_field: str | None
    content_field: str
    official_splits_available: bool
    sample_rows: list[dict[str, Any]]
    source_file_hashes: dict[str, str]


@dataclass(frozen=True)
class BuildArgs:
    source_dir: Path
    benchmark_dir: Path
    output_dir: Path
    train_size: int
    validation_size: int
    dev_size: int
    smoke_test: bool
    smoke_train_size: int
    smoke_validation_size: int
    smoke_dev_size: int
    min_total_tokens: int
    max_total_tokens: int
    min_prompt_tokens: int
    min_continuation_tokens: int
    max_continuation_tokens: int
    min_split_ratio: float
    max_split_ratio: float
    seed: int
    near_duplicate_threshold: float
    batch_size: int
    tokenizer_name: str
    local_files_only: bool
    overwrite: bool
    write_dev: bool


@dataclass(frozen=True)
class SourceRecord:
    source_split: str
    source_index: int
    label: int
    title: str | None
    content: str


@dataclass(frozen=True)
class PreparedExample:
    id: str
    prompt: str
    continuation: str
    label: int
    source: str
    source_split: str
    source_index: int
    text: str
    total_tokens: int
    prompt_tokens: int
    continuation_tokens: int
    split_token_index: int
    split_ratio: float
    normalized_text_hash: str
    prompt_hash: str
    continuation_hash: str
    exact_fingerprint: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    return text.strip(string.punctuation + string.whitespace)


def text_fingerprint(text: str) -> str:
    return sha256_text(normalize_for_match(text))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_for_match(text)))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def meaningful_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text)) and len(token_set(text)) >= 3


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arrow_files_for_split(source_dir: Path, split: str) -> list[Path]:
    state_path = source_dir / split / "state.json"
    if not state_path.exists():
        return sorted((source_dir / split).glob("data-*.arrow"))
    state = load_json(state_path)
    files = [source_dir / split / item["filename"] for item in state.get("_data_files", [])]
    return [path for path in files if path.exists()]


def read_arrow_table(path: Path, columns: Sequence[str] | None = None) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            table = reader.read_all()
    if columns is not None:
        table = table.select([column for column in columns if column in table.column_names])
    return table


def parse_label_names(dataset_info: dict[str, Any]) -> list[str]:
    label_info = dataset_info.get("features", {}).get("label", {})
    names = label_info.get("names", [])
    if not isinstance(names, list) or "positive" not in names:
        raise ValueError("Could not map the positive label from dataset_info.json")
    return [str(name) for name in names]


def inspect_source(source_dir: Path, sample_limit: int = 3) -> SourceInspection:
    dataset_dict_path = source_dir / "dataset_dict.json"
    if not dataset_dict_path.exists():
        raise FileNotFoundError(f"Missing {dataset_dict_path}")
    splits = list(load_json(dataset_dict_path).get("splits", []))
    if not splits:
        raise ValueError(f"No splits found in {dataset_dict_path}")

    data_files: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    sample_rows: list[dict[str, Any]] = []
    dataset_info = load_json(source_dir / splits[0] / "dataset_info.json")
    label_names = parse_label_names(dataset_info)
    positive_label = label_names.index("positive")

    for split in splits:
        files = arrow_files_for_split(source_dir, split)
        data_files[split] = [str(path) for path in files]
        info_path = source_dir / split / "dataset_info.json"
        state_path = source_dir / split / "state.json"
        for metadata_path in [info_path, state_path]:
            if metadata_path.exists():
                source_hashes[str(metadata_path)] = sha256_file(metadata_path)
        for path in files:
            source_hashes[str(path)] = sha256_file(path)
        split_info = load_json(info_path) if info_path.exists() else {}
        info_count = split_info.get("splits", {}).get(split, {}).get("num_examples")
        if isinstance(info_count, int):
            row_counts[split] = info_count
        else:
            row_counts[split] = sum(read_arrow_table(path, columns=["label"]).num_rows for path in files)
        if len(sample_rows) < sample_limit and files:
            table = read_arrow_table(files[0]).slice(0, sample_limit - len(sample_rows))
            for row in table.to_pylist():
                sample_rows.append(
                    {
                        key: (value[:160] + "..." if isinstance(value, str) and len(value) > 160 else value)
                        for key, value in row.items()
                    }
                )

    features = dataset_info.get("features", {})
    fields = {name: str(spec.get("dtype", spec.get("_type", ""))) for name, spec in features.items()}
    content_field = "content" if "content" in features else "text"
    if content_field not in features:
        raise ValueError("Could not find a review content/text field in source features")

    return SourceInspection(
        source_dir=str(source_dir),
        format="huggingface_saved_arrow",
        splits=splits,
        data_files=data_files,
        row_counts=row_counts,
        fields=fields,
        label_names=label_names,
        positive_label=positive_label,
        title_field="title" if "title" in features else None,
        content_field=content_field,
        official_splits_available="train" in splits and "test" in splits,
        sample_rows=sample_rows,
        source_file_hashes=source_hashes,
    )


def count_labels(source_dir: Path, split: str, batch_size: int = 65536) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in arrow_files_for_split(source_dir, split):
        table = read_arrow_table(path, columns=["label"])
        label_index = table.schema.get_field_index("label")
        for batch in table.to_batches(max_chunksize=batch_size):
            labels = batch.column(label_index).to_pylist()
            counts.update(int(label) for label in labels if label is not None)
    return counts


def iter_source_records(
    source_dir: Path,
    split: str,
    title_field: str | None,
    content_field: str,
    batch_size: int,
) -> Iterator[SourceRecord]:
    source_index = 0
    columns = ["label", content_field] + ([title_field] if title_field else [])
    for path in arrow_files_for_split(source_dir, split):
        table = read_arrow_table(path, columns=columns)
        for batch in table.to_batches(max_chunksize=batch_size):
            names = batch.schema.names
            label_col = batch.column(names.index("label")).to_pylist()
            content_col = batch.column(names.index(content_field)).to_pylist()
            title_col = batch.column(names.index(title_field)).to_pylist() if title_field else [None] * len(label_col)
            for label, title, content in zip(label_col, title_col, content_col):
                yield SourceRecord(
                    source_split=split,
                    source_index=source_index,
                    label=int(label) if label is not None else -1,
                    title=str(title) if title is not None else None,
                    content=str(content) if content is not None else "",
                )
                source_index += 1


def combine_review(title: str | None, content: str) -> str:
    title = normalize_text(title or "")
    content = normalize_text(content or "")
    if title and content:
        separator = " " if title.endswith((".", "!", "?", "\"", "'")) else ". "
        return normalize_text(f"{title}{separator}{content}")
    return normalize_text(content or title)


def choose_split_index(
    token_ids: Sequence[int],
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    min_prompt_tokens: int,
    min_continuation_tokens: int,
    max_continuation_tokens: int,
    min_split_ratio: float,
    max_split_ratio: float,
    rng: random.Random,
) -> int | None:
    total = len(token_ids)
    low = max(min_prompt_tokens, math.ceil(total * min_split_ratio))
    high = min(total - min_continuation_tokens, math.floor(total * max_split_ratio))
    high = min(high, total - 1)
    low = max(low, total - max_continuation_tokens)
    if low > high:
        return None

    candidates = list(range(low, high + 1))
    rng.shuffle(candidates)

    def boundary_score(index: int) -> tuple[int, int]:
        previous_text = tokenizer.decode(token_ids[index - 1 : index])
        next_text = tokenizer.decode(token_ids[index : min(total, index + 1)])
        previous = previous_text[-1:] if previous_text else ""
        next_char = next_text[:1]
        if previous in ".!?":
            score = 0
        elif previous.isspace() or next_char.isspace():
            score = 1
        elif previous in ",;:":
            score = 2
        else:
            score = 3
        target = (min_split_ratio + max_split_ratio) / 2
        ratio_distance = abs(index / total - target)
        return score, int(ratio_distance * 10000)

    candidates.sort(key=boundary_score)
    return candidates[0]


def prepare_record(
    record: SourceRecord,
    tokenizer: PreTrainedTokenizerBase,
    positive_label: int,
    args: BuildArgs,
) -> tuple[PreparedExample | None, str]:
    if record.label != positive_label:
        return None, "non_positive_label"
    text = combine_review(record.title, record.content)
    if not meaningful_text(text):
        return None, "malformed"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)
    if total < args.min_total_tokens or total > args.max_total_tokens:
        return None, "length_violation"

    rng_seed = f"{args.seed}:{record.source_split}:{record.source_index}:{sha256_text(text)[:16]}"
    rng = random.Random(rng_seed)
    split_index = choose_split_index(
        token_ids,
        text,
        tokenizer,
        args.min_prompt_tokens,
        args.min_continuation_tokens,
        args.max_continuation_tokens,
        args.min_split_ratio,
        args.max_split_ratio,
        rng,
    )
    if split_index is None:
        return None, "split_violation"

    prompt = tokenizer.decode(token_ids[:split_index])
    continuation = tokenizer.decode(token_ids[split_index:])
    if prompt + continuation != text:
        return None, "reconstruction_violation"
    if split_index < args.min_prompt_tokens:
        return None, "split_violation"
    continuation_tokens = total - split_index
    if continuation_tokens < args.min_continuation_tokens or continuation_tokens > args.max_continuation_tokens:
        return None, "split_violation"
    if not meaningful_text(continuation):
        return None, "trivial_continuation"

    exact = text_fingerprint(text)
    return (
        PreparedExample(
            id=f"amazon-polarity-positive-{record.source_index:08d}",
            prompt=prompt,
            continuation=continuation,
            label=1,
            source="amazon_polarity",
            source_split=record.source_split,
            source_index=record.source_index,
            text=text,
            total_tokens=total,
            prompt_tokens=split_index,
            continuation_tokens=continuation_tokens,
            split_token_index=split_index,
            split_ratio=split_index / total,
            normalized_text_hash=sha256_text(text),
            prompt_hash=sha256_text(prompt),
            continuation_hash=sha256_text(continuation),
            exact_fingerprint=exact,
        ),
        "accepted",
    )


def extract_json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from extract_json_strings(nested)
    elif isinstance(value, list):
        for item in value:
            yield from extract_json_strings(item)


def benchmark_strings(benchmark_paths: Sequence[Path]) -> list[str]:
    strings: list[str] = []
    for path in benchmark_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                row_strings = list(extract_json_strings(row))
                strings.extend(row_strings)
                prompt = row.get("prompt")
                continuation = row.get("continuation")
                prompt_text = prompt.get("text") if isinstance(prompt, dict) else prompt
                continuation_text = continuation.get("text") if isinstance(continuation, dict) else continuation
                if isinstance(prompt_text, str) and isinstance(continuation_text, str):
                    strings.append(prompt_text + continuation_text)
    return strings


def benchmark_leakage_index(benchmark_paths: Sequence[Path]) -> tuple[set[str], list[set[str]]]:
    strings = benchmark_strings(benchmark_paths)
    exact = {text_fingerprint(text) for text in strings if normalize_for_match(text)}
    near = [tokens for tokens in (token_set(text) for text in strings) if tokens]
    return exact, near


def build_near_duplicate_index(near_benchmark: Sequence[set[str]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for row_index, tokens in enumerate(near_benchmark):
        for token in tokens:
            index.setdefault(token, []).append(row_index)
    return index


def has_near_duplicate(
    tokens: set[str],
    near_benchmark: Sequence[set[str]],
    threshold: float,
    near_index: dict[str, list[int]] | None = None,
) -> bool:
    if not tokens:
        return False
    if near_index is None:
        return any(jaccard(tokens, bench_tokens) >= threshold for bench_tokens in near_benchmark)
    candidate_counts: Counter[int] = Counter()
    for token in tokens:
        candidate_counts.update(near_index.get(token, []))
    token_count = len(tokens)
    min_len = threshold * token_count
    max_len = token_count / threshold if threshold > 0 else float("inf")
    for row_index, overlap in candidate_counts.items():
        bench_tokens = near_benchmark[row_index]
        bench_len = len(bench_tokens)
        if bench_len < min_len or bench_len > max_len:
            continue
        # For Jaccard >= t, overlap must satisfy i / (a + b - i) >= t.
        min_overlap = threshold * (token_count + bench_len) / (1.0 + threshold)
        if overlap >= min_overlap and jaccard(tokens, bench_tokens) >= threshold:
            return True
    return False


def has_benchmark_leakage(
    example: PreparedExample,
    exact_benchmark: set[str],
    near_benchmark: Sequence[set[str]],
    threshold: float,
    near_index: dict[str, list[int]] | None = None,
) -> tuple[bool, str]:
    candidates = [example.text, example.prompt, example.continuation, example.prompt + example.continuation]
    if any(text_fingerprint(text) in exact_benchmark for text in candidates):
        return True, "benchmark_exact_leakage"
    if threshold <= 0:
        return False, ""
    for tokens in (token_set(text) for text in candidates):
        if has_near_duplicate(tokens, near_benchmark, threshold, near_index):
            return True, "benchmark_near_leakage"
    return False, ""


def split_examples(
    examples: Sequence[PreparedExample],
    train_size: int,
    validation_size: int,
    dev_size: int,
    seed: int,
) -> dict[str, list[PreparedExample]]:
    ordered = sorted(
        examples,
        key=lambda ex: sha256_text(f"{seed}:{ex.source_split}:{ex.source_index}:{ex.exact_fingerprint}"),
    )
    train = list(ordered[:train_size])
    validation = list(ordered[train_size : train_size + validation_size])
    dev = list(ordered[train_size + validation_size : train_size + validation_size + dev_size])
    return {"train": train, "validation": validation, "dev": dev}


def output_record(example: PreparedExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "prompt": example.prompt,
        "continuation": example.continuation,
        "label": example.label,
        "source": example.source,
        "source_split": example.source_split,
        "source_index": example.source_index,
    }


def manifest_record(example: PreparedExample, split: str) -> dict[str, Any]:
    data = asdict(example)
    data.pop("text")
    data["output_split"] = split
    return data


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {"min": min(values), "max": max(values), "avg": mean(values)}


def build_dataset(args: BuildArgs) -> dict[str, Any]:
    inspection = inspect_source(args.source_dir)
    if inspection.positive_label != 1:
        raise ValueError(f"Expected positive label id 1, found {inspection.positive_label}")
    if "train" not in inspection.splits:
        raise ValueError("Official train split is required for version 1")

    train_size = args.smoke_train_size if args.smoke_test else args.train_size
    validation_size = args.smoke_validation_size if args.smoke_test else args.validation_size
    dev_size = args.smoke_dev_size if args.smoke_test else (args.dev_size if args.write_dev else 0)
    needed = train_size + validation_size + dev_size

    benchmark_paths = [
        args.benchmark_dir / "negative_prompts.jsonl",
        args.benchmark_dir / "neutral_prompts.jsonl",
        args.benchmark_dir / "positive_prompts.jsonl",
    ]
    exact_benchmark, near_benchmark = benchmark_leakage_index(benchmark_paths)
    near_index = build_near_duplicate_index(near_benchmark)

    label_counts = count_labels(args.source_dir, "train", args.batch_size)
    counters: Counter[str] = Counter()
    examples: list[PreparedExample] = []
    seen_exact: set[str] = set()

    LOGGER.info("Source inspection: %s", json.dumps(asdict(inspection), indent=2)[:4000])
    LOGGER.info("Need %d final examples", needed)

    for record in iter_source_records(
        args.source_dir,
        "train",
        inspection.title_field,
        inspection.content_field,
        args.batch_size,
    ):
        if len(examples) >= needed:
            break
        example, status = prepare_record(record, AutoTokenizerHolder.tokenizer(args), inspection.positive_label, args)
        if example is None:
            counters[status] += 1
            continue
        if example.exact_fingerprint in seen_exact:
            counters["exact_duplicate"] += 1
            continue
        leaked, leakage_status = has_benchmark_leakage(
            example,
            exact_benchmark,
            near_benchmark,
            args.near_duplicate_threshold,
            near_index,
        )
        if leaked:
            counters[leakage_status] += 1
            continue
        seen_exact.add(example.exact_fingerprint)
        examples.append(example)
        counters["accepted"] += 1
        if len(examples) % 1000 == 0:
            LOGGER.info("Accepted %d/%d examples", len(examples), needed)

    splits = split_examples(examples, train_size, validation_size, dev_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        if name == "dev" and not rows:
            if path.exists() and args.overwrite:
                path.unlink()
            continue
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        write_jsonl(path, (output_record(row) for row in rows))

    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to replace it")
    write_jsonl(
        manifest_path,
        (manifest_record(row, split) for split, rows in splits.items() for row in rows if rows),
    )

    all_rows = [row for rows in splits.values() for row in rows]
    output_hashes = {
        str(path): sha256_file(path)
        for path in sorted(args.output_dir.glob("*.jsonl"))
        if path.is_file()
    }
    report = {
        "source_directory": str(args.source_dir),
        "source_inspection": asdict(inspection),
        "source_format": inspection.format,
        "field_mapping": {
            "label": "label",
            "title": inspection.title_field,
            "content": inspection.content_field,
        },
        "positive_label_mapping": {"positive": inspection.positive_label, "label_names": inspection.label_names},
        "raw_row_counts_by_split": inspection.row_counts,
        "train_label_counts": dict(sorted(label_counts.items())),
        "positive_rows_before_filtering": label_counts.get(inspection.positive_label, 0),
        "removed_malformed_rows": counters["malformed"],
        "removed_length_violations": counters["length_violation"],
        "removed_split_violations": counters["split_violation"],
        "removed_reconstruction_violations": counters["reconstruction_violation"],
        "removed_trivial_continuations": counters["trivial_continuation"],
        "removed_exact_duplicates": counters["exact_duplicate"],
        "removed_near_duplicates": counters["near_duplicate"],
        "removed_benchmark_leakage": counters["benchmark_exact_leakage"] + counters["benchmark_near_leakage"],
        "removed_benchmark_exact_leakage": counters["benchmark_exact_leakage"],
        "removed_benchmark_near_leakage": counters["benchmark_near_leakage"],
        "processed_input_rows_until_target_met": sum(counters.values()),
        "final_counts": {split: len(rows) for split, rows in splits.items() if rows or split != "dev"},
        "token_stats": {
            "total": stats([row.total_tokens for row in all_rows]),
            "prompt": stats([row.prompt_tokens for row in all_rows]),
            "continuation": stats([row.continuation_tokens for row in all_rows]),
            "split_ratio": stats([row.split_ratio for row in all_rows]),
        },
        "source_file_hashes": inspection.source_file_hashes,
        "processed_output_hashes": output_hashes,
        "data_report_note": "data_report.json is not included in processed_output_hashes because it records hashes.",
        "random_seed": args.seed,
        "preprocessing_args": serializable_args(args),
        "benchmark_files": [str(path) for path in benchmark_paths],
        "benchmark_string_count": len(near_benchmark),
        "near_duplicate_method": "token-set Jaccard against benchmark strings",
        "near_duplicate_threshold": args.near_duplicate_threshold,
    }
    report_path = args.output_dir / "data_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; pass --overwrite to replace it")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


class AutoTokenizerHolder:
    _tokenizer: PreTrainedTokenizerBase | None = None

    @classmethod
    def tokenizer(cls, args: BuildArgs) -> PreTrainedTokenizerBase:
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                local_files_only=args.local_files_only,
            )
        return cls._tokenizer


def serializable_args(args: BuildArgs) -> dict[str, Any]:
    data = asdict(args)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def parse_args(argv: Sequence[str] | None = None) -> BuildArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset/RAD_train/amazon_polarity"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("dataset/rad_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/RAD_train/router_amazon_polarity"))
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--write-dev", action="store_true", help="Write the dev split using --dev-size.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke split sizes.")
    parser.add_argument("--smoke-train-size", type=int, default=1000)
    parser.add_argument("--smoke-validation-size", type=int, default=200)
    parser.add_argument("--smoke-dev-size", type=int, default=0)
    parser.add_argument("--min-total-tokens", type=int, default=20)
    parser.add_argument("--max-total-tokens", type=int, default=150)
    parser.add_argument("--min-prompt-tokens", type=int, default=5)
    parser.add_argument("--min-continuation-tokens", type=int, default=8)
    parser.add_argument("--max-continuation-tokens", type=int, default=80)
    parser.add_argument("--min-split-ratio", type=float, default=0.30)
    parser.add_argument("--max-split-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    ns = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, ns.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    if ns.output_dir.exists() and any(ns.output_dir.iterdir()) and not ns.overwrite:
        raise FileExistsError(f"{ns.output_dir} is not empty; pass --overwrite to replace outputs")
    return BuildArgs(
        source_dir=ns.source_dir,
        benchmark_dir=ns.benchmark_dir,
        output_dir=ns.output_dir,
        train_size=ns.train_size,
        validation_size=ns.validation_size,
        dev_size=ns.dev_size,
        smoke_test=ns.smoke_test,
        smoke_train_size=ns.smoke_train_size,
        smoke_validation_size=ns.smoke_validation_size,
        smoke_dev_size=ns.smoke_dev_size,
        min_total_tokens=ns.min_total_tokens,
        max_total_tokens=ns.max_total_tokens,
        min_prompt_tokens=ns.min_prompt_tokens,
        min_continuation_tokens=ns.min_continuation_tokens,
        max_continuation_tokens=ns.max_continuation_tokens,
        min_split_ratio=ns.min_split_ratio,
        max_split_ratio=ns.max_split_ratio,
        seed=ns.seed,
        near_duplicate_threshold=ns.near_duplicate_threshold,
        batch_size=ns.batch_size,
        tokenizer_name=ns.tokenizer_name,
        local_files_only=not ns.allow_tokenizer_download,
        overwrite=ns.overwrite,
        write_dev=ns.write_dev,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    report = build_dataset(args)
    LOGGER.info("Wrote router data to %s", args.output_dir)
    LOGGER.info("Final counts: %s", report["final_counts"])


if __name__ == "__main__":
    main()
