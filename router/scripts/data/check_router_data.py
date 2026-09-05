#!/usr/bin/env python3
"""Validate generated router Amazon Polarity JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from router.scripts.data.build_router_amazon_polarity import benchmark_leakage_index, text_fingerprint


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/RAD_train/router_amazon_polarity"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("dataset/rad_benchmark"))
    args = parser.parse_args(argv)

    benchmark_paths = [
        args.benchmark_dir / "negative_prompts.jsonl",
        args.benchmark_dir / "neutral_prompts.jsonl",
        args.benchmark_dir / "positive_prompts.jsonl",
    ]
    exact_benchmark, _ = benchmark_leakage_index(benchmark_paths)
    ids: set[str] = set()
    fingerprints: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for split in ["train", "validation", "dev"]:
        path = args.output_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        counts[split] = len(rows)
        for row in rows:
            required = {"id", "prompt", "continuation", "label", "source", "source_split", "source_index"}
            missing = required - set(row)
            if missing:
                raise AssertionError(f"{path} row missing fields: {sorted(missing)}")
            if row["id"] in ids:
                raise AssertionError(f"Duplicate id: {row['id']}")
            ids.add(row["id"])
            if row["label"] != 1 or row["source"] != "amazon_polarity":
                raise AssertionError(f"Unexpected label/source in {row['id']}")
            source_key = (row["source_split"], int(row["source_index"]))
            if source_key in source_keys:
                raise AssertionError(f"Source row appears in multiple examples: {source_key}")
            source_keys.add(source_key)
            for text in [row["prompt"], row["continuation"], row["prompt"] + row["continuation"]]:
                fp = text_fingerprint(text)
                if fp in exact_benchmark:
                    raise AssertionError(f"Benchmark exact leakage in {row['id']}")
            full_fp = text_fingerprint(row["prompt"] + row["continuation"])
            if full_fp in fingerprints:
                raise AssertionError(f"Duplicate normalized full text: {row['id']}")
            fingerprints.add(full_fp)
    print(json.dumps({"status": "ok", "counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
