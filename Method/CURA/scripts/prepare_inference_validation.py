#!/usr/bin/env python3
"""Create a fixed, balanced CURA inference-tuning split without train/test overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text(value) -> str:
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def prompt_hash(row: dict) -> str:
    prompt = text(row.get("prompt")).strip()
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="dataset/cura_astar_source/train.jsonl")
    parser.add_argument("--validation", default="dataset/cura_astar_source/validation.jsonl")
    parser.add_argument("--final-test", default="dataset/rad_benchmark/all.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be positive")

    train_path = resolve(args.train)
    validation_path = resolve(args.validation)
    final_test_path = resolve(args.final_test)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    final_test_rows = load_jsonl(final_test_path)
    forbidden = {prompt_hash(row) for row in train_rows + final_test_rows}

    eligible = {0: [], 1: []}
    seen = set()
    excluded_overlap = 0
    excluded_duplicate = 0
    for row in validation_rows:
        digest = prompt_hash(row)
        if digest in forbidden:
            excluded_overlap += 1
            continue
        if digest in seen:
            excluded_duplicate += 1
            continue
        seen.add(digest)
        label = int(row["label"])
        if label not in eligible:
            continue
        eligible[label].append(row)

    rng = random.Random(args.seed)
    selected = []
    for label in (0, 1):
        rng.shuffle(eligible[label])
        if len(eligible[label]) < args.per_class:
            raise RuntimeError(
                f"Only {len(eligible[label])} eligible validation rows for label {label}"
            )
        selected.extend(eligible[label][: args.per_class])
    rng.shuffle(selected)

    subset_path = output_dir / "validation_prompts.jsonl"
    temporary = subset_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(subset_path)

    manifest = {
        "seed": args.seed,
        "per_class": args.per_class,
        "rows": len(selected),
        "counts": {str(label): sum(int(row["label"]) == label for row in selected) for label in (0, 1)},
        "excluded_train_or_final_test_overlap": excluded_overlap,
        "excluded_validation_duplicates": excluded_duplicate,
        "sources": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
            "final_test": {"path": str(final_test_path), "sha256": sha256_file(final_test_path)},
        },
        "output": {"path": str(subset_path), "sha256": sha256_file(subset_path)},
        "selected_ids": [row.get("id") for row in selected],
        "selected_prompt_sha256": [prompt_hash(row) for row in selected],
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("seed", "rows", "counts", "excluded_train_or_final_test_overlap", "excluded_validation_duplicates", "output")}, indent=2))


if __name__ == "__main__":
    main()
