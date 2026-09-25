"""Build balanced, benchmark-disjoint CURA sentiment data from local SST-2 parquet files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.config import PROJECT_ROOT


def normalized(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_parquet(download_dir, split):
    matches = []
    for metadata_path in download_dir.glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if f"/data/{split}-" in metadata.get("url", ""):
            matches.append(metadata_path.with_suffix(""))
    if len(matches) != 1 or not matches[0].exists():
        raise RuntimeError(f"Expected one local SST-2 {split} parquet file, found: {matches}")
    return matches[0]


def benchmark_fingerprints(directory):
    fingerprints = set()
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("prompt", "continuation", "reference"):
                    value = row.get(field)
                    value = value.get("text", "") if isinstance(value, dict) else value
                    if value:
                        fingerprints.add(normalized(value))
    return fingerprints


def prepare_split(frame, tokenizer, split, per_class, seed, benchmark):
    rng = random.Random(seed + (0 if split == "train" else 1))
    selected, removed_overlap, removed_short = {0: [], 1: []}, 0, 0
    for label in (0, 1):
        indexes = frame.index[frame["label"] == label].tolist()
        rng.shuffle(indexes)
        for index in indexes:
            sentence = str(frame.at[index, "sentence"]).strip()
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            if len(token_ids) < 6:
                removed_short += 1
                continue
            boundary = max(2, min(len(token_ids) - 2, round(len(token_ids) * 0.5)))
            prompt = tokenizer.decode(token_ids[:boundary])
            continuation = tokenizer.decode(token_ids[boundary:])
            if any(normalized(value) in benchmark for value in (sentence, prompt, continuation)):
                removed_overlap += 1
                continue
            source_index = int(frame.at[index, "idx"])
            selected[label].append({
                "id": f"sst2-{split}-{label}-{source_index:06d}",
                "prompt": prompt,
                "response": continuation,
                "label": label,
                "source": "sst2",
                "source_split": split,
                "source_index": source_index,
            })
            if per_class is not None and len(selected[label]) >= per_class:
                break
    rows = []
    for pair in zip(selected[0], selected[1]):
        rows.extend(pair)
    longer = selected[0] if len(selected[0]) > len(selected[1]) else selected[1]
    rows.extend(longer[len(rows) // 2:])
    return rows, {
        "counts": {str(label): len(values) for label, values in selected.items()},
        "removed_exact_benchmark_overlap": removed_overlap,
        "removed_short": removed_short,
    }


def atomic_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", default="dataset/RAD_train/sst2_hf/downloads")
    parser.add_argument("--benchmark-dir", default="dataset/rad_benchmark")
    parser.add_argument("--tokenizer", default="models/gpt2-large")
    parser.add_argument("--output-dir", default="dataset/cura_astar_source")
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--validation-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value
    downloads, benchmark_dir = resolve(args.downloads), resolve(args.benchmark_dir)
    output_dir, tokenizer_path = resolve(args.output_dir), resolve(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    benchmark = benchmark_fingerprints(benchmark_dir)
    report = {"seed": args.seed, "benchmark_fingerprints": len(benchmark), "splits": {}}
    for split, limit in (("train", args.train_per_class), ("validation", args.validation_per_class)):
        source = discover_parquet(downloads, split)
        frame = pd.read_parquet(source)
        rows, stats = prepare_split(frame, tokenizer, split, limit, args.seed, benchmark)
        output = output_dir / f"{split}.jsonl"
        atomic_jsonl(rows, output)
        report["splits"][split] = {
            **stats, "rows": len(rows), "source": portable_path(source),
            "source_sha256": sha256_file(source), "output": portable_path(output),
            "output_sha256": sha256_file(output),
        }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
