#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def record_key(record):
    if record.get("md5_hash"):
        return ("md5", str(record["md5_hash"]))
    if record.get("id") is not None:
        return ("id", str(record["id"]))
    return ("prompt", str(record.get("prompt", "")))


def score_of(record, fields):
    for field in fields:
        value = record.get(field)
        if value is not None:
            return float(value)
    raise KeyError(f"Missing score field in record. Tried: {fields}")


def main():
    parser = argparse.ArgumentParser(description="Compute fixed-best and per-prompt oracle routing gap.")
    parser.add_argument("--results", nargs="+", required=True, help="Result JSON/JSONL files.")
    parser.add_argument("--score-fields", nargs="+", default=["score", "reward", "rm_reward", "sentiment"])
    parser.add_argument("--output", default="Method/MultiSignal/reports/oracle_study.csv")
    args = parser.parse_args()

    methods = {}
    for raw_path in args.results:
        path = Path(raw_path)
        records = [r for r in load_records(path) if r.get("status", "success") == "success"]
        method = records[0].get("method", path.stem) if records else path.stem
        methods[method] = {record_key(record): score_of(record, args.score_fields) for record in records}

    shared_keys = set.intersection(*(set(scores) for scores in methods.values()))
    if not shared_keys:
        raise RuntimeError("No shared prompt keys across result files.")

    fixed_means = {
        method: sum(scores[key] for key in shared_keys) / len(shared_keys)
        for method, scores in methods.items()
    }
    fixed_best_method, fixed_best_score = max(fixed_means.items(), key=lambda item: item[1])

    oracle_scores = []
    winners = defaultdict(int)
    for key in shared_keys:
        winner, score = max(((method, scores[key]) for method, scores in methods.items()), key=lambda item: item[1])
        oracle_scores.append(score)
        winners[winner] += 1

    oracle_score = sum(oracle_scores) / len(oracle_scores)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["shared_prompts", len(shared_keys)])
        writer.writerow(["fixed_best_method", fixed_best_method])
        writer.writerow(["fixed_best_score", f"{fixed_best_score:.6f}"])
        writer.writerow(["oracle_score", f"{oracle_score:.6f}"])
        writer.writerow(["oracle_gap", f"{oracle_score - fixed_best_score:.6f}"])
        for method, count in sorted(winners.items()):
            writer.writerow([f"oracle_winner_count/{method}", count])

    print(f"Shared prompts: {len(shared_keys)}")
    print(f"Fixed best: {fixed_best_method} = {fixed_best_score:.6f}")
    print(f"Oracle: {oracle_score:.6f}; gap = {oracle_score - fixed_best_score:.6f}")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
