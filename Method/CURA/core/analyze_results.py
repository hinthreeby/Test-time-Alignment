from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load(path, score_field):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); key = str(row.get("md5_hash") or row.get("prompt_id") or row.get("id"))
            if row.get(score_field) is not None: rows[key] = float(row[score_field])
    return rows


def percentile(values, q):
    values = sorted(values); index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap analysis for scored CURA ablations/seeds.")
    parser.add_argument("--files", nargs="+", required=True, help="NAME=JSONL")
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="Method/CURA/reports/paired_analysis.json")
    args = parser.parse_args()
    methods = {item.split("=", 1)[0]: load(item.split("=", 1)[1], args.score_field) for item in args.files}
    if args.baseline not in methods: raise ValueError("baseline must name one of --files")
    common = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if not common: raise RuntimeError("No paired prompts with scores")
    rng = random.Random(args.seed); baseline = methods[args.baseline]
    report = {"paired_prompts": len(common), "score_field": args.score_field, "baseline": args.baseline, "methods": {}}
    raw_p = {}
    for name, rows in methods.items():
        differences = [rows[key] - baseline[key] for key in common]
        boot = []
        for _ in range(args.bootstrap):
            boot.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
        observed = abs(statistics.fmean(differences))
        permuted = [
            abs(sum(value if rng.random() < 0.5 else -value for value in differences) / len(differences))
            for _ in range(args.permutations)
        ]
        p_value = (1 + sum(value >= observed for value in permuted)) / (args.permutations + 1)
        raw_p[name] = p_value
        difference_std = statistics.stdev(differences) if len(differences) > 1 else 0.0
        report["methods"][name] = {
            "mean": statistics.fmean(rows[key] for key in common), "mean_difference": statistics.fmean(differences),
            "difference_ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
            "probability_improvement": sum(value > 0 for value in boot) / len(boot),
            "paired_permutation_p": p_value,
            "paired_effect_dz": statistics.fmean(differences) / difference_std if difference_std else 0.0,
        }
    comparisons = [(name, value) for name, value in raw_p.items() if name != args.baseline]
    ordered = sorted(comparisons, key=lambda item: item[1])
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, value * (len(ordered) - rank))
        running = max(running, adjusted)
        report["methods"][name]["holm_adjusted_p"] = running
    report["methods"][args.baseline]["holm_adjusted_p"] = 1.0
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
