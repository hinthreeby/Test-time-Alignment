from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.cache_io import ShardedCuraDataset, verify_cache
from Method.CURA.core.config import PROJECT_ROOT


def mean(values):
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser(description="Token/step oracle gap on held-out CURA cache targets.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", default="Method/CURA/reports/token_oracle.json")
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute(): cache_dir = PROJECT_ROOT / cache_dir
    verify_cache(cache_dir)
    dataset = ShardedCuraDataset(cache_dir)
    names = dataset.manifest["signals"]
    expert_utilities = {name: [] for name in names}
    oracle_utilities, base_utilities = [], []
    winners = {name: 0 for name in names}
    for row in dataset:
        target = row.get("target_utilities")
        if target is None: continue
        target = target.float()
        base_utilities.append(float(target[row["base_logits"].argmax()]))
        oracle_utilities.append(float(target.max()))
        step_values = []
        for index, name in enumerate(names):
            value = float(target[row["raw_scores"][index].argmax()])
            expert_utilities[name].append(value); step_values.append(value)
        winners[names[max(range(len(names)), key=lambda i: step_values[i])]] += 1
    if not oracle_utilities: raise RuntimeError("Cache has no held-out target_utilities")
    expert_means = {name: mean(values) for name, values in expert_utilities.items()}
    fixed_best = max(expert_means, key=expert_means.get)
    report = {
        "steps": len(oracle_utilities), "base_utility": mean(base_utilities),
        "expert_utilities": expert_means, "fixed_best": fixed_best,
        "fixed_best_utility": expert_means[fixed_best], "token_oracle_utility": mean(oracle_utilities),
        "fixed_to_token_oracle_gap": mean(oracle_utilities) - expert_means[fixed_best],
        "expert_win_counts": winners, "target_utility": dataset.manifest.get("target_utility"),
    }
    output = Path(args.output)
    if not output.is_absolute(): output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
