#!/usr/bin/env python3
"""Build a tie-aware binary TokenRouter dataset from scored w rollouts.

This script consumes the state features and counterfactual utilities produced
by PARM_TARO's token-headroom experiment.  It deliberately does not accept the
dynamic-alpha objective-probe JSONL: objective rewards are not intervention
labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HEADROOM = (
    ROOT
    / "Method/PARM_TARO/results/parm_taro/adaptive_parm/04_token_headroom"
)
DEFAULT_FEATURES = HEADROOM / "state_features.csv"
DEFAULT_UTILITY = HEADROOM / "state_utility.csv"
DEFAULT_OUTPUT = ROOT / "results/w_oracle_train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--state-utility", type=Path, default=DEFAULT_UTILITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-weight", type=float, default=0.0,
                        help="No-PARM action used as the utility baseline.")
    parser.add_argument("--guided-weight", type=float, default=1.0,
                        help="PARM action represented by gate label 1.")
    parser.add_argument("--tie-epsilon", type=float, default=0.005)
    parser.add_argument("--compute-cost", type=float, default=0.0,
                        help="Utility cost subtracted from the guided action.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Must match the top-k mass in state_features.csv.")
    parser.add_argument("--allow-weak-gate", action="store_true",
                        help="Allow export even when the recorded oracle verdict is WEAK.")
    args = parser.parse_args()
    if args.tie_epsilon < 0 or args.compute_cost < 0:
        parser.error("--tie-epsilon and --compute-cost must be non-negative")
    if args.top_k < 2:
        parser.error("--top-k must be >= 2")
    if args.reference_weight == args.guided_weight:
        parser.error("reference and guided weights must differ")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key!r} in state {row.get('state_id')!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in state {row.get('state_id')!r}")
    return value


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verdict_guard(args: argparse.Namespace) -> str | None:
    report = args.state_features.parent / "report.md"
    if not report.is_file():
        return None
    text = report.read_text(encoding="utf-8")
    marker = "Verdict: **"
    verdict = text.split(marker, 1)[1].split("**", 1)[0] if marker in text else None
    if verdict and "WEAK" in verdict and not args.allow_weak_gate:
        raise RuntimeError(
            f"Recorded oracle verdict is {verdict}. README requires Gate E to pass "
            "before router training. Re-run a passing oracle experiment, or use "
            "--allow-weak-gate only for an engineering smoke test."
        )
    return verdict


def main() -> None:
    args = parse_args()
    verdict = verdict_guard(args)
    features = read_csv(args.state_features)
    utilities = read_csv(args.state_utility)

    by_state: dict[str, dict[float, float]] = defaultdict(dict)
    for row in utilities:
        state_id = str(row.get("state_id", ""))
        weight = finite_float(row, "weight")
        by_state[state_id][weight] = finite_float(row, "mip")

    rows: list[dict[str, Any]] = []
    tied = missing = 0
    for state in features:
        state_id = str(state.get("state_id", ""))
        actions = by_state.get(state_id, {})
        if args.reference_weight not in actions or args.guided_weight not in actions:
            missing += 1
            continue
        advantage = (
            actions[args.guided_weight]
            - actions[args.reference_weight]
            - args.compute_cost
        )
        if abs(advantage) <= args.tie_epsilon:
            tied += 1
            continue

        # compact_v1 is intentionally base-side only.  Deficits and prev_gate
        # were not recorded by the historical headroom run, so they are zero.
        feature_values = [
            finite_float(state, "base_entropy"),
            finite_float(state, "base_top1_probability"),
            finite_float(state, "base_top1_top2_margin"),
            finite_float(state, "base_topk_mass"),
            finite_float(state, "normalized_generation_position"),
            finite_float(state, "alpha_helpfulness"),
            finite_float(state, "alpha_harmlessness"),
            0.0,
            0.0,
            0.0,
        ]
        rows.append({
            "schema_version": 1,
            "feature_schema": "compact_v1",
            "top_k": args.top_k,
            "num_objectives": 2,
            "state_id": state_id,
            "sample_id": str(state.get("sample_id", "")),
            "feature_values": feature_values,
            "gate_label": int(advantage > 0),
            "guided_advantage": advantage,
            "utility_w0": actions[args.reference_weight],
            "utility_w1": actions[args.guided_weight],
            "tie_epsilon": args.tie_epsilon,
            "compute_cost": args.compute_cost,
        })

    if not rows:
        raise RuntimeError("No non-tied router examples were produced")
    atomic_jsonl(args.out, rows)
    positives = sum(row["gate_label"] for row in rows)
    print(f"Oracle verdict: {verdict or 'not recorded'}")
    print(f"States: {len(features)} | usable: {len(rows)} | ties: {tied} | missing: {missing}")
    print(f"Labels: invoke={positives} skip={len(rows) - positives}")
    print(args.out)


if __name__ == "__main__":
    main()
