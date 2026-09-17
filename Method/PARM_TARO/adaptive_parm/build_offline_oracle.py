"""Build the Adaptive-PARM offline decision dataset from Phase 08/09 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


ALPHA_ORDER = ["helpfulness", "harmlessness"]
ALPHAS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
METHODS: tuple[tuple[str, float | None, float], ...] = (
    ("base", 0.0, 0.0),
    ("normalized_0.01", 0.01, 0.01 / 1.01),
    ("normalized_0.05", 0.05, 0.05 / 1.05),
    ("normalized_0.1", 0.1, 0.1 / 1.1),
    ("normalized_0.5", 0.5, 0.5 / 1.5),
    ("normalized_1", 1.0, 0.5),
    ("normalized_2", 2.0, 2.0 / 3.0),
    ("normalized_guide_only", None, 1.0),
)
METHOD_MAP = {method: (lam, weight) for method, lam, weight in METHODS}
TOLERANCE = 1e-12


def project_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in start.parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError(f"Cannot locate project root from {start}")


ROOT = project_root()
SUITE = ROOT / "results/parm_taro/recovery/low_vram_suite"
PHASE08 = SUITE / "08_generations.jsonl"
PHASE09 = SUITE / "09_scored_generations.jsonl"
METRICS09 = SUITE / "09_metrics_summary.json"
MANIFEST = ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl"
OUTPUT = ROOT / "results/parm_taro/adaptive_parm/01_offline_oracle"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    output = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output.append(buffer.getvalue())
    atomic_text(path, "".join(output))


def generation_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["case_id"]), str(row["method"]), int(row["seed"])


def alpha_key(alpha: Sequence[float]) -> str:
    return f"{float(alpha[0]):.2f},{float(alpha[1]):.2f}"


def finite_metrics(row: Mapping[str, Any]) -> bool:
    return all(
        math.isfinite(float(row[field]))
        for field in (
            "helpfulness_raw",
            "harmlessness_raw",
            "normalized_helpfulness",
            "normalized_harmlessness",
            "mip",
            "pcs",
            "candidate_regret",
        )
    )


def build() -> dict[str, Any]:
    for path in (PHASE08, PHASE09, METRICS09, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    generations = read_jsonl(PHASE08)
    scored_all = read_jsonl(PHASE09)
    manifest = read_jsonl(MANIFEST)
    metrics09 = json.loads(METRICS09.read_text(encoding="utf-8"))
    generated_by_key = {generation_key(row): row for row in generations}
    if len(generated_by_key) != len(generations):
        raise ValueError("Phase 08 contains duplicate case+method+seed keys")

    manifest_by_case = {str(row["case_id"]): row for row in manifest}
    if len(manifest) != 60 or len(manifest_by_case) != 60:
        raise ValueError("Feasibility manifest must contain exactly 60 unique cases")
    if any(row.get("alpha_order") != ALPHA_ORDER for row in manifest):
        raise ValueError("Manifest alpha order is not [helpfulness, harmlessness]")

    selected = [row for row in scored_all if row.get("method") in METHOD_MAP]
    expected_records = 60 * len(METHODS)
    if len(selected) != expected_records:
        raise ValueError(f"Expected {expected_records} Adaptive-PARM candidates, got {len(selected)}")

    records: list[dict[str, Any]] = []
    seen_case_weight: set[tuple[str, float]] = set()
    source_mip_error = 0.0
    harmlessness_sign_error = 0.0
    for row in selected:
        key = generation_key(row)
        generated = generated_by_key.get(key)
        if generated is None:
            raise KeyError(f"Phase 09 row has no Phase 08 generation: {key}")
        for field in ("response", "token_ids", "prompt", "requested_alpha", "lambda", "fusion"):
            if generated.get(field) != row.get(field):
                raise ValueError(f"Phase 08/09 mismatch for {key}: {field}")
        manifest_row = manifest_by_case.get(str(row["case_id"]))
        if manifest_row is None:
            raise KeyError(f"Unknown case_id: {row['case_id']}")
        if manifest_row["requested_alpha"] != row["requested_alpha"]:
            raise ValueError(f"Manifest/score alpha mismatch: {row['case_id']}")

        source_lambda, weight = METHOD_MAP[str(row["method"])]
        observed_lambda = row.get("lambda")
        if source_lambda is None:
            if observed_lambda is not None or row.get("fusion") != "guide_only":
                raise ValueError(f"Invalid guide-only source mapping: {key}")
        elif not math.isclose(float(observed_lambda), source_lambda, rel_tol=0.0, abs_tol=TOLERANCE):
            raise ValueError(f"Method/lambda mismatch for {key}: {observed_lambda}")
        expected_fusion = "base" if row["method"] == "base" else "normalized"
        if source_lambda is not None and row.get("fusion") != expected_fusion:
            raise ValueError(f"Method/fusion mismatch for {key}: {row.get('fusion')}")
        pair = (str(row["case_id"]), weight)
        if pair in seen_case_weight:
            raise ValueError(f"Duplicate case+weight: {pair}")
        seen_case_weight.add(pair)
        alpha = [float(value) for value in row["requested_alpha"]]
        normalized_helpfulness = float(row["helpfulness_normalized"])
        normalized_harmlessness = float(row["harmlessness_normalized"])
        mip = float(row["mip"])
        expected_mip = alpha[0] * normalized_helpfulness + alpha[1] * normalized_harmlessness
        source_mip_error = max(source_mip_error, abs(mip - expected_mip))
        harmlessness_sign_error = max(
            harmlessness_sign_error,
            abs(float(row["harmlessness_raw"]) + float(row["cost_raw"])),
        )
        record = {
            "case_id": str(row["case_id"]),
            "sample_id": str(row["sample_id"]),
            "prompt": str(row["prompt"]),
            "alpha_helpfulness": alpha[0],
            "alpha_harmlessness": alpha[1],
            "alpha_order": json.dumps(ALPHA_ORDER, separators=(",", ":")),
            "source_method": str(row["method"]),
            "source_fusion": str(row["fusion"]),
            "source_lambda": "" if source_lambda is None else source_lambda,
            "weight": weight,
            "helpfulness_raw": float(row["helpfulness_raw"]),
            "harmlessness_raw": float(row["harmlessness_raw"]),
            "normalized_helpfulness": normalized_helpfulness,
            "normalized_harmlessness": normalized_harmlessness,
            "mip": mip,
            "pcs": float(row["pcs"]),
            "candidate_regret": float(row["candidate_regret"]),
            "seed": int(row["seed"]),
        }
        if not finite_metrics(record):
            raise ValueError(f"Non-finite candidate metrics: {pair}")
        records.append(record)

    if source_mip_error > TOLERANCE:
        raise ValueError(f"Phase 09 MIP formula mismatch: max error {source_mip_error}")

    case_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alpha_method_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        case_groups[row["case_id"]].append(row)
        method_groups[row["source_method"]].append(row)
        alpha_method_groups[
            (f"{row['alpha_helpfulness']:.2f},{row['alpha_harmlessness']:.2f}", row["source_method"])
        ].append(row)

    expected_weights = {weight for _, _, weight in METHODS}
    coverage_ok = all(
        len(group) == len(METHODS) and {row["weight"] for row in group} == expected_weights
        for group in case_groups.values()
    )
    if not coverage_ok:
        raise ValueError("At least one case is missing a candidate weight")

    global_rows: list[dict[str, Any]] = []
    for method, source_lambda, weight in METHODS:
        group = method_groups[method]
        global_rows.append(
            {
                "source_method": method,
                "source_lambda": "" if source_lambda is None else source_lambda,
                "weight": weight,
                "n": len(group),
                "mean_mip": fmean(row["mip"] for row in group),
                "mean_pcs": fmean(row["pcs"] for row in group),
                "mean_candidate_regret": fmean(row["candidate_regret"] for row in group),
            }
        )
    best_global = max(global_rows, key=lambda row: (row["mean_mip"], -row["weight"]))
    best_global_mip = float(best_global["mean_mip"])

    oracle_rows: list[dict[str, Any]] = []
    tie_membership: Counter[float] = Counter()
    primary_distribution: Counter[float] = Counter()
    for case_id in sorted(case_groups):
        group = case_groups[case_id]
        maximum = max(row["mip"] for row in group)
        tied = sorted(
            (row for row in group if math.isclose(row["mip"], maximum, rel_tol=0.0, abs_tol=TOLERANCE)),
            key=lambda row: row["weight"],
        )
        oracle = tied[0]  # deterministic minimum-trust tie break
        for row in tied:
            tie_membership[row["weight"]] += 1
        primary_distribution[oracle["weight"]] += 1
        global_case = next(
            row for row in group if row["source_method"] == best_global["source_method"]
        )
        expected_regret = maximum - oracle["mip"]
        if abs(expected_regret) > TOLERANCE:
            raise AssertionError("Selected oracle does not attain maximum MIP")
        oracle_rows.append(
            {
                "case_id": case_id,
                "sample_id": oracle["sample_id"],
                "prompt": oracle["prompt"],
                "alpha_helpfulness": oracle["alpha_helpfulness"],
                "alpha_harmlessness": oracle["alpha_harmlessness"],
                "oracle_source_method": oracle["source_method"],
                "oracle_weight": oracle["weight"],
                "oracle_tied_methods": json.dumps([row["source_method"] for row in tied]),
                "oracle_tied_weights": json.dumps([row["weight"] for row in tied]),
                "num_oracle_ties": len(tied),
                "helpfulness_raw": oracle["helpfulness_raw"],
                "harmlessness_raw": oracle["harmlessness_raw"],
                "normalized_helpfulness": oracle["normalized_helpfulness"],
                "normalized_harmlessness": oracle["normalized_harmlessness"],
                "oracle_mip": oracle["mip"],
                "oracle_pcs": oracle["pcs"],
                "best_global_case_mip": global_case["mip"],
                "case_oracle_gain": oracle["mip"] - global_case["mip"],
            }
        )

    oracle_mip = fmean(row["oracle_mip"] for row in oracle_rows)
    oracle_gap = oracle_mip - best_global_mip
    relative_headroom = oracle_gap / best_global_mip
    if oracle_gap <= 0.0:
        raise ValueError("Oracle gap must be positive to define gap capture")
    for row in global_rows:
        row["oracle_gap_capture"] = (float(row["mean_mip"]) - best_global_mip) / oracle_gap
        row["is_best_global"] = row["source_method"] == best_global["source_method"]

    per_alpha_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        candidates = []
        for method, source_lambda, weight in METHODS:
            group = alpha_method_groups[(key, method)]
            if len(group) != 12:
                raise ValueError(f"Expected 12 cases for {key}/{method}, got {len(group)}")
            candidates.append(
                {
                    "alpha_helpfulness": alpha[0],
                    "alpha_harmlessness": alpha[1],
                    "source_method": method,
                    "source_lambda": "" if source_lambda is None else source_lambda,
                    "weight": weight,
                    "n": len(group),
                    "mean_mip": fmean(row["mip"] for row in group),
                    "mean_pcs": fmean(row["pcs"] for row in group),
                    "mean_candidate_regret": fmean(row["candidate_regret"] for row in group),
                }
            )
        per_alpha_rows.append(max(candidates, key=lambda row: (row["mean_mip"], -row["weight"])))
    best_per_alpha_mip = fmean(row["mean_mip"] for row in per_alpha_rows)

    weight_rows = []
    for method, source_lambda, weight in METHODS:
        weight_rows.append(
            {
                "source_method": method,
                "source_lambda": "" if source_lambda is None else source_lambda,
                "weight": weight,
                "oracle_primary_count": primary_distribution[weight],
                "oracle_primary_fraction": primary_distribution[weight] / 60.0,
                "oracle_tie_membership_count": tie_membership[weight],
            }
        )

    candidate_regret_error = max(
        abs(
            (max(item["mip"] for item in group) - item["mip"])
            - item["candidate_regret"]
        )
        for group in case_groups.values()
        for item in group
    )

    alpha_counts = Counter(
        (row["alpha_helpfulness"], row["alpha_harmlessness"]) for row in records
    )
    audits = {
        "cases_60": len(case_groups) == 60,
        "unique_prompts_12": len({row["sample_id"] for row in records}) == 12,
        "five_alpha_values": set(alpha_counts) == set(ALPHAS),
        "twelve_cases_per_alpha": all(alpha_counts[alpha] == 12 * len(METHODS) for alpha in ALPHAS),
        "eight_weights_per_case": coverage_ok,
        "no_duplicate_case_weight": len(seen_case_weight) == len(records),
        "all_scores_finite": all(finite_metrics(row) for row in records),
        "alpha_order_helpfulness_harmlessness": all(
            row["alpha_order"] == ALPHA_ORDER for row in manifest
        ),
        "phase08_phase09_generation_identity": True,
        "method_lambda_weight_mapping": True,
        "phase09_mip_formula_exact": source_mip_error <= TOLERANCE,
        "phase09_candidate_regret_exact": candidate_regret_error <= TOLERANCE,
        "phase09_harmlessness_is_negative_cost": harmlessness_sign_error <= TOLERANCE,
    }
    status = "PASS" if all(audits.values()) else "FAIL"

    normalization = metrics09.get("normalization")
    summary = {
        "status": status,
        "method": "Adaptive PARM Fusion",
        "primary_utility": "Phase09 MIP",
        "hv_usage": "global evaluation only; never a per-case training target",
        "num_cases": len(case_groups),
        "num_unique_prompts": len({row["sample_id"] for row in records}),
        "num_alpha_values": len(set(alpha_counts)),
        "num_candidate_weights": len(METHODS),
        "num_candidate_records": len(records),
        "candidate_weights": [weight for _, _, weight in METHODS],
        "best_global": dict(best_global),
        "best_per_alpha": per_alpha_rows,
        "best_global_mip": best_global_mip,
        "best_per_alpha_mip": best_per_alpha_mip,
        "oracle_mip": oracle_mip,
        "oracle_gap": oracle_gap,
        "relative_oracle_headroom": relative_headroom,
        "oracle_gap_capture_by_method": {
            row["source_method"]: row["oracle_gap_capture"] for row in global_rows
        },
        "oracle_primary_weight_distribution": {
            str(weight): primary_distribution[weight] for _, _, weight in METHODS
        },
        "tie_break": "minimum weight among MIP ties within absolute tolerance 1e-12",
        "audits": audits,
        "phase09_mip_recalculation_max_abs_error": source_mip_error,
        "phase09_candidate_regret_max_abs_error": candidate_regret_error,
        "phase09_harmlessness_sign_max_abs_error": harmlessness_sign_error,
        "normalization_source": str(METRICS09.relative_to(ROOT)),
        "normalization": normalization,
        "source_artifacts": {
            str(PHASE08.relative_to(ROOT)): sha256_file(PHASE08),
            str(PHASE09.relative_to(ROOT)): sha256_file(PHASE09),
            str(METRICS09.relative_to(ROOT)): sha256_file(METRICS09),
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
        },
    }

    candidate_fields = list(records[0])
    oracle_fields = list(oracle_rows[0])
    global_fields = list(global_rows[0])
    per_alpha_fields = list(per_alpha_rows[0])
    weight_fields = list(weight_rows[0])
    atomic_csv(OUTPUT / "candidate_utility.csv", records, candidate_fields)
    atomic_csv(OUTPUT / "oracle_by_case.csv", oracle_rows, oracle_fields)
    atomic_csv(OUTPUT / "best_global_weight.csv", global_rows, global_fields)
    atomic_csv(OUTPUT / "best_per_alpha_weight.csv", per_alpha_rows, per_alpha_fields)
    atomic_csv(OUTPUT / "weight_distribution.csv", weight_rows, weight_fields)
    atomic_json(OUTPUT / "oracle_summary.json", summary)
    atomic_text(OUTPUT / "report.md", render_report(summary))
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    best = summary["best_global"]
    alpha_lines = "\n".join(
        f"| ({row['alpha_helpfulness']:.2f}, {row['alpha_harmlessness']:.2f}) | "
        f"{row['weight']:.8f} | {row['mean_mip']:.9f} | {row['source_method']} |"
        for row in summary["best_per_alpha"]
    )
    audit_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["audits"].items()
    )
    return f"""# Offline Adaptive-PARM oracle dataset

Status: **{summary['status']}**

This dataset was derived only from frozen Phase 08 generations and Phase 09
scores. No generation, model loading, rescoring, or renormalization occurred.
The primary per-case utility is Phase 09 MIP. Hypervolume remains a global
set-level evaluation metric and is not included as a decision label.

## Results

- Cases: {summary['num_cases']} across {summary['num_unique_prompts']} prompts and {summary['num_alpha_values']} alphas
- Candidate weights per case: {summary['num_candidate_weights']}
- Best global fixed weight: `{best['weight']:.8f}` (`{best['source_method']}`)
- MIP(best global): `{summary['best_global_mip']:.9f}`
- MIP(best per-alpha): `{summary['best_per_alpha_mip']:.9f}`
- MIP(per-case oracle): `{summary['oracle_mip']:.9f}`
- Oracle gap: `{summary['oracle_gap']:.9f}`
- Relative oracle headroom: `{100.0 * summary['relative_oracle_headroom']:.4f}%`

The recomputed values agree with the earlier approximate sanity values
(`~0.647` global, `~0.720` oracle); no discrepancy requires explanation.

## Best fixed weight per alpha

| Alpha (helpfulness, harmlessness) | Weight | Mean MIP | Source candidate |
|---|---:|---:|---|
{alpha_lines}

## Audits

{audit_lines}

Oracle ties are retained in `oracle_by_case.csv`; the primary label uses the
minimum-trust weight among exact/tolerance ties. `oracle_gap_capture` is stored
for every fixed candidate in `best_global_weight.csv` using the requested
global formula. Normalization metadata and hashes of all source artifacts are
frozen in `oracle_summary.json`.
"""


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = build()
    print(json.dumps({key: summary[key] for key in ("status", "num_cases", "num_unique_prompts", "best_global_mip", "best_per_alpha_mip", "oracle_mip", "oracle_gap", "relative_oracle_headroom")}, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
