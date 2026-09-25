#!/usr/bin/env python3
"""Combine CURA tuning reports and generation telemetry into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = ("learned", "lambda2", "lambda3", "lambda4", "gate075", "no_gate")
TELEMETRY_FIELDS = (
    "mean_gate",
    "mean_strength_pre_projection",
    "mean_strength_post_projection",
    "mean_pre_projection_kl",
    "mean_kl",
    "kl_limit_hit_rate",
    "signal_calls_per_token",
    "generated_tokens_per_second",
)


def finite_mean(rows: list[dict], field: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.experiment_dir.resolve()
    output = (args.output or root / "comparison.csv").resolve()
    combined = []

    for variant in VARIANTS:
        report_path = root / "reports" / variant / "evaluation_report.csv"
        generation_path = root / "outputs" / f"cura_{variant}.json"
        if not report_path.exists() or not generation_path.exists():
            raise FileNotFoundError(
                f"Missing report/output for {variant}: {report_path}, {generation_path}"
            )
        with report_path.open(newline="", encoding="utf-8-sig") as handle:
            report_rows = list(csv.DictReader(handle))
        if len(report_rows) != 1:
            raise ValueError(f"Expected one evaluation row in {report_path}")
        generation_rows = json.loads(generation_path.read_text(encoding="utf-8"))
        successful = [
            row for row in generation_rows
            if row.get("status", "success") == "success"
        ]
        row = {"Variant": variant, **report_rows[0]}
        row.pop("Method", None)
        row["Successful Outputs"] = len(successful)
        for field in TELEMETRY_FIELDS:
            row[f"Telemetry {field}"] = finite_mean(successful, field)
        combined.append(row)

    fields = []
    for row in combined:
        for field in row:
            if field not in fields:
                fields.append(field)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved {len(combined)} variants -> {output}")


if __name__ == "__main__":
    main()
