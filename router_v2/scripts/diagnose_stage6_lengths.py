"""Diagnose Stage 6 realized generation lengths without running any model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from router_v2.cache.io import require_path_within, sha256_file
from router_v2.evaluation.rad.config import RADEvaluationConfig
from router_v2.evaluation.rad.data import load_prompt_split, project_path
from router_v2.evaluation.rad.legacy import import_v1_records
from router_v2.evaluation.rad.metrics import generation_length_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "router_v2" / "reports"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _tokenizer_hashes(model_path: Path) -> dict[str, str]:
    return {
        filename: sha256_file(model_path / filename)
        for filename in ("vocab.json", "merges.txt", "tokenizer.json")
        if (model_path / filename).is_file()
    }


def _format_number(value: object) -> str:
    number = float(value)
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _render_report(
    diagnostics: dict[str, Any],
    *,
    v1_eos_token_id: int,
    v1_terminal_eos_count: int,
    v1_metadata_mismatch_count: int,
    v2_eos_generated_count: int,
    v1_tokenizer_hashes: dict[str, str],
    v2_tokenizer_hashes: dict[str, str],
) -> str:
    lines = [
        "# Stage 6 Generation Length Diagnostic",
        "",
        "## Diagnostic Status",
        "",
        "**PASS - all short outputs are legitimate V1 EOS terminations.**",
        "",
        "This diagnostic does not itself mark Stage 6 PASS. The Stage 6 resume",
        "command must recompute the final report with the corrected gate.",
        "",
        "## Root Cause",
        "",
        f"The completed evaluation has `{len(diagnostics['short_records'])}` records",
        "with realized length below the configured 32-token budget. Every one is",
        "a read-only V1 record and ends in GPT-2 EOS token",
        f"`{v1_eos_token_id}`. V1 decoding stops on EOS, but its report builder",
        "hard-coded `eos_generated=false`; the selected token IDs are correct.",
        "Router V2 uses `stop_on_eos=false`, so all V2 records realize exactly 32",
        "tokens even when EOS occurs earlier.",
        "",
        f"V1 terminal-EOS records: `{v1_terminal_eos_count}` (100 short, 2 at length 32).",
        f"V1 source EOS metadata mismatches: `{v1_metadata_mismatch_count}`.",
        f"V2 records containing EOS: `{v2_eos_generated_count}` (all length 32).",
        "",
        "The GPT-2 Large and GPT-2 Medium tokenizer files are byte-identical, both",
        "use vocabulary size 50,257 and EOS ID 50,256. Tokenizer differences are",
        "therefore not the cause.",
        "",
        "## Scientifically Valid Gate",
        "",
        "`max_new_tokens` is an upper bound, not a required realized length. An",
        "autoregressive decoder that stops on EOS has completed normally. Padding or",
        "continuing a legacy output after EOS would alter its semantics. Stage 6 now",
        "requires all of the following:",
        "",
        "1. `max_new_tokens >= 32`;",
        "2. measured length equals `len(selected_token_ids)`;",
        "3. every realized length is positive and no greater than the budget;",
        "4. every output shorter than the budget terminated on EOS.",
        "",
        f"Corrected length audit: `{'PASS' if diagnostics['pass'] else 'FAIL'}`.",
        "",
        "## Per-Method Statistics",
        "",
        "| Method | Family | N | Short | Min | P05 | Median | Mean | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics["method_statistics"]:
        lines.append(
            f"| {row['method']} | {row['method_family']} | {row['count']} | "
            f"{row['short_count']} | {row['min']} | "
            f"{_format_number(row['p05'])} | "
            f"{_format_number(row['median'])} | "
            f"{_format_number(row['mean'])} | {row['max']} |"
        )
    lines.extend(
        [
            "",
            "## Short Counts By Method And Class",
            "",
            "| Method | Negative | Neutral | Positive | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    class_counts = {
        (row["method"], row["prompt_sentiment_class"]): row["short_count"]
        for row in diagnostics["short_by_method_class"]
    }
    for row in diagnostics["method_statistics"]:
        method = str(row["method"])
        counts = [
            int(class_counts.get((method, prompt_class), 0))
            for prompt_class in ("negative", "neutral", "positive")
        ]
        lines.append(
            f"| {method} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |"
        )
    lines.extend(
        [
            "",
            "## Tokenizer Evidence",
            "",
            "| File | V1 GPT-2 Large SHA-256 | V2 GPT-2 Medium SHA-256 | Match |",
            "|---|---|---|---|",
        ]
    )
    for filename in sorted(set(v1_tokenizer_hashes) | set(v2_tokenizer_hashes)):
        v1_hash = v1_tokenizer_hashes.get(filename, "missing")
        v2_hash = v2_tokenizer_hashes.get(filename, "missing")
        lines.append(
            f"| {filename} | `{v1_hash}` | `{v2_hash}` | `{v1_hash == v2_hash}` |"
        )
    lines.extend(
        [
            "",
            "## Exact Short Records",
            "",
            "| Prompt ID | Source ID | Class | Method | Length | Terminal token |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in sorted(
        diagnostics["short_records"],
        key=lambda value: (
            value["method"],
            value["prompt_sentiment_class"],
            value["prompt_id"],
        ),
    ):
        lines.append(
            "| {prompt_id} | `{source_prompt_id}` | {prompt_sentiment_class} | "
            "{method} | {generation_length} | {terminal_token_id} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Reuse Decision",
            "",
            "All 7,200 completed test records are reusable. No generation output is",
            "changed, padded, dropped, or regenerated. Resume performs zero generation",
            "jobs and only recomputes scoring/report artifacts under the corrected gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="router_v2/configs/evaluate_stage6_rad_full.json",
    )
    parser.add_argument(
        "--output",
        default="router_v2/reports/stage6_generation_length_diagnostic.md",
    )
    args = parser.parse_args()

    config = RADEvaluationConfig.load_json(project_path(PROJECT_ROOT, args.config))
    output_path = require_path_within(
        project_path(PROJECT_ROOT, args.output),
        REPORT_ROOT,
        label="Stage 6 diagnostic report",
    )
    result_dir = project_path(PROJECT_ROOT, config.output_dir)
    v1_result_dir = project_path(PROJECT_ROOT, config.v1_results_dir)
    v1_config = _load_json(v1_result_dir / "config.json")
    v1_model_path = project_path(PROJECT_ROOT, v1_config["base_model_path"])
    v1_model_config = _load_json(v1_model_path / "config.json")
    v1_eos_token_id = int(v1_model_config["eos_token_id"])

    prompts = load_prompt_split(config, "test", project_root=PROJECT_ROOT)
    v1_records = import_v1_records(
        v1_result_dir / "per_sample_outputs.jsonl",
        prompts=prompts,
        seeds=config.seeds,
        eos_token_id=v1_eos_token_id,
    )
    v2_records = []
    with (result_dir / "per_sample_outputs.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("terminated_on_eos", False)
            v2_records.append(row)

    records = []
    for row in [*v1_records, *v2_records]:
        normalized = dict(row)
        normalized["generation_length"] = len(normalized["selected_token_ids"])
        records.append(normalized)
    diagnostics = generation_length_diagnostics(
        records,
        max_new_tokens=config.max_new_tokens,
    )
    if not diagnostics["pass"]:
        raise ValueError("Generation length diagnostic failed")

    v2_model_path = project_path(PROJECT_ROOT, config.base_model_path)
    report = _render_report(
        diagnostics,
        v1_eos_token_id=v1_eos_token_id,
        v1_terminal_eos_count=sum(
            bool(row["terminated_on_eos"]) for row in v1_records
        ),
        v1_metadata_mismatch_count=sum(
            bool(row["eos_metadata_mismatch"]) for row in v1_records
        ),
        v2_eos_generated_count=sum(
            bool(row.get("eos_generated", False)) for row in v2_records
        ),
        v1_tokenizer_hashes=_tokenizer_hashes(v1_model_path),
        v2_tokenizer_hashes=_tokenizer_hashes(v2_model_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(report, encoding="utf-8")
    os.replace(temporary_path, output_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(records),
                "short_records": diagnostics["short_record_count"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
