"""Build a non-destructive recovery audit from evidence retained in Git.

This is deliberately a standard-library-only preflight.  It never treats
missing ignored checkpoints/models as a successful measurement and never
promotes the experiment to retraining or staged validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "parm_taro" / "recovery" / "diagnosis"


def read_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: Path) -> dict[str, Any]:
    records: list[str] = []
    total = 0
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix())
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        records.append(f"{relative}\t{size}\t{sha256_file(item)}\n")
        total += size
    return {
        "path": str(path),
        "present": path.exists(),
        "file_count": len(files),
        "total_bytes": total,
        "tree_sha256": hashlib.sha256("".join(records).encode()).hexdigest(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    collapse = read_json("PARM_TARO/reports/stage9_alpha_collapse_diagnostic.json")
    alpha_path = read_json("PARM_TARO/reports/stage9_alpha_path_v2_diagnostic.json")
    run = read_json("results/parm_taro/training/v2_alpha_preference/run_status.json")
    stage10 = read_json("results/parm_taro/evaluation/full/method_aggregates.json")
    router_config = read_json("PARM_TARO/configs/router_v2_alpha_tulu2.json")

    protected = {
        "created_for": "D0 pre-change snapshot",
        "trees_available_in_this_snapshot": {
            name: hash_tree(ROOT / relative)
            for name, relative in {
                "parm": "PARM",
                "router_v1": "router",
                "method_rad": "Method/RAD",
                "stage9_training_records": "results/parm_taro/training",
                "stage10_evaluation": "results/parm_taro/evaluation/full",
            }.items()
        },
        "required_ignored_assets": {
            relative: (ROOT / relative).exists()
            for relative in (
                "models/tulu-2-7b",
                "results/parm_taro/checkpoints/parm_pku_pblora",
                "results/parm_taro/training/taro/best.pt",
                "results/parm_taro/training/v2_no_alpha/best.pt",
                "results/parm_taro/training/v2_alpha_preference/final.pt",
            )
        },
        "archived_stage10_hashes": read_json("results/parm_taro/evaluation/full/stage10_report.json")["protected_after"],
    }
    protected["complete_for_retraining"] = all(protected["required_ignored_assets"].values())
    write_json("protected_artifacts_before.json", protected)

    write_json(
        "static_equivalence.json",
        {
            "status": "BLOCKED_RUNTIME_ASSETS_MISSING",
            "required_states": 100,
            "measured_states": 0,
            "source_equation": "log_softmax(log_softmax(base_logits) + lambda * log_softmax(guide_logits))",
            "source_audit": {
                "lambda_0_reduces_algebraically_to_base": True,
                "lambda_1_uses_same_function_for_static_and_adaptive_decoding": True,
                "files": [
                    "PARM_TARO/decoding/static_regression.py",
                    "PARM_TARO/decoding/adaptive.py",
                    "PARM_TARO/training/routing.py",
                ],
            },
            "numeric_pass": False,
            "reason": "The ignored Tulu-2-7B/PBLoRA assets and PyTorch environment are absent from this workspace snapshot.",
        },
    )

    full = next(item for item in stage10 if item["method"] == "parm_v2_full_alpha")
    minimum = full["lambda"]["min"]
    maximum = full["lambda"]["max"]
    write_json(
        "lambda_parameterization.json",
        {
            "checkpoint_path": "results/parm_taro/training/v2_alpha_preference/final.pt",
            "checkpoint_present": (
                ROOT / "results/parm_taro/training/v2_alpha_preference/final.pt"
            ).is_file(),
            "checkpoint_sha256_archived": "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20",
            "lambda_max": router_config["lambda_max"],
            "lambda_min_exact": router_config["lambda_max"] * router_config["lambda_eps"],
            "lambda_ceiling_exact": router_config["lambda_max"] * (1.0 - router_config["lambda_eps"]),
            "initial_output_bias_from_config": math.log(router_config["initial_gate"] / (1.0 - router_config["initial_gate"])),
            "trained_output_bias": None,
            "trained_output_bias_status": "BLOCKED_CHECKPOINT_MISSING",
            "raw_output": {
                "mean": None,
                "std": None,
                "p05": None,
                "p50": None,
                "p95": None,
                "min_implied_by_observed_lambda": math.log(minimum / (1.0 - minimum)),
                "max_implied_by_observed_lambda": math.log(maximum / (1.0 - maximum)),
            },
            "sigmoid_gate": full["lambda"],
            "final_lambda": full["lambda"],
            "final_mapping": "lambda_t = 1.0 * clamp(sigmoid(raw), 1e-6, 1-1e-6)",
            "decision": "Gate itself is near zero; lambda_max is not a 0.05 scale cap.",
        },
    )

    sweep_rows: list[dict[str, Any]] = []
    for cell in collapse["fixed_lambda_sweep"]["cells"]:
        sweep_rows.append(
            {
                "evidence_scope": "archived_teacher_forced_validation",
                "alpha_helpfulness": cell["alpha_helpfulness"],
                "alpha_harmlessness": cell["alpha_harmlessness"],
                "lambda": cell["fixed_lambda"],
                "mean_gold_token_nll": cell["dual_response_nll"]["mean"],
                "samples": cell["dual_response_nll"]["count"],
                "hv": "",
                "mip": "",
                "pcs": "",
                "preference_regret": "",
                "ppl": "",
                "note": "Not the File-12 sequence-level val200 sweep",
            }
        )
    fields = list(sweep_rows[0])
    write_csv("constant_lambda_sweep.csv", fields, sweep_rows)
    write_csv("gold_token_utility.csv", fields, sweep_rows)

    write_csv(
        "token_gradient_audit.csv",
        ["status", "split", "tokens", "fraction_dL_dlambda_positive_at_0", "reason"],
        [{
            "status": "BLOCKED_RUNTIME_ASSETS_MISSING",
            "split": "train+validation",
            "tokens": 0,
            "fraction_dL_dlambda_positive_at_0": "",
            "reason": "Exact full-vocabulary token distributions were not retained in Git; aggregate fixed-lambda NLL evidence cannot recover the requested per-token fraction.",
        }],
    )

    failure = read_json("PARM_TARO/reports/stage9_alpha_pilot_v1_failure_diagnostic.json")
    ablation_rows = []
    for name, value in failure["initialization_ablation"].items():
        gate = value.get("gate_parameterization", {})
        ablation_rows.append({
            "evidence_scope": "archived_stage9_initialization_ablation",
            "variant": name,
            "mean_lambda": gate.get("mean_lambda", ""),
            "mean_sigmoid_derivative": gate.get("mean_sigmoid_derivative", ""),
            "nll_only": "",
            "entropy": "",
            "smoothness": "",
            "strength": "",
            "note": "File-12 equal-step regularizer ablation remains blocked",
        })
    write_csv("regularizer_ablation.csv", list(ablation_rows[0]), ablation_rows)

    before = run["validation_before"]
    after = run["validation_after"]
    write_csv(
        "alpha_counterfactual.csv",
        ["scope", "states_or_tokens", "mean_lambda", "lambda_std", "endpoint_alpha_delta", "correct_vs_shuffled_delta", "alpha_encoder_gradient_norm", "fusion_alpha_gradient_norm", "note"],
        [
            {
                "scope": "archived_teacher_forced_before_production",
                "states_or_tokens": before["correct_lambda"]["count"],
                "mean_lambda": before["correct_lambda"]["mean"],
                "lambda_std": before["correct_lambda"]["std"],
                "endpoint_alpha_delta": before["endpoint_alpha_lambda_delta"]["mean"],
                "correct_vs_shuffled_delta": before["correct_vs_shuffled_lambda_delta"]["mean"],
                "alpha_encoder_gradient_norm": alpha_path["gradient_scale"]["preference"]["preference_encoder"]["mean"],
                "fusion_alpha_gradient_norm": alpha_path["gradient_scale"]["preference"]["fusion_layers"]["mean"],
                "note": "same-state alpha evidence retained",
            },
            {
                "scope": "archived_teacher_forced_after_production",
                "states_or_tokens": after["correct_lambda"]["count"],
                "mean_lambda": after["correct_lambda"]["mean"],
                "lambda_std": after["correct_lambda"]["std"],
                "endpoint_alpha_delta": after["endpoint_alpha_lambda_delta"]["mean"],
                "correct_vs_shuffled_delta": after["correct_vs_shuffled_lambda_delta"]["mean"],
                "alpha_encoder_gradient_norm": "",
                "fusion_alpha_gradient_norm": "",
                "note": "production checkpoint itself is absent",
            },
        ],
    )

    generated = full["lambda"]
    write_json(
        "exposure_shift.json",
        {
            "status": "PARTIAL_AGGREGATES_ONLY",
            "teacher_forced_validation_lambda": after["correct_lambda"],
            "generated_test_lambda": generated,
            "mean_lambda_ratio_generated_over_teacher_forced": generated["mean"] / after["correct_lambda"]["mean"],
            "feature_distribution_audit": "BLOCKED_GENERATION_RECORDS_AND_RUNTIME_MISSING",
            "warning": "The generated aggregate is Stage-10 test evidence and is diagnostic only; it must not tune the recovery fix.",
        },
    )


if __name__ == "__main__":
    main()
