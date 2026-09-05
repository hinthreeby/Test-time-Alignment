"""Final Stage 9 gate for completed production alpha-preference training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PARM_TARO.training.alpha_production import (
    REQUIRED_PRODUCTION_CHECKS,
    require_pilot_v3_pass,
)
from PARM_TARO.training.constant_control_audit import (
    source_failure_is_constant_only,
)
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import PROJECT_ROOT, project_path
from router_v2.cache.io import sha256_file
from router_v2.training.audit import hash_tree


CONSTANT_CONTROL_AUDIT_PATH = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_constant_alpha_control_audit.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _passed_stage(path: Path) -> tuple[bool, dict[str, Any] | None]:
    if not path.is_file():
        return False, None
    report = _load(path)
    checks = report.get("checks", {})
    return (
        report.get("status") == "PASS"
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        report,
    )


def constant_audit_authorizes_correction(
    production: dict[str, Any],
    audit: dict[str, Any] | None,
    *,
    production_path: Path,
    checkpoint_valid: bool,
) -> bool:
    if audit is None or not checkpoint_valid:
        return False
    checkpoint = production.get("checkpoint", {})
    source = audit.get("source", {})
    gate = audit.get("constant_control", {}).get("gate", {})
    return bool(
        source_failure_is_constant_only(production)
        and audit.get("status") == "PASS"
        and audit.get("corrected_production_status") == "PASS"
        and all(value is True for value in audit.get("checks", {}).values())
        and source.get("run_status_sha256") == sha256_file(production_path)
        and source.get("checkpoint_sha256") == checkpoint.get("sha256")
        and source.get("production_tree")
        == hash_tree(production_path.parent)
        and gate.get("threshold") == 0.001
        and gate.get("threshold_unchanged") is True
        and gate.get("pass") is True
        and float(gate.get("value", 0.0)) >= 0.001
    )


def build_stage9_final_gate(
    config: AlphaPreferenceProductionConfig,
) -> dict[str, Any]:
    pilot_v3 = require_pilot_v3_pass(config)
    training_root = PROJECT_ROOT / "results/parm_taro/training"
    taro_pass, taro = _passed_stage(training_root / "taro/run_status.json")
    no_alpha_pass, no_alpha = _passed_stage(
        training_root / "v2_no_alpha/run_status.json"
    )
    production_path = project_path(config.output_dir) / "run_status.json"
    production = _load(production_path) if production_path.is_file() else None
    audit = _load(CONSTANT_CONTROL_AUDIT_PATH) if CONSTANT_CONTROL_AUDIT_PATH.is_file() else None
    production_checks = production.get("checks", {}) if production else {}
    production_checkpoint_valid = False
    if production:
        checkpoint = production.get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        production_checkpoint_valid = (
            checkpoint_path.is_file()
            and checkpoint.get("sha256") == sha256_file(checkpoint_path)
        )
    native_required_production_checks = all(
        production_checks.get(name) is True for name in REQUIRED_PRODUCTION_CHECKS
    )
    correction_authorized = bool(
        production
        and constant_audit_authorizes_correction(
            production,
            audit,
            production_path=production_path,
            checkpoint_valid=production_checkpoint_valid,
        )
    )
    corrected_required_production_checks = bool(
        production
        and correction_authorized
        and all(
            production_checks.get(name) is True
            for name in REQUIRED_PRODUCTION_CHECKS
            if name != "constant_alpha_differs_materially"
        )
    )
    required_production_checks = (
        native_required_production_checks
        or corrected_required_production_checks
    )
    production_pass = bool(
        production
        and (
            production.get("status") == "PASS"
            or (
                production.get("status") == "NOT_PASS"
                and correction_authorized
            )
        )
        and production.get("method_label")
        == "PARM_TARO_ALPHA_PREFERENCE_PRODUCTION"
        and production.get("split_counts")
        == {"train": 8000, "validation": 500, "test": 0}
        and production.get("test_split_used") is False
        and required_production_checks
        and production_checkpoint_valid
    )
    parm_baseline = _load(PROJECT_ROOT / "PARM_TARO/reports/parm_baseline_hash.json")
    parm_current = hash_tree(PROJECT_ROOT / "PARM")
    checks = {
        "taro_pass": taro_pass,
        "v2_no_alpha_pass": no_alpha_pass,
        "pilot_v3_pass_authorization": pilot_v3.get("status") == "PASS",
        "v2_alpha_preference_production_present": production is not None,
        "v2_alpha_preference_production_pass": production_pass,
        "production_required_checks_pass": required_production_checks,
        "production_checkpoint_hash_valid": production_checkpoint_valid,
        "production_train_count_8000": bool(
            production
            and production.get("split_counts", {}).get("train") == 8000
        ),
        "production_validation_count_500": bool(
            production
            and production.get("split_counts", {}).get("validation") == 500
        ),
        "test_split_unused": bool(
            production
            and production.get("split_counts", {}).get("test") == 0
            and production.get("test_split_used") is False
        ),
        "parm_tree_unchanged": (
            parm_current["tree_sha256"] == parm_baseline["tree_sha256"]
        ),
    }
    if production is None:
        status = "HOST EXECUTION REQUIRED"
    elif all(checks.values()):
        status = "STAGE 9 PASS"
    else:
        status = "STAGE 9 NOT PASS"
    return {
        "schema_version": 1,
        "stage": 9,
        "status": status,
        "checks": checks,
        "inputs": {
            "taro_status": str(training_root / "taro/run_status.json"),
            "v2_no_alpha_status": str(
                training_root / "v2_no_alpha/run_status.json"
            ),
            "production_status": str(production_path),
            "pilot_v3_report": config.required_pilot_v3_report_path,
            "constant_control_audit": str(CONSTANT_CONTROL_AUDIT_PATH),
        },
        "checkpoints": {
            "taro": taro.get("best_checkpoint") if taro else None,
            "v2_no_alpha": no_alpha.get("best_checkpoint") if no_alpha else None,
            "v2_alpha_preference": (
                production.get("checkpoint") if production else None
            ),
        },
        "ignored_as_final_alpha_evidence": [
            "results/parm_taro/training/v2_alpha",
            "results/parm_taro/training/v2_alpha_preference_pilot",
            "results/parm_taro/training/v2_alpha_preference_pilot_v2",
            "results/parm_taro/training/v2_alpha_preference_pilot_v3",
        ],
        "parm_tree": parm_current,
        "constant_control_resolution": {
            "mode": (
                "native"
                if native_required_production_checks
                else "non_identical_audit"
                if correction_authorized
                else "unresolved"
            ),
            "native_production_checks_pass": native_required_production_checks,
            "audit_present": audit is not None,
            "audit_pass": bool(audit and audit.get("status") == "PASS"),
            "correction_authorized": correction_authorized,
            "threshold": 0.001,
            "threshold_changed": False,
        },
        "constant_control_audit": audit,
    }


def render_stage9_final_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 9 Final Gate",
        "",
        "## Status",
        "",
        "```text",
        str(report["status"]),
        "```",
        "",
        "Pilot V3 authorizes the production recipe but cannot satisfy the final gate. "
        "Only the isolated 8,000-train/500-validation production run is accepted.",
        "A NOT_PASS production report may be corrected only by the read-only "
        "non-identical constant-alpha audit with the unchanged 0.001 threshold.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "The failed legacy `v2_alpha` run and all three pilots remain read-only "
            "evidence and are never promoted by this gate.",
            "",
        ]
    )
    return "\n".join(lines)
