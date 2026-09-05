"""Read-only full-validation audit of the Stage 9 constant-alpha control."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.scripts.pilot_alpha_preference_v2 import _endpoint_routes
from PARM_TARO.training.alpha import build_alpha_controls
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.alpha_production import _production_snapshot
from PARM_TARO.training.alpha_residual_router import (
    AlphaResidualSmartRouter,
    load_alpha_residual_checkpoint,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.constant_control import ConstantControlAccumulator
from PARM_TARO.training.data import (
    SequenceTooLongError,
    iter_validation_tasks,
    load_examples,
)
from PARM_TARO.training.engine import (
    _assert_no_frozen_gradients,
    frozen_response_pair,
    route_response_pair,
)
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import load_training_runtime, project_path
from router_v2.cache.io import sha256_file
from router_v2.training.audit import hash_tree


def source_failure_is_constant_only(report: dict[str, Any]) -> bool:
    checks = report.get("checks", {})
    failed = {name for name, value in checks.items() if value is not True}
    return report.get("status") == "NOT_PASS" and failed == {
        "constant_alpha_differs_materially"
    }


def _semantic_config_matches(
    configured: AlphaPreferenceProductionConfig,
    source: AlphaPreferenceProductionConfig,
) -> bool:
    ignored = {"device", "allow_cpu_fallback"}
    left = {key: value for key, value in configured.to_dict().items() if key not in ignored}
    right = {key: value for key, value in source.to_dict().items() if key not in ignored}
    return left == right


def audit_constant_alpha_control(
    config: AlphaPreferenceProductionConfig,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
    progress_every_tasks: int = 50,
) -> dict[str, Any]:
    if bootstrap_samples < 1 or progress_every_tasks < 1:
        raise ValueError("Audit controls must be positive")
    output_dir = project_path(config.output_dir)
    source_status_path = output_dir / "run_status.json"
    source_report = json.loads(source_status_path.read_text(encoding="utf-8"))
    source_config = AlphaPreferenceProductionConfig.from_dict(
        source_report["config"]
    )
    if not _semantic_config_matches(config, source_config):
        raise ValueError("Audit config does not match the production run")
    source_config = replace(
        source_config,
        device=config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    checkpoint_path = Path(source_report["checkpoint"]["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_path(checkpoint_path)
    checkpoint_hash_before = sha256_file(checkpoint_path)
    source_status_hash_before = sha256_file(source_status_path)
    production_tree_before = hash_tree(output_dir)
    if checkpoint_hash_before != source_report["checkpoint"]["sha256"]:
        raise ValueError("Production checkpoint hash does not match run_status")
    training_config = ParmRouterTrainingConfig.load_json(
        project_path(source_config.base_training_config_path)
    )
    training_config = replace(
        training_config,
        device=source_config.device,
        allow_cpu_fallback=source_config.allow_cpu_fallback,
        output_dir=source_config.output_dir,
        max_train_samples=source_config.expected_train_samples,
        max_validation_samples=source_config.expected_validation_samples,
    )
    runtime = load_training_runtime(training_config)
    router, payload = load_alpha_residual_checkpoint(
        checkpoint_path, map_location=runtime.device
    )
    if not isinstance(router, AlphaResidualSmartRouter):
        raise TypeError("Constant-control audit requires the V3 production Router")
    if payload["metadata"].get("production_config") != source_report["config"]:
        raise ValueError("Checkpoint production config does not match run_status")
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    if payload["metadata"].get("tokenizer_semantic_sha256") != tokenizer_hash:
        raise ValueError("Production checkpoint tokenizer hash mismatch")
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    examples = load_examples(
        project_path(training_config.data_root),
        "validation",
        max_samples=source_config.expected_validation_samples,
    )
    if len(examples) != source_config.expected_validation_samples:
        raise ValueError("Audit requires all 500 validation examples")
    tasks = list(iter_validation_tasks(examples, training_config.validation_alpha_grid))
    controls = build_alpha_controls(torch.stack([alpha for _, alpha in tasks]))
    accumulator = ConstantControlAccumulator()
    endpoint_values: list[float] = []
    skipped: list[str] = []
    with torch.no_grad():
        for task_index, (example, alpha_cpu) in enumerate(tasks):
            alpha = alpha_cpu.to(runtime.device)
            constant_alpha = controls.constant[task_index].to(runtime.device)
            try:
                frozen = frozen_response_pair(
                    runtime, training_config, example, alpha
                )
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            correct_routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=alpha,
                static_scale=1.0,
            )
            constant_routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=constant_alpha,
                static_scale=1.0,
            )
            endpoints = _endpoint_routes(router, frozen, runtime.device)
            task_deltas = []
            for response_index in (0, 1):
                task_deltas.append(
                    (
                        correct_routes[response_index].lambda_t
                        - constant_routes[response_index].lambda_t
                    )
                    .abs()
                    .detach()
                    .cpu()
                    .reshape(-1)
                )
                endpoint_values.extend(
                    float(value)
                    for value in (
                        endpoints[0][response_index].lambda_t
                        - endpoints[1][response_index].lambda_t
                    )
                    .abs()
                    .detach()
                    .cpu()
                    .reshape(-1)
                )
            accumulator.update(alpha_cpu, controls.constant[task_index], torch.cat(task_deltas))
            completed = task_index + 1
            if completed % progress_every_tasks == 0:
                print(
                    json.dumps(
                        {
                            "event": "stage9_constant_control_audit_progress",
                            "completed_tasks": completed,
                            "total_tasks": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    constant_metrics = accumulator.finalize(
        threshold=source_config.min_control_lambda_delta,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    endpoint_metrics = summarize(endpoint_values)
    _assert_no_frozen_gradients(runtime)
    router_gradients_none = all(
        parameter.grad is None for parameter in router.parameters()
    )
    protected_current = _production_snapshot(source_config, training_config, runtime)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    source_status_hash_after = sha256_file(source_status_path)
    production_tree_after = hash_tree(output_dir)
    source_all_delta = float(
        source_report["validation_after"][
            "correct_vs_constant_lambda_delta"
        ]["mean"]
    )
    recomputed_all_delta = float(
        constant_metrics["all_tasks"]["token_weighted"]["mean"]
    )
    identical_max = float(
        constant_metrics["identical_tasks_only"]["token_weighted"]["max"]
    )
    per_alpha = constant_metrics["per_alpha_helpfulness"]
    checks = {
        "source_failed_only_constant_control": source_failure_is_constant_only(
            source_report
        ),
        "constant_alpha_is_half_half": constant_metrics["constant_alpha"]
        == [0.5, 0.5],
        "full_validation_tasks_recomputed": (
            constant_metrics["total_tasks"] == 2500 and not skipped
        ),
        "identical_task_count_is_500": (
            constant_metrics["identical_tasks"] == 500
        ),
        "non_identical_task_count_is_2000": (
            constant_metrics["non_identical_tasks"] == 2000
        ),
        "identical_alpha_delta_approximately_zero": identical_max <= 1e-7,
        "midpoint_group_recognized_as_identical": (
            per_alpha["0.50"]["identical_to_constant"] is True
        ),
        "other_alpha_groups_are_informative": all(
            per_alpha[key]["identical_to_constant"] is False
            for key in ("0.00", "0.25", "0.75", "1.00")
        ),
        "all_task_delta_matches_original_report": (
            abs(recomputed_all_delta - source_all_delta) <= 1e-6
        ),
        "non_identical_constant_delta_passes_unchanged_threshold": bool(
            constant_metrics["gate"]["pass"]
        ),
        "all_task_gate_was_diluted_by_identical_controls": (
            source_all_delta < source_config.min_control_lambda_delta
            and bool(constant_metrics["gate"]["pass"])
        ),
        "shuffled_gate_remains_passed": (
            source_report["checks"].get(
                "shuffled_alpha_changes_lambda_materially"
            )
            is True
            and float(
                source_report["validation_after"][
                    "correct_vs_shuffled_lambda_delta"
                ]["mean"]
            )
            >= source_config.min_control_lambda_delta
        ),
        "endpoint_gate_remains_passed": (
            source_report["checks"].get(
                "correct_alpha_changes_lambda_materially"
            )
            is True
            and float(endpoint_metrics["mean"])
            >= source_config.min_control_lambda_delta
        ),
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "source_run_status_unchanged": (
            source_status_hash_before == source_status_hash_after
        ),
        "production_tree_unchanged": production_tree_before == production_tree_after,
        "frozen_artifacts_unchanged": (
            protected_current == source_report["protected_after"]
        ),
        "all_gradients_none": router_gradients_none,
        "test_split_unused": True,
    }
    if not all(math.isfinite(float(value)) for value in endpoint_values):
        raise FloatingPointError("Non-finite endpoint delta in control audit")
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    return {
        "schema_version": 1,
        "stage": 9,
        "audit": "constant_alpha_non_identical_control",
        "status": status,
        "checks": checks,
        "scientific_gate": {
            "old_statistic": (
                "mean constant-control delta over all validation tasks"
            ),
            "corrected_statistic": constant_metrics["gate"]["statistic"],
            "threshold": source_config.min_control_lambda_delta,
            "threshold_changed": False,
            "old_gate_diluted": checks[
                "all_task_gate_was_diluted_by_identical_controls"
            ],
        },
        "constant_control": constant_metrics,
        "endpoint_alpha_delta": endpoint_metrics,
        "source_metrics": {
            "all_task_constant_delta": source_all_delta,
            "shuffled_delta": source_report["validation_after"][
                "correct_vs_shuffled_lambda_delta"
            ],
            "endpoint_delta": source_report["validation_after"][
                "endpoint_alpha_lambda_delta"
            ],
        },
        "source": {
            "run_status_path": str(source_status_path),
            "run_status_sha256": source_status_hash_before,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash_before,
            "production_tree": production_tree_before,
        },
        "runtime": {
            "device": str(runtime.device),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "validation_examples": len(examples),
            "validation_tasks": len(tasks),
            "skipped_sample_ids": sorted(set(skipped)),
            "test_split_used": False,
        },
        "corrected_production_status": (
            "PASS" if status == "PASS" else "NOT_PASS"
        ),
    }


def render_constant_control_audit(report: dict[str, Any]) -> str:
    control = report["constant_control"]
    informative = control["non_identical_tasks_only"]
    lines = [
        "# Stage 9 Constant-Alpha Control Audit",
        "",
        "## Status",
        "",
        "```text",
        str(report["status"]),
        "```",
        "",
        f"Constant alpha: `{control['constant_alpha']}`",
        f"Identical tasks: `{control['identical_tasks']}`",
        f"Non-identical tasks: `{control['non_identical_tasks']}`",
        f"All-task token-weighted delta: "
        f"`{control['all_tasks']['token_weighted']['mean']:.10f}`",
        f"Non-identical token-weighted delta: "
        f"`{informative['token_weighted']['mean']:.10f}`",
        f"Unchanged threshold: `{control['gate']['threshold']}`",
        "",
        "The checkpoint, source run status, production output tree, frozen models, "
        "PBLORA, PARM tree, and Stage 8 data are read-only inputs to this audit.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["checks"].items()
    )
    lines.append("")
    return "\n".join(lines)
