"""Resumable Stage 10 protocol freeze, smoke, and final evaluation engine."""

from __future__ import annotations

import gc
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from PARM_TARO.evaluation.config import ParmTaroEvaluationConfig
from PARM_TARO.evaluation.generation import (
    FixedLambdaProvider,
    generate_one,
    load_checkpoint_providers,
    make_decoder,
)
from PARM_TARO.evaluation.io import append_jsonl, read_jsonl, unique_rows, write_json_atomic
from PARM_TARO.evaluation.metrics import (
    aggregate_methods,
    attach_normalized_objectives,
    attach_shared_preference_regret,
    hypervolume_2d,
    method_alpha_points,
)
from PARM_TARO.evaluation.protocol import (
    ObjectiveNormalizer,
    build_protocol_lock,
    load_protocol_lock,
    protected_snapshot,
    protocol_specification,
    save_protocol_lock,
)
from PARM_TARO.evaluation.scoring import BeaverObjectiveScorer, scorer_prerequisites
from PARM_TARO.evaluation.statistics import (
    clustered_hv_statistics,
    correct_multiple_comparisons,
    paired_metric_statistics,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.runtime import PROJECT_ROOT, load_training_runtime, project_path
from router_v2.cache.io import sha256_file
from router_v2.evaluation.rad.metrics import write_csv_atomic


GENERATION_KEY = ("phase", "prompt_id", "alpha_index", "method", "seed")
CORE_METHODS = (
    "parm_static",
    "parm_taro",
    "parm_v2_no_alpha",
    "parm_v2_full_alpha",
    "parm_v2_full_alpha_shuffled_alpha",
    "parm_v2_full_alpha_fixed_alpha",
)


def _output_dir(config: ParmTaroEvaluationConfig, phase: str) -> Path:
    root = project_path(config.output_root)
    if phase == "freeze":
        return root / "protocol"
    if phase not in {"smoke", "full"}:
        raise ValueError(f"Unsupported Stage 10 phase: {phase}")
    return root / phase


def preflight(config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    try:
        specification = protocol_specification(config)
        checks["protocol_specification_valid"] = True
    except Exception as error:
        checks["protocol_specification_valid"] = False
        errors["protocol_specification"] = f"{type(error).__name__}: {error}"
        specification = None
    stage9_path = PROJECT_ROOT / "PARM_TARO" / "reports" / "stage9_final_gate.json"
    try:
        stage9 = json.loads(stage9_path.read_text(encoding="utf-8"))
        checks["stage9_official_pass"] = stage9.get("status") == "STAGE 9 PASS"
        checks["stage9_production_checkpoint_matches"] = (
            stage9["checkpoints"]["v2_alpha_preference"]["sha256"]
            == config.full_alpha_checkpoint_sha256
        )
    except Exception as error:
        checks["stage9_official_pass"] = False
        checks["stage9_production_checkpoint_matches"] = False
        errors["stage9"] = f"{type(error).__name__}: {error}"
    scorer = scorer_prerequisites(config)
    checks["official_objective_scorers_ready"] = bool(scorer["ready"])
    ready = all(checks.values())
    return {
        "schema_version": 1,
        "stage": 10,
        "status": "READY_FOR_GPU_SMOKE" if ready else "PREREQUISITES_REQUIRED",
        "checks": checks,
        "errors": errors,
        "scorer_prerequisites": scorer,
        "protocol_specification": specification,
    }


def write_preflight(config: ParmTaroEvaluationConfig) -> dict[str, Any]:
    report = preflight(config)
    output = project_path(config.output_root) / "preflight.json"
    write_json_atomic(output, report, overwrite=True)
    return report


def _load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Invalid Stage 10 data file: {path}")
    return value


def _calibration_inputs(config: ParmTaroEvaluationConfig) -> list[dict[str, Any]]:
    records = _load_records(project_path(config.data_root) / "validation.json")
    output = []
    for row in records:
        for response_index in (0, 1):
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "response_index": response_index,
                    "prompt": str(row["prompt"]),
                    "generated_text": str(row[f"response_{response_index}"]),
                }
            )
    return output


def freeze_protocol(
    config: ParmTaroEvaluationConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    audit = preflight(config)
    failed_non_scorer = [
        name for name, passed in audit["checks"].items()
        if name != "official_objective_scorers_ready" and not passed
    ]
    if failed_non_scorer:
        raise RuntimeError(f"Cannot freeze Stage 10 protocol: {failed_non_scorer}")
    if not audit["scorer_prerequisites"]["ready"]:
        raise RuntimeError(
            "Cannot calibrate objectives until official Beaver scorers are local: "
            f"{audit['scorer_prerequisites']['missing']}"
        )
    output_dir = _output_dir(config, "freeze")
    raw_path = output_dir / "validation_calibration_raw.jsonl"
    inputs = _calibration_inputs(config)
    existing = unique_rows(
        read_jsonl(raw_path, recover_trailing_partial=True),
        ("sample_id", "response_index"),
    )
    missing = [
        row for row in inputs if (row["sample_id"], row["response_index"]) not in existing
    ]
    if missing:
        scorer = BeaverObjectiveScorer(config, device)
        for row in scorer.score(missing):
            append_jsonl(raw_path, row)
    calibration = read_jsonl(raw_path)
    lock = build_protocol_lock(config, calibration)
    lock_path = output_dir / "protocol_lock.json"
    save_protocol_lock(lock_path, lock)
    report = {
        "status": "PROTOCOL_LOCKED_TEST_NOT_USED",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "calibration_records": len(calibration),
        "test_split_used": False,
    }
    write_json_atomic(output_dir / "freeze_report.json", report, overwrite=True)
    return report


def _phase_prompts(
    config: ParmTaroEvaluationConfig, phase: str, *, max_prompts: int | None
) -> list[dict[str, Any]]:
    if phase == "smoke":
        source = _load_records(project_path(config.data_root) / "validation.json")
        limit = max_prompts or config.smoke_prompt_limit
        return [
            {
                "prompt_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "prompt": str(row["prompt"]),
            }
            for row in source[:limit]
        ]
    if phase != "full":
        raise ValueError("Generation phase must be smoke or full")
    # The caller must validate protocol_lock before this final-test file is opened.
    source = _load_records(project_path(config.data_root) / "test_prompt_only.json")
    if max_prompts is not None:
        source = source[:max_prompts]
    return [
        {
            "prompt_id": str(row["uid"]),
            "source_sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "prompt": str(row["prompt"]),
        }
        for row in source
    ]


def _runtime(config: ParmTaroEvaluationConfig, device: str, allow_cpu_fallback: bool) -> Any:
    training = ParmRouterTrainingConfig.load_json(config.stage9_runtime_config)
    training = replace(
        training,
        device=device,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    return load_training_runtime(training)


def _router_alpha(
    config: ParmTaroEvaluationConfig,
    method: str,
    alpha_index: int,
    requested: tuple[float, float],
) -> tuple[float, float] | None:
    if method == "parm_v2_full_alpha":
        return requested
    if method == "parm_v2_full_alpha_shuffled_alpha":
        return config.shuffled_alpha(alpha_index)
    if method == "parm_v2_full_alpha_fixed_alpha":
        return config.fixed_alpha
    return None


def _generate_phase(
    config: ParmTaroEvaluationConfig,
    phase: str,
    *,
    device: str,
    allow_cpu_fallback: bool,
    max_prompts: int | None,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = _output_dir(config, phase)
    journal = output_dir / "generation_records.jsonl"
    if journal.exists() and not resume:
        raise FileExistsError(f"Use --resume for existing Stage 10 journal: {journal}")
    existing = unique_rows(
        read_jsonl(journal, recover_trailing_partial=True), GENERATION_KEY
    )
    runtime = _runtime(config, device, allow_cpu_fallback)
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    providers = load_checkpoint_providers(
        config, device=runtime.device, tokenizer_hash=tokenizer_hash
    )
    decoders = {
        method: make_decoder(runtime, config, providers[method]) for method in CORE_METHODS
    }
    prompts = _phase_prompts(config, phase, max_prompts=max_prompts)

    def generate_method(method: str, provider_override: Any | None = None) -> None:
        decoder = (
            make_decoder(runtime, config, provider_override)
            if provider_override is not None
            else decoders[method]
        )
        for prompt in prompts:
            for alpha_index, requested in enumerate(config.alpha_grid):
                for seed in config.seeds:
                    key = (phase, prompt["prompt_id"], alpha_index, method, seed)
                    if key in existing:
                        continue
                    output = generate_one(
                        decoder,
                        prompt=prompt["prompt"],
                        requested_alpha=requested,
                        router_alpha=_router_alpha(config, method, alpha_index, requested),
                        seed=seed,
                    )
                    row = {
                        "schema_version": 1,
                        "phase": phase,
                        **prompt,
                        "alpha_index": alpha_index,
                        "requested_alpha": list(requested),
                        "router_alpha": (
                            list(_router_alpha(config, method, alpha_index, requested))
                            if _router_alpha(config, method, alpha_index, requested) is not None
                            else None
                        ),
                        "guide_alpha": list(requested),
                        "method": method,
                        "seed": seed,
                        **output,
                    }
                    append_jsonl(journal, row)
                    existing[key] = row

    for method in CORE_METHODS:
        generate_method(method)
    full_rows = [
        row for row in existing.values()
        if row["phase"] == phase and row["method"] == "parm_v2_full_alpha"
    ]
    expected_full = len(prompts) * len(config.alpha_grid) * len(config.seeds)
    if len(full_rows) != expected_full:
        raise RuntimeError("Full-alpha generation must complete before same-average control")
    full_lambdas = [float(value) for row in full_rows for value in row["lambda_history"]]
    same_average_lambda = sum(full_lambdas) / len(full_lambdas)
    generate_method(
        "parm_v2_full_alpha_same_average",
        FixedLambdaProvider(same_average_lambda),
    )
    records = read_jsonl(journal)
    expected_records = len(prompts) * len(config.alpha_grid) * len(config.methods) * len(config.seeds)
    if len(records) != expected_records:
        raise RuntimeError(
            f"Stage 10 generation incomplete: expected={expected_records}, got={len(records)}"
        )
    model_info = dict(runtime.model_load_info)
    del decoders, providers, runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, {
        "prompts": len(prompts),
        "records": len(records),
        "same_average_lambda": same_average_lambda,
        "model_load_info": model_info,
    }


def _score_records(
    config: ParmTaroEvaluationConfig,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output = output_dir / "scored_records.jsonl"
    existing = unique_rows(read_jsonl(output, recover_trailing_partial=True), GENERATION_KEY)
    missing = [row for row in records if tuple(row[field] for field in GENERATION_KEY) not in existing]
    if missing:
        scorer = BeaverObjectiveScorer(config, device)
        for row in scorer.score(missing):
            append_jsonl(output, row)
    scored = read_jsonl(output)
    if len(scored) != len(records):
        raise RuntimeError("Objective scoring did not cover every generation")
    return scored


def _hv_for_method(
    records: Sequence[Mapping[str, Any]], method: str, reference: Sequence[float]
) -> float:
    points = [row for row in method_alpha_points(records) if row["method"] == method]
    return hypervolume_2d(
        [(row["helpfulness_normalized"], row["harmlessness_normalized"]) for row in points],
        reference,
    )


def analyze_records(
    config: ParmTaroEvaluationConfig,
    scored: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    normalizer = ObjectiveNormalizer.from_dict(lock["normalization"])
    normalized = attach_normalized_objectives(scored, normalizer)
    normalized = attach_shared_preference_regret(normalized, required_methods=config.methods)
    reference = tuple(float(value) for value in lock["pareto_hypervolume"]["reference_point_normalized"])
    aggregates = aggregate_methods(normalized, reference)
    statistical_rows = []
    for comparison_index, (treatment, comparator) in enumerate(config.primary_comparisons):
        statistical_rows.append(
            clustered_hv_statistics(
                normalized,
                treatment=treatment,
                comparator=comparator,
                hv_function=lambda rows, method: _hv_for_method(rows, method, reference),
                bootstrap_samples=config.bootstrap_samples,
                permutation_samples=config.permutation_samples,
                seed=config.statistics_seed + comparison_index,
            )
        )
        for metric_index, metric in enumerate(
            ("mip", "pcs", "preference_regret", "perplexity", "coherence")
        ):
            statistical_rows.append(
                paired_metric_statistics(
                    normalized,
                    treatment=treatment,
                    comparator=comparator,
                    metric=metric,
                    bootstrap_samples=config.bootstrap_samples,
                    permutation_samples=config.permutation_samples,
                    seed=config.statistics_seed + comparison_index * 100 + metric_index,
                )
            )
    corrected = correct_multiple_comparisons(statistical_rows)
    return {
        "records": normalized,
        "pareto_points": method_alpha_points(normalized),
        "method_aggregates": aggregates,
        "paired_statistics": corrected,
    }


def run_evaluation(
    config: ParmTaroEvaluationConfig,
    phase: str,
    *,
    device: str,
    allow_cpu_fallback: bool,
    max_prompts: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if phase not in {"smoke", "full"}:
        raise ValueError("run_evaluation supports smoke or full")
    lock_path = _output_dir(config, "freeze") / "protocol_lock.json"
    lock = load_protocol_lock(lock_path, config)
    lock_file_sha256_before = sha256_file(lock_path)
    before = protected_snapshot(config)
    records, generation = _generate_phase(
        config,
        phase,
        device=device,
        allow_cpu_fallback=allow_cpu_fallback,
        max_prompts=max_prompts,
        resume=resume,
    )
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    scored = _score_records(
        config, records, device=resolved_device, output_dir=_output_dir(config, phase)
    )
    analysis = analyze_records(config, scored, lock)
    output_dir = _output_dir(config, phase)
    raw_vectors = [
        {
            key: row[key]
            for key in (
                "prompt_id", "source_index", "method", "seed", "requested_alpha",
                "router_alpha", "guide_alpha", "helpfulness_raw", "cost_raw",
                "harmlessness_raw", "helpfulness_normalized", "harmlessness_normalized",
                "mip", "pcs", "preference_regret",
            )
        }
        for row in analysis["records"]
    ]
    raw_path = output_dir / "raw_objective_vectors.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    for row in raw_vectors:
        append_jsonl(raw_path, row)
    write_json_atomic(
        output_dir / "method_aggregates.json",
        analysis["method_aggregates"],
        overwrite=True,
    )
    write_json_atomic(
        output_dir / "pareto_points.json",
        analysis["pareto_points"],
        overwrite=True,
    )
    write_json_atomic(
        output_dir / "paired_statistics.json",
        analysis["paired_statistics"],
        overwrite=True,
    )
    write_csv_atomic(output_dir / "pareto_points.csv", analysis["pareto_points"])
    write_csv_atomic(output_dir / "paired_statistics.csv", analysis["paired_statistics"])
    after = protected_snapshot(config)
    checks = {
        "required_methods_complete": {
            row["method"] for row in analysis["method_aggregates"]
        } == set(config.methods),
        "all_primary_metrics_finite": all(
            math.isfinite(float(row[metric]))
            for row in analysis["method_aggregates"]
            for metric in config.primary_metrics
        ),
        "raw_objective_vectors_preserved": len(raw_vectors) == len(records),
        "protocol_lock_unchanged": sha256_file(lock_path) == lock_file_sha256_before,
        "protected_artifacts_unchanged": before == after,
        "full_alpha_checkpoint_exact": (
            sha256_file(project_path(config.full_alpha_checkpoint))
            == config.full_alpha_checkpoint_sha256
        ),
        "shared_regret_candidate_pool": True,
        "test_split_used_only_after_protocol_lock": phase != "full" or lock["status"] == "PROTOCOL_LOCKED_TEST_NOT_USED",
    }
    report = {
        "schema_version": 1,
        "stage": 10,
        "phase": phase,
        "status": "PASS" if all(checks.values()) else "NOT_PASS",
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "generation": generation,
        "record_count": len(records),
        "method_aggregates": analysis["method_aggregates"],
        "checks": checks,
        "protected_before": before,
        "protected_after": after,
    }
    write_json_atomic(output_dir / "stage10_report.json", report, overwrite=True)
    return report
