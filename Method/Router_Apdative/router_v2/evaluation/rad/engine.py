"""End-to-end Stage 6 RAD evaluation engine."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

from router_v2.cache.io import (
    require_path_within,
    sha256_file,
    write_json_atomic,
)
from router_v2.checkpoint import load_taro_checkpoint
from router_v2.device import resolve_device
from router_v2.evaluation.rad.config import RADEvaluationConfig
from router_v2.evaluation.rad.data import (
    PromptRecord,
    load_prompt_split,
    project_path,
)
from router_v2.evaluation.rad.decoding import (
    RADV2Decoder,
    RouteSpec,
    test_route_specs,
    validation_route_specs,
)
from router_v2.evaluation.rad.legacy import (
    audit_v1_results,
    import_v1_records,
)
from router_v2.evaluation.rad.metrics import (
    aggregate_metrics,
    attach_ppl_degradation,
    generation_length_diagnostics,
    lambda_diagnostics,
    mean_routing_strength,
    metric_record,
    pareto_points,
    select_by_alignment,
    write_csv_atomic,
)
from router_v2.evaluation.rad.scoring import SentimentScorer
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.guide_model.runtime import (
    configure_deterministic_inference,
    load_compatible_tokenizer_pair,
    load_frozen_causal_model_pair,
)
from router_v2.smart_checkpoint import load_smart_checkpoint
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "router_v2" / "rad"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _protected_snapshot(config: RADEvaluationConfig) -> dict[str, Any]:
    v1_results = project_path(PROJECT_ROOT, config.v1_results_dir)
    return {
        "router_v1": hash_tree(PROJECT_ROOT / "router"),
        "parm": hash_tree(PROJECT_ROOT / "PARM"),
        "method_rad": hash_tree(PROJECT_ROOT / "Method" / "RAD"),
        "rad_benchmark": hash_tree(PROJECT_ROOT / "dataset" / "rad_benchmark"),
        "v1_results": hash_tree(v1_results),
    }


def _compare_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    checks = {f"{name}_unchanged": before[name] == after[name] for name in before}
    return {
        "before": before,
        "after": after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _stage5_checkpoint_audit(config: RADEvaluationConfig) -> dict[str, Any]:
    values = {
        "taro": project_path(PROJECT_ROOT, config.taro_checkpoint_path),
        "state": project_path(PROJECT_ROOT, config.state_checkpoint_path),
        "history": project_path(PROJECT_ROOT, config.history_checkpoint_path),
    }
    stages: dict[str, Any] = {}
    for stage, checkpoint in values.items():
        status_path = checkpoint.parent / "run_status.json"
        if not checkpoint.is_file() or not status_path.is_file():
            raise FileNotFoundError(f"Stage 5 {stage} artifact is absent")
        status = _load_json(status_path)
        stages[stage] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "run_status_path": str(status_path),
            "run_status_sha256": sha256_file(status_path),
            "status": status.get("status"),
            "frozen_artifacts_unchanged": status.get("checks", {}).get(
                "frozen_artifacts_unchanged"
            ),
        }
    checks = {
        "all_stage5_status_pass": all(
            value["status"] == "PASS" for value in stages.values()
        ),
        "all_stage5_frozen_audits_pass": all(
            value["frozen_artifacts_unchanged"] is True
            for value in stages.values()
        ),
    }
    return {"stages": stages, "checks": checks, "pass": all(checks.values())}


def _load_routers(
    config: RADEvaluationConfig,
    device: torch.device,
) -> dict[str, Any]:
    taro, _ = load_taro_checkpoint(
        project_path(PROJECT_ROOT, config.taro_checkpoint_path),
        map_location=device,
    )
    state, _ = load_smart_checkpoint(
        project_path(PROJECT_ROOT, config.state_checkpoint_path),
        map_location=device,
    )
    history, _ = load_smart_checkpoint(
        project_path(PROJECT_ROOT, config.history_checkpoint_path),
        map_location=device,
    )
    if not isinstance(state, SmartTokenRouter) or state.config.variant != (
        "v2_topk_confidence_position"
    ):
        raise ValueError("State checkpoint has the wrong Smart Router variant")
    if not isinstance(history, SmartTokenRouter) or history.config.variant != (
        "v2_topk_state_history"
    ):
        raise ValueError("History checkpoint has the wrong Smart Router variant")
    routers = {"taro": taro, "v2_state": state, "v2_history": history}
    for router in routers.values():
        router.eval()
        for parameter in router.parameters():
            parameter.requires_grad_(False)
    return routers


def _record_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["prompt_id"]), str(row["method"]), int(row["seed"]))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _record_key(row)
            if key in keys:
                raise ValueError(f"Duplicate resumable evaluation record: {key}")
            keys.add(key)
            records.append(row)
    return records


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _raw_v2_record(
    prompt: PromptRecord,
    spec: RouteSpec,
    seed: int,
    generation: Any,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt.prompt_id,
        "source_prompt_id": prompt.source_prompt_id,
        "source_index": prompt.source_index,
        "prompt": prompt.prompt,
        "prompt_sentiment_class": prompt.prompt_sentiment_class,
        "method": spec.label,
        "route_kind": spec.kind,
        "route_value": spec.value,
        "method_family": "v2_autoregressive_sentiment_guide",
        "ppl_model_family": "v2_gpt2_medium_conditional",
        "ppl_protocol": "continuation_conditional_under_frozen_v2_base",
        "seed": int(seed),
        **generation.to_dict(),
        "perplexity": generation.base_conditional_perplexity,
        "failure_status": "success",
        "source": "stage6_v2_generation",
    }


def _run_generation_jobs(
    *,
    decoder: RADV2Decoder,
    prompts: Sequence[PromptRecord],
    seeds: Sequence[int],
    specs: Sequence[RouteSpec],
    output_path: Path,
    log_every_jobs: int,
) -> list[dict[str, Any]]:
    existing = _read_jsonl(output_path)
    completed = {_record_key(row) for row in existing}
    expected = {
        (prompt.prompt_id, spec.label, int(seed))
        for prompt in prompts
        for seed in seeds
        for spec in specs
    }
    unknown = completed - expected
    if unknown:
        raise ValueError(f"Resume output contains unplanned records: {sorted(unknown)[:3]}")
    completed_count = len(completed)
    total = len(expected)
    start = time.perf_counter()
    for prompt in prompts:
        for seed in seeds:
            for spec in specs:
                key = (prompt.prompt_id, spec.label, int(seed))
                if key in completed:
                    continue
                generation = decoder.generate(prompt.prompt, spec, seed=int(seed))
                row = _raw_v2_record(prompt, spec, int(seed), generation)
                _append_jsonl(output_path, row)
                completed.add(key)
                completed_count += 1
                if completed_count % log_every_jobs == 0 or completed_count == total:
                    elapsed = time.perf_counter() - start
                    print(
                        json.dumps(
                            {
                                "event": "stage6_progress",
                                "completed": completed_count,
                                "total": total,
                                "method": spec.label,
                                "prompt_class": prompt.prompt_sentiment_class,
                                "elapsed_seconds_this_session": elapsed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    records = _read_jsonl(output_path)
    if {_record_key(row) for row in records} != expected:
        raise RuntimeError("Evaluation job materialization is incomplete")
    return records


def _score_v2_records(
    records: Sequence[dict[str, Any]],
    scorer: SentimentScorer,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        scores = scorer.score([str(row["generated_text"]) for row in batch])
        outputs.extend({**row, **score} for row, score in zip(batch, scores))
    return outputs


def _compact_metric_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "prompt_id",
        "prompt_sentiment_class",
        "method",
        "legacy_method",
        "method_family",
        "ppl_model_family",
        "ppl_protocol",
        "seed",
        "alignment_success",
        "target_probability",
        "perplexity",
        "ppl_degradation",
        "coherence",
        "repeated_unigram_rate",
        "repeated_bigram_rate",
        "repeated_trigram_rate",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "generation_length",
        "eos_generated",
        "terminated_on_eos",
        "source_eos_generated",
        "eos_metadata_mismatch",
        "latency_per_token",
        "total_latency",
        "failure_status",
        "source",
    )
    return [{key: row.get(key) for key in keys} for row in records]


def _method_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["method"]) for row in records).items()))


def _markdown_report(report: dict[str, Any]) -> str:
    selection = report["validation_selection"]
    checks = report["checks"]
    return (
        "# Stage 6 RAD V2 Validation Report\n\n"
        f"Status: **{report['status']}**\n\n"
        f"- Best fixed lambda selected on validation: `{selection['best_fixed_lambda']}`\n"
        f"- Best heuristic selected on validation: `{selection['best_heuristic']}`\n"
        f"- TARO validation mean lambda: `{selection['taro_same_average_lambda']}`\n"
        f"- V2 history validation mean lambda: `{selection['v2_same_average_lambda']}`\n"
        f"- Combined held-out records: `{report['record_counts']['combined']}`\n"
        f"- Protected artifacts unchanged: `{checks['protected_artifacts_unchanged']}`\n"
        f"- Router collapse absent: `{checks['adaptive_lambdas_nonconstant']}`\n\n"
        "Primary Pareto uses sentiment alignment versus PPL degradation relative "
        "to the base method from the same model family. Raw V1 and V2 PPL values "
        "must not be compared as if they used the same backbone/protocol.\n"
    )


def run_rad_evaluation(
    config: RADEvaluationConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = require_path_within(
        project_path(PROJECT_ROOT, config.output_dir),
        OUTPUT_ROOT,
        label="Stage 6 output directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    if resume:
        if not config_path.exists():
            raise FileNotFoundError("Resume requires resolved_config.json")
        if _load_json(config_path) != config.to_dict():
            raise ValueError("Resume config does not match existing Stage 6 run")
    elif any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 6 output directory is not empty: {output_dir}")
    else:
        write_json_atomic(config_path, config.to_dict())

    protected_before = _protected_snapshot(config)
    write_json_atomic(
        output_dir / "protected_audit_before.json",
        protected_before,
        overwrite=(output_dir / "protected_audit_before.json").exists(),
    )
    stage5_audit = _stage5_checkpoint_audit(config)
    if not stage5_audit["pass"]:
        raise ValueError("Stage 5 checkpoints are not approved for Stage 6")
    classifier_path = project_path(PROJECT_ROOT, config.sentiment_classifier_path)
    v1_audit = audit_v1_results(
        project_path(PROJECT_ROOT, config.v1_results_dir),
        project_root=PROJECT_ROOT,
        test_prompt_offset_per_class=config.test_prompt_offset_per_class,
        test_prompt_limit_per_class=config.test_prompt_limit_per_class,
        max_new_tokens=config.max_new_tokens,
        sentiment_classifier_path=classifier_path,
        seeds=config.seeds,
    )
    if not v1_audit["pass"]:
        raise ValueError("V1 baseline provenance audit failed")
    write_json_atomic(
        output_dir / "v1_baseline_audit.json",
        v1_audit,
        overwrite=(output_dir / "v1_baseline_audit.json").exists(),
    )
    write_json_atomic(
        output_dir / "stage5_checkpoint_audit.json",
        stage5_audit,
        overwrite=(output_dir / "stage5_checkpoint_audit.json").exists(),
    )

    validation_prompts = load_prompt_split(
        config, "validation", project_root=PROJECT_ROOT
    )
    test_prompts = load_prompt_split(config, "test", project_root=PROJECT_ROOT)
    device = resolve_device(
        config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    classifier_device = resolve_device(
        config.sentiment_classifier_device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    runtime = configure_deterministic_inference(seed=config.seeds[0], device=device)
    base_path = project_path(PROJECT_ROOT, config.base_model_path)
    guide_path = project_path(PROJECT_ROOT, config.guide_model_path)
    tokenizer_pair = load_compatible_tokenizer_pair(base_path, guide_path)
    model_pair = load_frozen_causal_model_pair(
        base_path,
        guide_path,
        device=device,
        precision=config.precision,
    )
    routers = _load_routers(config, device)
    decoder = RADV2Decoder(
        base_model=model_pair.base,
        guide_model=model_pair.guide,
        tokenizer=tokenizer_pair.base,
        routers=routers,
        device=device,
        top_k=config.top_k,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        stop_on_eos=config.stop_on_eos,
    )
    scorer = SentimentScorer(classifier_path, device=classifier_device)
    provenance = {
        "runtime": runtime,
        "device": str(device),
        "classifier_device": str(classifier_device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available_in_process": torch.cuda.is_available(),
        "base_model": checkpoint_descriptor(base_path),
        "guide_model": checkpoint_descriptor(guide_path),
        "sentiment_classifier": checkpoint_descriptor(classifier_path),
        "stage5": stage5_audit,
        "v1": v1_audit,
    }
    write_json_atomic(
        output_dir / "provenance.json",
        provenance,
        overwrite=(output_dir / "provenance.json").exists(),
    )

    validation_specs = validation_route_specs(
        config.fixed_lambda_sweep,
        config.heuristic_methods,
    )
    validation_raw = _run_generation_jobs(
        decoder=decoder,
        prompts=validation_prompts,
        seeds=config.seeds,
        specs=validation_specs,
        output_path=output_dir / "validation_outputs.jsonl",
        log_every_jobs=config.log_every_jobs,
    )
    validation_scored = _score_v2_records(
        validation_raw,
        scorer,
        batch_size=config.sentiment_classifier_batch_size,
    )
    validation_metrics = [metric_record(row) for row in validation_scored]
    fixed_labels = [spec.label for spec in validation_specs if spec.kind == "fixed"]
    best_fixed_label = select_by_alignment(validation_metrics, fixed_labels)
    best_fixed_spec = next(
        spec for spec in validation_specs if spec.label == best_fixed_label
    )
    if best_fixed_spec.value is None:
        raise RuntimeError("Selected fixed route has no value")
    best_heuristic = select_by_alignment(
        validation_metrics,
        config.heuristic_methods,
    )
    taro_mean = mean_routing_strength(validation_metrics, "taro")
    v2_mean = mean_routing_strength(validation_metrics, "v2_history")
    selection = {
        "selection_split": "validation",
        "negative_class_primary": True,
        "best_fixed_method": best_fixed_label,
        "best_fixed_lambda": best_fixed_spec.value,
        "best_heuristic": best_heuristic,
        "same_average_source": "all validation prompt classes and seeds",
        "taro_same_average_lambda": taro_mean,
        "v2_same_average_lambda": v2_mean,
        "validation_records": len(validation_metrics),
    }
    write_json_atomic(
        output_dir / "validation_selection.json",
        selection,
        overwrite=(output_dir / "validation_selection.json").exists(),
    )

    test_specs = test_route_specs(
        best_fixed_lambda=best_fixed_spec.value,
        best_heuristic=best_heuristic,
        taro_same_average=taro_mean,
        v2_same_average=v2_mean,
    )
    test_raw = _run_generation_jobs(
        decoder=decoder,
        prompts=test_prompts,
        seeds=config.seeds,
        specs=test_specs,
        output_path=output_dir / "per_sample_outputs.jsonl",
        log_every_jobs=config.log_every_jobs,
    )
    test_scored = _score_v2_records(
        test_raw,
        scorer,
        batch_size=config.sentiment_classifier_batch_size,
    )
    v2_metrics = [metric_record(row) for row in test_scored]
    v1_records = import_v1_records(
        v1_audit["outputs_path"],
        prompts=test_prompts,
        seeds=config.seeds,
        eos_token_id=int(v1_audit["base_model_eos_token_id"]),
    )
    v1_metrics = [metric_record(row) for row in v1_records]
    combined = attach_ppl_degradation(
        [*v1_metrics, *v2_metrics],
        family_base_methods={
            "v1_rad_scalar_reward": "v1_base",
            "v2_autoregressive_sentiment_guide": "v2_base",
        },
    )
    aggregates = aggregate_metrics(
        combined,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_samples=config.bootstrap_samples,
    )
    pareto = pareto_points(aggregates)
    lambda_rows = lambda_diagnostics(combined)
    write_csv_atomic(output_dir / "per_sample_metrics.csv", _compact_metric_rows(combined))
    write_csv_atomic(output_dir / "aggregate_metrics.csv", aggregates)
    write_csv_atomic(output_dir / "pareto_points.csv", pareto)
    write_csv_atomic(output_dir / "lambda_diagnostics.csv", lambda_rows)

    expected_per_method = len(test_prompts) * len(config.seeds)
    counts = _method_counts(combined)
    required_methods = {
        "v1_base",
        "v1_fixed",
        "v1_heuristic",
        "v1_router",
        "v2_base",
        "v2_fixed",
        "v2_heuristic",
        "taro",
        "v2_state",
        "v2_history",
        "taro_same_average",
        "v2_same_average",
    }
    all_finite = all(
        math.isfinite(float(row[metric]))
        for row in combined
        for metric in (
            "alignment_success",
            "target_probability",
            "perplexity",
            "ppl_degradation",
            "coherence",
            "latency_per_token",
        )
    )
    adaptive_summary = {
        str(row["method"]): row
        for row in lambda_rows
        if row["position"] == "all"
    }
    adaptive_nonconstant = all(
        float(adaptive_summary[method]["std"]) > 1e-6
        for method in ("v1_router", "taro", "v2_state", "v2_history")
    )
    length_diagnostics = generation_length_diagnostics(
        combined,
        max_new_tokens=config.max_new_tokens,
    )
    protected_after = _protected_snapshot(config)
    protected_audit = _compare_snapshots(protected_before, protected_after)
    write_json_atomic(
        output_dir / "protected_audit_after.json",
        protected_audit,
        overwrite=(output_dir / "protected_audit_after.json").exists(),
    )
    checks = {
        "stage5_checkpoints_pass": stage5_audit["pass"],
        "v1_provenance_pass": v1_audit["pass"],
        "required_methods_complete": (
            set(counts) == required_methods
            and all(count == expected_per_method for count in counts.values())
        ),
        "no_failed_generations": all(
            row["failure_status"] == "success" for row in combined
        ),
        "generation_budget_at_least_32": length_diagnostics["checks"][
            "max_new_tokens_at_least_required_budget"
        ],
        "generation_lengths_and_termination_valid": length_diagnostics["pass"],
        "all_primary_metrics_finite": all_finite,
        "same_average_controls_present": {
            "taro_same_average",
            "v2_same_average",
        } <= set(counts),
        "adaptive_lambdas_nonconstant": adaptive_nonconstant,
        "protected_artifacts_unchanged": protected_audit["pass"],
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    readiness = {
        "stable": checks["all_primary_metrics_finite"],
        "no_lambda_collapse": checks["adaptive_lambdas_nonconstant"],
        "ready_for_parm_review": (
            checks["all_primary_metrics_finite"]
            and checks["adaptive_lambdas_nonconstant"]
            and checks["protected_artifacts_unchanged"]
        ),
    }
    report = {
        "schema_version": 1,
        "status": status,
        "validation_selection": selection,
        "record_counts": {
            "validation_v2": len(validation_metrics),
            "test_v2": len(v2_metrics),
            "test_v1_imported": len(v1_metrics),
            "combined": len(combined),
            "by_method": counts,
        },
        "checks": checks,
        "generation_length_diagnostics": length_diagnostics,
        "parm_readiness": readiness,
        "scientific_comparability": {
            "alignment_evaluator_shared": True,
            "prompt_ids_and_seeds_shared": True,
            "raw_ppl_cross_family_comparable": False,
            "primary_pareto_x": "family-relative PPL degradation",
            "v1_family": "GPT-2 Large + scalar RAD reward",
            "v2_family": "GPT-2 Medium + autoregressive sentiment guide",
        },
        "pareto_points": pareto,
        "stage5_checkpoint_audit": stage5_audit,
        "v1_baseline_audit": v1_audit,
        "protected_audit": protected_audit,
        "output_hashes": {
            name: sha256_file(output_dir / name)
            for name in (
                "validation_outputs.jsonl",
                "per_sample_outputs.jsonl",
                "per_sample_metrics.csv",
                "aggregate_metrics.csv",
                "pareto_points.csv",
                "lambda_diagnostics.csv",
            )
        },
    }
    write_json_atomic(
        output_dir / "stage6_report.json",
        report,
        overwrite=(output_dir / "stage6_report.json").exists(),
    )
    report_path = output_dir / "stage6_report.md"
    report_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def with_runtime_overrides(
    config: RADEvaluationConfig,
    *,
    device: str | None = None,
    output_dir: str | None = None,
) -> RADEvaluationConfig:
    values: dict[str, Any] = {}
    if device is not None:
        values["device"] = device
    if output_dir is not None:
        values["output_dir"] = output_dir
    return replace(config, **values) if values else config
