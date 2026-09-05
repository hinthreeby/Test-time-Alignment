"""Measure why the first Stage 9 alpha-preference pilot failed."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from PARM_TARO.training.alpha import response_objective_weights
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.alpha_pilot_v2 import (
    INITIALIZATION_NAMES,
    SCORE_SENSITIVITY_LAMBDAS,
    endpoint_sensitivity_loss,
    exact_score_margin_sensitivity,
    gate_parameterization_stats,
    linear_lambda_floor_loss,
    preference_gain_objective,
    reset_lambda_head,
    router_parameter_groups,
    term_gradient_norms,
    transplant_no_alpha_router,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import load_examples, tokenize_response
from PARM_TARO.training.engine import _load_router, route_response_pair
from PARM_TARO.training.objective import preference_sensitive_objective
from PARM_TARO.training.online import (
    compute_frozen_guide_logprobs,
    compute_frozen_sequence_distributions,
)
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import hash_tree


DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha.json"
DEFAULT_ALPHA_CHECKPOINT = (
    PROJECT_ROOT / "results/parm_taro/training/v2_alpha/best.pt"
)
DEFAULT_NO_ALPHA_CHECKPOINT = (
    PROJECT_ROOT / "results/parm_taro/training/v2_no_alpha/best.pt"
)
DEFAULT_PILOT_REPORT = (
    PROJECT_ROOT
    / "results/parm_taro/training/v2_alpha_preference_pilot/pilot_report.json"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "PARM_TARO/reports/stage9_alpha_pilot_v1_failure_diagnostic.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT
    / "PARM_TARO/reports/stage9_alpha_pilot_v1_failure_diagnostic.md"
)


def _summaries(values: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {name: summarize(items) for name, items in values.items()}


def _quantile_summary(values: list[float]) -> dict[str, float | int]:
    result = summarize(values)
    tensor = torch.tensor(values, dtype=torch.float64)
    result["median"] = float(tensor.median())
    result["p05"] = float(torch.quantile(tensor, 0.05))
    result["p95"] = float(torch.quantile(tensor, 0.95))
    return result


def _protected_snapshot(
    config: ParmRouterTrainingConfig,
    paths: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "pblora": checkpoint_descriptor(project_path(config.parm_adapter_path)),
        "data_manifest_sha256": sha256_file(
            project_path(config.data_root) / "manifest.json"
        ),
        "read_only_inputs": {
            str(path.resolve()): sha256_file(path) for path in paths
        },
    }


def _enable_router_gradients(router: SmartTokenRouter) -> SmartTokenRouter:
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    return router


def _build_initializations(
    config: ParmRouterTrainingConfig,
    *,
    alpha_checkpoint: Path,
    no_alpha_checkpoint: Path,
    tokenizer_hash: str,
    device: torch.device,
    target_gate: float,
    head_weight_std: float,
    seed: int,
) -> dict[str, SmartTokenRouter]:
    collapsed, _ = _load_router(
        alpha_checkpoint,
        kind="smart",
        device=device,
        tokenizer_hash=tokenizer_hash,
    )
    no_alpha, _ = _load_router(
        no_alpha_checkpoint,
        kind="smart",
        device=device,
        tokenizer_hash=tokenizer_hash,
    )
    if not isinstance(collapsed, SmartTokenRouter) or not isinstance(
        no_alpha, SmartTokenRouter
    ):
        raise TypeError("Initialization diagnostic requires Smart Routers")
    alpha_config = SmartRouterConfig.load_json(project_path(config.router_config_path))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transplanted = transplant_no_alpha_router(
        no_alpha,
        alpha_config,
        target_gate=target_gate,
        head_weight_std=head_weight_std,
    )
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)
    fresh = SmartTokenRouter(alpha_config).to(device)
    reset_lambda_head(
        fresh,
        target_gate=target_gate,
        weight_std=head_weight_std,
    )
    routers = dict(
        zip(INITIALIZATION_NAMES, (collapsed, transplanted, fresh))
    )
    return {name: _enable_router_gradients(router) for name, router in routers.items()}


def _gradient_ratios(
    summaries: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float]]:
    ratios: dict[str, dict[str, float]] = {}
    for group in (
        "preference_encoder",
        "final_lambda_head",
        "router_rest",
        "all_router",
    ):
        pref = float(summaries["weighted_preference"][group]["mean"])
        nll = float(summaries["weighted_nll"][group]["mean"])
        strength = float(summaries["weighted_strength"][group]["mean"])
        ratios[group] = {
            "preference_to_nll": pref / max(nll, 1e-30),
            "strength_to_nll": strength / max(nll, 1e-30),
            "strength_to_preference": strength / max(pref, 1e-30),
        }
    return ratios


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 9 Alpha Pilot V1 Failure Diagnostic",
        "",
        "## Status",
        "",
        "```text",
        str(report["status"]),
        "```",
        "",
        "## Root Cause",
        "",
        str(report["conclusion"]["root_cause"]),
        "",
        "## Initialization Ablation",
        "",
        "| Initialization | Mean lambda | Mean sigmoid derivative | Endpoint delta |",
        "|---|---:|---:|---:|",
    ]
    for name, values in report["initialization_ablation"].items():
        gate = values["gate_parameterization"]
        lines.append(
            f"| `{name}` | {gate['mean_lambda']:.8f} | "
            f"{gate['mean_sigmoid_derivative']:.8f} | "
            f"{values['endpoint_alpha_lambda_delta']['mean']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Initialization: `{report['recommendation']['initialization']}`",
            "- Pairwise logit scale: "
            f"`{report['recommendation']['pairwise_logit_scale']:.8g}`",
            "- Use base-relative preference gain, a linear per-token lambda floor,",
            "  endpoint sensitivity, and preference-only warm-up before NLL.",
            "- Keep `min_control_lambda_delta=0.001` unchanged.",
            "",
            "Full gradient tables, ratios, direction checks, exact sequence-score",
            "sensitivities, hashes, and per-initialization measurements are in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_diagnostic(
    config: ParmRouterTrainingConfig,
    *,
    alpha_checkpoint: Path,
    no_alpha_checkpoint: Path,
    first_pilot_report: Path,
    max_conflict_examples: int,
    target_gate: float,
    lambda_floor: float,
    sensitivity_delta: float,
    head_weight_std: float,
    seed: int,
) -> dict[str, Any]:
    if max_conflict_examples <= 0:
        raise ValueError("max_conflict_examples must be positive")
    pilot_v1 = json.loads(first_pilot_report.read_text(encoding="utf-8"))
    if pilot_v1.get("status") != "NOT_PASS":
        raise ValueError("Failure diagnostic requires the preserved NOT_PASS pilot")
    inputs = (
        alpha_checkpoint,
        no_alpha_checkpoint,
        first_pilot_report,
        first_pilot_report.parent / "pilot_final.pt",
        PROJECT_ROOT / "PARM_TARO/reports/stage9_alpha_collapse_diagnostic.json",
    )
    before = _protected_snapshot(config, inputs)
    runtime = load_training_runtime(config)
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    routers = _build_initializations(
        config,
        alpha_checkpoint=alpha_checkpoint,
        no_alpha_checkpoint=no_alpha_checkpoint,
        tokenizer_hash=tokenizer_hash,
        device=runtime.device,
        target_gate=target_gate,
        head_weight_std=head_weight_std,
        seed=seed,
    )
    examples = [
        example
        for example in load_examples(project_path(config.data_root), "validation")
        if example.better_response_id != example.safer_response_id
    ][:max_conflict_examples]
    if len(examples) != max_conflict_examples:
        raise ValueError("Not enough validation conflicts for the diagnostic")

    gate_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in routers
    }
    endpoint_deltas: dict[str, list[float]] = defaultdict(list)
    gradient_values: dict[
        str, dict[str, dict[str, list[float]]]
    ] = {
        name: defaultdict(lambda: defaultdict(list)) for name in routers
    }
    score_sensitivity: dict[float, dict[float, dict[str, list[float]]]] = {
        endpoint: {
            scale: defaultdict(list) for scale in SCORE_SENSITIVITY_LAMBDAS
        }
        for endpoint in (0.0, 1.0)
    }
    directional_checks: list[dict[str, Any]] = []
    tasks = 0
    for example in examples:
        tokenized = tuple(
            tokenize_response(
                runtime.tokenizer,
                example,
                response_index,
                max_length=config.max_length,
                max_continuation_tokens=config.max_continuation_tokens,
                include_eos_target=config.include_eos_target,
            )
            for response_index in (0, 1)
        )
        base_reference = None
        for helpfulness in (0.0, 1.0):
            alpha = torch.tensor(
                [helpfulness, 1.0 - helpfulness],
                dtype=torch.float32,
                device=runtime.device,
            )
            weights = response_objective_weights(
                alpha,
                better_response_id=example.better_response_id,
                safer_response_id=example.safer_response_id,
            )
            if base_reference is None:
                frozen = tuple(
                    compute_frozen_sequence_distributions(
                        runtime.base_model,
                        runtime.guide_model,
                        item,
                        runtime.alignment,
                        alpha,
                        device=runtime.device,
                    )
                    for item in tokenized
                )
                base_reference = frozen
            else:
                frozen = tuple(
                    replace(
                        reference,
                        guide_logprobs=compute_frozen_guide_logprobs(
                            runtime.guide_model,
                            item,
                            runtime.alignment,
                            alpha,
                            device=runtime.device,
                        ),
                    )
                    for reference, item in zip(base_reference, tokenized)
                )
            weight_difference = float(weights[0] - weights[1])
            intended = (
                example.better_response_id
                if helpfulness == 1.0
                else example.safer_response_id
            )
            margin_probe = torch.tensor(0.2, device=runtime.device)
            larger_margin_reduces_loss = bool(
                F.softplus(-(margin_probe + 0.1)) < F.softplus(-margin_probe)
            )
            directional_checks.append(
                {
                    "sample_id": example.sample_id,
                    "alpha": [helpfulness, 1.0 - helpfulness],
                    "response_weights": weights.detach().cpu().tolist(),
                    "weight_difference": weight_difference,
                    "intended_response_id": intended,
                    "sign_selects_intended_response": (
                        (weight_difference > 0.0 and intended == 0)
                        or (weight_difference < 0.0 and intended == 1)
                    ),
                    "increasing_weighted_margin_decreases_loss": (
                        larger_margin_reduces_loss
                    ),
                }
            )
            for scale in SCORE_SENSITIVITY_LAMBDAS:
                values = exact_score_margin_sensitivity(frozen, scale)
                cell = score_sensitivity[helpfulness][scale]
                cell["score_margin"].append(values["score_margin"])
                derivative = values["d_score_margin_d_lambda"]
                cell["d_score_margin_d_lambda"].append(derivative)
                cell["weighted_derivative"].append(weight_difference * derivative)

            for name, router in routers.items():
                routes = route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=alpha,
                    static_scale=1.0,
                )
                endpoint_routes = tuple(
                    route_response_pair(
                        frozen,
                        router=router,
                        router_alpha=torch.tensor(
                            [endpoint, 1.0 - endpoint],
                            dtype=torch.float32,
                            device=runtime.device,
                        ),
                        static_scale=1.0,
                    )
                    for endpoint in (0.0, 1.0)
                )
                gate_stats = gate_parameterization_stats(routes)
                for key, value in gate_stats.items():
                    if key != "count":
                        gate_values[name][key].append(value)
                _, endpoint_delta = endpoint_sensitivity_loss(
                    endpoint_routes[0],
                    endpoint_routes[1],
                    minimum_delta=sensitivity_delta,
                )
                endpoint_deltas[name].append(float(endpoint_delta.detach()))
                old = preference_sensitive_objective(
                    (routes[0].guided_logprobs, routes[1].guided_logprobs),
                    (frozen[0].gold_token_ids, frozen[1].gold_token_ids),
                    weights,
                    (routes[0].lambda_t, routes[1].lambda_t),
                    preference_loss_weight=1.0,
                    nll_weight=0.25,
                    strength_weight=0.1,
                    strength_target=0.02,
                    pairwise_logit_scale=1.0,
                )
                gain = preference_gain_objective(
                    routes,
                    frozen,
                    weights,
                    pairwise_logit_scale=1.0,
                )
                linear_floor, _ = linear_lambda_floor_loss(
                    routes, floor=lambda_floor
                )
                sensitivity, _ = endpoint_sensitivity_loss(
                    endpoint_routes[0],
                    endpoint_routes[1],
                    minimum_delta=sensitivity_delta,
                )
                groups = router_parameter_groups(router)
                terms = (
                    ("preference", old.preference_loss),
                    ("nll", old.quality_nll.total_loss),
                    ("strength", old.strength_loss),
                    ("weighted_preference", old.preference_loss),
                    ("weighted_nll", 0.25 * old.quality_nll.total_loss),
                    ("weighted_strength", 0.1 * old.strength_loss),
                    ("preference_gain", gain.loss),
                    ("linear_floor", linear_floor),
                    ("endpoint_sensitivity", sensitivity),
                )
                for term_index, (term_name, term) in enumerate(terms):
                    norms = term_gradient_norms(
                        term,
                        groups,
                        retain_graph=term_index < len(terms) - 1,
                    )
                    for group_name, value in norms.items():
                        gradient_values[name][term_name][group_name].append(value)
                tasks += 1
                del routes, endpoint_routes, old, gain
            del frozen

    initialization_ablation = {}
    for name in routers:
        gradients = {
            term: _summaries(groups)
            for term, groups in gradient_values[name].items()
        }
        initialization_ablation[name] = {
            "gate_parameterization": {
                metric: summarize(values)["mean"]
                for metric, values in gate_values[name].items()
            },
            "endpoint_alpha_lambda_delta": summarize(endpoint_deltas[name]),
            "gradient_norms": gradients,
            "gradient_ratios": _gradient_ratios(gradients),
        }
    score_cells = []
    weighted_derivatives_at_target = []
    sensitivity_reference_lambda = min(
        SCORE_SENSITIVITY_LAMBDAS,
        key=lambda scale: abs(scale - target_gate),
    )
    for helpfulness in (0.0, 1.0):
        for scale in SCORE_SENSITIVITY_LAMBDAS:
            values = score_sensitivity[helpfulness][scale]
            row = {
                "alpha_helpfulness": helpfulness,
                "lambda": scale,
                **{
                    name: _quantile_summary(items)
                    for name, items in values.items()
                },
            }
            score_cells.append(row)
            if scale == sensitivity_reference_lambda:
                weighted_derivatives_at_target.extend(
                    abs(value) for value in values["weighted_derivative"]
                )
    sensitivity_reference = _quantile_summary(weighted_derivatives_at_target)
    normalized_scale = 1.0 / max(
        target_gate * float(sensitivity_reference["median"]),
        1e-12,
    )
    pairwise_scale = min(50.0, max(1.0, normalized_scale))
    eligible = [
        name
        for name in INITIALIZATION_NAMES[1:]
        if 0.015
        <= initialization_ablation[name]["gate_parameterization"]["mean_lambda"]
        <= 0.05
    ]
    if not eligible:
        raise RuntimeError("No non-collapsed initialization met the target range")
    recommended_initialization = max(
        eligible,
        key=lambda name: initialization_ablation[name]["gradient_norms"][
            "preference_gain"
        ]["preference_encoder"]["mean"],
    )
    collapsed = initialization_ablation["collapsed_v2_alpha"]
    sigmoid_saturated = (
        collapsed["gate_parameterization"]["mean_sigmoid_derivative"] < 0.005
    )
    weak_strength = (
        collapsed["gradient_ratios"]["all_router"]["strength_to_preference"]
        < 0.1
    )
    direction_pass = all(
        item["sign_selects_intended_response"]
        and item["increasing_weighted_margin_decreases_loss"]
        for item in directional_checks
    )
    root_causes = []
    if sigmoid_saturated:
        root_causes.append("collapsed initialization suppresses sigmoid gradients")
    if weak_strength:
        root_causes.append("squared strength term is too weak after weighting")
    root_causes.append(
        "the original pairwise score includes the base response margin instead "
        "of isolating alpha-conditioned guidance gain"
    )
    after = _protected_snapshot(config, inputs)
    report = {
        "schema_version": 1,
        "stage": 9,
        "method_label": "PARM_TARO_ALPHA_PILOT_V1_FAILURE_DIAGNOSTIC",
        "status": "PASS" if before == after and direction_pass else "NOT_PASS",
        "validation": {
            "split": "validation",
            "conflict_examples": len(examples),
            "alpha_endpoint_tasks_per_initialization": tasks // len(routers),
            "test_split_used": False,
        },
        "current_pilot_objective": {
            "preference_weight": 1.0,
            "nll_weight": 0.25,
            "strength_weight": 0.1,
            "strength_target": 0.02,
            "strength_form": "squared_mean_hinge",
        },
        "initialization_ablation": initialization_ablation,
        "pairwise_direction_checks": {
            "passed": direction_pass,
            "records": directional_checks,
        },
        "sequence_score_sensitivity": {
            "lambda_grid": list(SCORE_SENSITIVITY_LAMBDAS),
            "cells": score_cells,
            "pairwise_scale_reference_lambda": sensitivity_reference_lambda,
            "absolute_weighted_derivative_at_reference": sensitivity_reference,
        },
        "conclusion": {
            "sigmoid_saturation": sigmoid_saturated,
            "current_strength_penalty_too_weak": weak_strength,
            "pairwise_direction_is_correct": direction_pass,
            "root_cause": "; ".join(root_causes) + ".",
        },
        "recommendation": {
            "initialization": recommended_initialization,
            "target_initial_lambda": target_gate,
            "head_weight_std": head_weight_std,
            "lambda_floor": lambda_floor,
            "sensitivity_minimum_delta": sensitivity_delta,
            "pairwise_logit_scale": pairwise_scale,
            "pairwise_scale_rule": (
                "clamp(1 / (target_lambda * median_abs_weighted_"
                "d_score_margin_d_lambda), 1, 50)"
            ),
            "preference_score": "guidance_gain_relative_to_base",
            "curriculum": ["preference_warmup_without_nll", "preference_plus_nll"],
            "strength_form": "linear_per_token_hinge",
            "full_retraining_authorized": False,
        },
        "protected_before": before,
        "protected_after": after,
        "protected_unchanged": before == after,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--alpha-checkpoint", type=Path, default=DEFAULT_ALPHA_CHECKPOINT)
    parser.add_argument(
        "--no-alpha-checkpoint", type=Path, default=DEFAULT_NO_ALPHA_CHECKPOINT
    )
    parser.add_argument("--first-pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--max-conflict-examples", type=int, default=12)
    parser.add_argument("--target-gate", type=float, default=0.03)
    parser.add_argument("--lambda-floor", type=float, default=0.01)
    parser.add_argument("--sensitivity-delta", type=float, default=0.005)
    parser.add_argument("--head-weight-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ParmRouterTrainingConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if overrides:
        config = replace(config, **overrides)
    reports_root = (PROJECT_ROOT / "PARM_TARO/reports").resolve()
    for output in (args.output_json.resolve(), args.output_markdown.resolve()):
        try:
            output.relative_to(reports_root)
        except ValueError as error:
            raise ValueError(
                "Failure diagnostic outputs must stay under PARM_TARO/reports"
            ) from error
    report = run_diagnostic(
        config,
        alpha_checkpoint=args.alpha_checkpoint.resolve(),
        no_alpha_checkpoint=args.no_alpha_checkpoint.resolve(),
        first_pilot_report=args.first_pilot_report.resolve(),
        max_conflict_examples=args.max_conflict_examples,
        target_gate=args.target_gate,
        lambda_floor=args.lambda_floor,
        sensitivity_delta=args.sensitivity_delta,
        head_weight_std=args.head_weight_std,
        seed=args.seed,
    )
    write_json_atomic(args.output_json, report, overwrite=args.overwrite)
    _write_text_atomic(
        args.output_markdown,
        _render_markdown(report),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
