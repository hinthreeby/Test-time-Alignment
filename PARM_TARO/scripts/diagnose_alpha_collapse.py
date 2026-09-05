"""Run the read-only Stage 9 fixed-lambda and alpha-sensitivity audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.training.alpha import response_objective_weights
from PARM_TARO.training.alpha_collapse import (
    FIXED_LAMBDAS,
    endpoint_distribution_sensitivity,
    finite_difference_lambda_alpha,
    fixed_lambda_task_metrics,
    smart_router_batch,
    summarize,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    SequenceTooLongError,
    load_examples,
    tokenize_response,
)
from PARM_TARO.training.engine import (
    _assert_no_frozen_gradients,
    _load_router,
    objective_for_routes,
    route_response_pair,
)
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    audit_training_prerequisites,
    load_training_runtime,
    project_path,
)
from PARM_TARO.training.online import (
    compute_frozen_guide_logprobs,
    compute_frozen_sequence_distributions,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import hash_tree


DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results/parm_taro/training/v2_alpha/best.pt"
)
DEFAULT_JSON = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_alpha_collapse_diagnostic.json"
)
DEFAULT_MARKDOWN = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_alpha_collapse_diagnostic.md"
)


def _protected_snapshot(config: ParmRouterTrainingConfig) -> dict[str, Any]:
    return {
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "pblora": checkpoint_descriptor(project_path(config.parm_adapter_path)),
        "data_manifest_sha256": sha256_file(
            project_path(config.data_root) / "manifest.json"
        ),
    }


def _preference_encoder_gradient_norm(
    router: SmartTokenRouter,
    routes: tuple[Any, Any],
    frozen: tuple[Any, Any],
    weights: torch.Tensor,
) -> float:
    if router.preference_encoder is None:
        raise ValueError("Diagnostic checkpoint has no preference encoder")
    objective = objective_for_routes(routes, frozen, weights)
    parameters = tuple(router.preference_encoder.parameters())
    gradients = torch.autograd.grad(
        objective.total_loss,
        parameters,
        allow_unused=True,
        retain_graph=False,
    )
    squared = torch.zeros((), device=weights.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.float().square().sum()
    return float(squared.sqrt())


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 9 Alpha Collapse Diagnostic",
        "",
        "## Status",
        "",
        "```text",
        str(report["status"]),
        "```",
        "",
        "## Fixed-Lambda Sweep",
        "",
        "| alpha_h | lambda* | base NLL | optimal NLL | preferred margin at lambda* |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["fixed_lambda_sweep"]["by_alpha"]:
        lines.append(
            "| {alpha:.2f} | {optimal:.4g} | {base:.8f} | {nll:.8f} | {margin:.8f} |".format(
                alpha=row["alpha_helpfulness"],
                optimal=row["optimal_fixed_lambda"],
                base=row["base_lambda_zero_nll"],
                nll=row["optimal_nll"],
                margin=row["optimal_preferred_vs_nonpreferred_margin"],
            )
        )
    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            f"- PBLORA responds to alpha: `{conclusion['pblora_responds_to_alpha']}`",
            "- Existing Router uses alpha materially: "
            f"`{conclusion['router_uses_alpha_materially']}`",
            f"- All optimal lambdas near zero: `{conclusion['all_optimal_lambdas_near_zero']}`",
            f"- Objective collapse diagnosed: `{conclusion['objective_collapse']}`",
            "",
            "Detailed sweep cells, distribution sensitivity, finite-difference",
            "Router derivatives, preference activation/gradient norms, provenance,",
            "and protected hashes are recorded in the adjacent JSON report.",
            "",
            "## Scientific Decision",
            "",
            str(conclusion["scientific_decision"]),
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
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite diagnostic report: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_diagnostic(
    config: ParmRouterTrainingConfig,
    *,
    checkpoint: Path,
    max_validation_samples: int | None,
    progress_every: int,
) -> dict[str, Any]:
    required_alpha_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    if tuple(config.validation_alpha_grid) != required_alpha_grid:
        raise ValueError(
            "Alpha-collapse diagnostic requires validation_alpha_grid="
            f"{required_alpha_grid}"
        )
    prerequisite = audit_training_prerequisites(config)
    if not prerequisite["ready"]:
        failed = [name for name, passed in prerequisite["checks"].items() if not passed]
        raise RuntimeError(f"Stage 9 prerequisites are not ready: {failed}")
    before = _protected_snapshot(config)
    runtime = load_training_runtime(config)
    all_examples = load_examples(project_path(config.data_root), "validation")
    examples = (
        all_examples[:max_validation_samples]
        if max_validation_samples is not None
        else all_examples
    )
    router, payload = _load_router(
        checkpoint,
        kind="smart",
        device=runtime.device,
        tokenizer_hash=str(runtime.tokenizer_descriptor["semantic_sha256"]),
    )
    if not isinstance(router, SmartTokenRouter) or not router.config.use_preference:
        raise ValueError("Diagnostic requires a preference-aware Smart Router")
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    router.eval()

    sweep: dict[float, dict[float, dict[str, list[float]]]] = {
        alpha: {
            scale: defaultdict(list)
            for scale in FIXED_LAMBDAS
        }
        for alpha in config.validation_alpha_grid
    }
    pblora_sensitivity: dict[str, list[float]] = defaultdict(list)
    router_sensitivity: dict[str, list[float]] = defaultdict(list)
    preference_activation_norms: list[float] = []
    preference_gradient_norms: list[float] = []
    current_router_lambdas: list[float] = []
    skipped: list[str] = []
    tasks = 0
    for example_index, example in enumerate(examples):
        try:
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
        except SequenceTooLongError:
            skipped.append(example.sample_id)
            continue
        endpoint_zero = None
        base_reference = None
        for helpfulness in config.validation_alpha_grid:
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
            tasks += 1
            with torch.no_grad():
                for scale in FIXED_LAMBDAS:
                    metrics = fixed_lambda_task_metrics(frozen, weights, scale)
                    cell = sweep[float(helpfulness)][scale]
                    cell["dual_response_nll"].append(metrics.dual_response_nll)
                    cell["weighted_preference_margin"].append(
                        metrics.weighted_preference_margin
                    )
                    if metrics.preferred_vs_nonpreferred_margin is not None:
                        cell["preferred_margin"].append(
                            metrics.preferred_vs_nonpreferred_margin
                        )
            if helpfulness == config.validation_alpha_grid[0]:
                endpoint_zero = tuple(
                    item.guide_logprobs.detach().clone() for item in frozen
                )
            if helpfulness == config.validation_alpha_grid[-1]:
                if endpoint_zero is None:
                    raise RuntimeError("Missing alpha endpoint distributions")
                for response_index, item in enumerate(frozen):
                    left = replace(item, guide_logprobs=endpoint_zero[response_index])
                    values = endpoint_distribution_sensitivity(left, item)
                    for name, value in values.items():
                        pblora_sensitivity[name].append(value)

            routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=alpha,
                static_scale=1.0,
            )
            current_router_lambdas.extend(
                float(value)
                for route in routes
                for value in route.lambda_t.detach().cpu().reshape(-1)
            )
            for item in frozen:
                derivative = finite_difference_lambda_alpha(
                    router, item, float(helpfulness)
                )
                for name, value in derivative.items():
                    router_sensitivity[name].append(value)
                groups = router.build_static_feature_groups(
                    smart_router_batch(router, item, alpha)
                )
                preference_activation_norms.extend(
                    float(value)
                    for value in groups["preference"]
                    .detach()
                    .float()
                    .norm(dim=-1)
                    .cpu()
                    .reshape(-1)
                )
            preference_gradient_norms.append(
                _preference_encoder_gradient_norm(
                    router, routes, frozen, weights
                )
            )
            del frozen, routes
        if progress_every > 0 and (example_index + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "stage9_alpha_diagnostic_progress",
                        "examples_completed": example_index + 1,
                        "examples_total": len(examples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    by_alpha = []
    all_cells = []
    optimal_lambdas = []
    for helpfulness in config.validation_alpha_grid:
        rows = []
        for scale in FIXED_LAMBDAS:
            values = sweep[float(helpfulness)][scale]
            row = {
                "alpha_helpfulness": helpfulness,
                "alpha_harmlessness": 1.0 - helpfulness,
                "fixed_lambda": scale,
                "dual_response_nll": summarize(values["dual_response_nll"]),
                "weighted_preference_margin": summarize(
                    values["weighted_preference_margin"]
                ),
                "preferred_vs_nonpreferred_margin": (
                    summarize(values["preferred_margin"])
                    if values["preferred_margin"]
                    else None
                ),
            }
            rows.append(row)
            all_cells.append(row)
        optimal = min(rows, key=lambda item: item["dual_response_nll"]["mean"])
        optimal_lambdas.append(float(optimal["fixed_lambda"]))
        by_alpha.append(
            {
                "alpha_helpfulness": helpfulness,
                "base_lambda_zero_nll": rows[0]["dual_response_nll"]["mean"],
                "optimal_fixed_lambda": optimal["fixed_lambda"],
                "optimal_nll": optimal["dual_response_nll"]["mean"],
                "optimal_preferred_vs_nonpreferred_margin": (
                    optimal["preferred_vs_nonpreferred_margin"]["mean"]
                    if optimal["preferred_vs_nonpreferred_margin"] is not None
                    else 0.0
                ),
            }
        )
    expected_tasks = len(examples) * len(config.validation_alpha_grid)
    full_validation = (
        len(examples) == len(all_examples)
        and tasks == expected_tasks
        and not skipped
    )
    all_near_zero = all(value <= 0.001 for value in optimal_lambdas)
    pblora_summary = {
        name: summarize(values) for name, values in pblora_sensitivity.items()
    }
    router_summary = {
        name: summarize(values) for name, values in router_sensitivity.items()
    }
    pblora_responds = (
        pblora_summary["mean_abs_logprob_delta"]["mean"] > 1e-6
        and pblora_summary["mean_js_divergence"]["mean"] > 1e-8
    )
    router_uses_alpha = (
        router_summary["mean_abs_derivative"]["mean"]
        >= config.min_control_lambda_delta
    )
    objective_collapse = full_validation and all_near_zero and pblora_responds
    decision = (
        "The frozen PBLORA responds to alpha, but fixed-lambda gold-token NLL "
        "is minimized near zero for every alpha. Preserve NLL as a quality "
        "term and test a pairwise preference-sensitive Router objective."
        if objective_collapse
        else "Do not revise the objective until the full diagnostic establishes "
        "near-zero optima with a preference-responsive PBLORA guide."
    )
    after = _protected_snapshot(config)
    _assert_no_frozen_gradients(runtime)
    report = {
        "schema_version": 1,
        "stage": 9,
        "method_label": "PARM_TARO_ALPHA_COLLAPSE_DIAGNOSTIC",
        "status": "PASS" if full_validation and before == after else "PARTIAL",
        "validation": {
            "split": "validation",
            "examples_total": len(all_examples),
            "examples_evaluated": len(examples),
            "tasks_evaluated": tasks,
            "tasks_expected": expected_tasks,
            "full_validation": full_validation,
            "skipped_sample_ids": sorted(set(skipped)),
            "test_split_used": False,
        },
        "fixed_lambda_sweep": {
            "lambda_grid": list(FIXED_LAMBDAS),
            "alpha_grid_helpfulness": list(config.validation_alpha_grid),
            "cells": all_cells,
            "by_alpha": by_alpha,
            "optimal_lambda_range": [min(optimal_lambdas), max(optimal_lambdas)],
            "optimal_lambda_changes_meaningfully": (
                max(optimal_lambdas) - min(optimal_lambdas) >= 0.005
            ),
            "base_lambda_zero_nll_over_alpha": summarize(
                [row["base_lambda_zero_nll"] for row in by_alpha]
            ),
        },
        "pblora_alpha_sensitivity": pblora_summary,
        "router_alpha_sensitivity": {
            "finite_difference_epsilon": 0.01,
            **router_summary,
            "current_lambda": summarize(current_router_lambdas),
        },
        "preference_encoder": {
            "activation_l2_norm": summarize(preference_activation_norms),
            "nll_gradient_l2_norm": summarize(preference_gradient_norms),
        },
        "conclusion": {
            "all_optimal_lambdas_near_zero": all_near_zero,
            "pblora_responds_to_alpha": pblora_responds,
            "router_uses_alpha_materially": router_uses_alpha,
            "objective_collapse": objective_collapse,
            "scientific_decision": decision,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "metadata": payload["metadata"],
        },
        "protected_before": before,
        "protected_after": after,
        "protected_unchanged": before == after,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
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
    output_json = args.output_json.resolve()
    output_markdown = args.output_markdown.resolve()
    for output in (output_json, output_markdown):
        try:
            output.relative_to(reports_root)
        except ValueError as error:
            raise ValueError("Diagnostic outputs must stay under PARM_TARO/reports") from error
    report = run_diagnostic(
        config,
        checkpoint=args.checkpoint.resolve(),
        max_validation_samples=args.max_validation_samples,
        progress_every=args.progress_every,
    )
    write_json_atomic(output_json, report, overwrite=args.overwrite)
    _write_text_atomic(
        output_markdown,
        _render_markdown(report),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
