"""Run an isolated alpha-sensitive pairwise Router pilot after diagnosis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.training.alpha import (
    build_alpha_controls,
    response_objective_weights,
    sample_alpha,
)
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    SequenceTooLongError,
    deterministic_example_order,
    iter_validation_tasks,
    load_examples,
)
from PARM_TARO.training.engine import (
    _assert_no_frozen_gradients,
    _assert_optimizer_fp32,
    _load_router,
    frozen_response_pair,
    objective_for_routes,
    route_response_pair,
)
from PARM_TARO.training.objective import preference_sensitive_objective
from PARM_TARO.training.pilot_config import AlphaPreferencePilotConfig
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_checkpoint import save_smart_checkpoint
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree


DEFAULT_CONFIG = (
    PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot.json"
)


def _require_diagnostic(config: AlphaPreferencePilotConfig) -> dict[str, Any]:
    path = project_path(config.required_diagnostic_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required alpha-collapse diagnostic: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    full_validation = (
        report.get("status") == "PASS"
        and report.get("validation", {}).get("full_validation") is True
        and report.get("validation", {}).get("examples_evaluated")
        == report.get("validation", {}).get("examples_total")
        and report.get("validation", {}).get("tasks_evaluated")
        == report.get("validation", {}).get("tasks_expected")
        and not report.get("validation", {}).get("skipped_sample_ids")
    )
    collapse = report.get("conclusion", {}).get("objective_collapse") is True
    if not full_validation or not collapse:
        raise RuntimeError(
            "Preference pilot requires a full PASS diagnostic with "
            "objective_collapse=true"
        )
    return report


def _objective(
    routes: tuple[Any, Any],
    frozen: tuple[Any, Any],
    weights: torch.Tensor,
    config: AlphaPreferencePilotConfig,
):
    return preference_sensitive_objective(
        (routes[0].guided_logprobs, routes[1].guided_logprobs),
        (frozen[0].gold_token_ids, frozen[1].gold_token_ids),
        weights,
        (routes[0].lambda_t, routes[1].lambda_t),
        preference_loss_weight=config.preference_loss_weight,
        nll_weight=config.nll_weight,
        strength_weight=config.strength_weight,
        strength_target=config.strength_target,
        pairwise_logit_scale=config.pairwise_logit_scale,
    )


def _evaluate(
    router: SmartTokenRouter,
    no_alpha_router: SmartTokenRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    pilot_config: AlphaPreferencePilotConfig,
    examples: list[Any],
) -> dict[str, Any]:
    tasks = list(iter_validation_tasks(examples, training_config.validation_alpha_grid))
    controls = build_alpha_controls(torch.stack([alpha for _, alpha in tasks]))
    values: dict[str, list[float]] = defaultdict(list)
    correct_lambdas: list[torch.Tensor] = []
    skipped: list[str] = []
    router.eval()
    no_alpha_router.eval()
    with torch.no_grad():
        for task_index, (example, alpha) in enumerate(tasks):
            alpha = alpha.to(runtime.device)
            weights = response_objective_weights(
                alpha,
                better_response_id=example.better_response_id,
                safer_response_id=example.safer_response_id,
            )
            try:
                frozen = frozen_response_pair(
                    runtime, training_config, example, alpha
                )
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            router_alphas = {
                "correct": alpha,
                "shuffled": controls.shuffled[task_index].to(runtime.device),
                "constant": controls.constant[task_index].to(runtime.device),
            }
            routes_by_control = {
                name: route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=router_alpha,
                    static_scale=1.0,
                )
                for name, router_alpha in router_alphas.items()
            }
            endpoint_routes = tuple(
                route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=torch.tensor(
                        [helpfulness, 1.0 - helpfulness],
                        dtype=torch.float32,
                        device=runtime.device,
                    ),
                    static_scale=1.0,
                )
                for helpfulness in (0.0, 1.0)
            )
            objectives = {
                name: _objective(routes, frozen, weights, pilot_config)
                for name, routes in routes_by_control.items()
            }
            no_alpha_routes = route_response_pair(
                frozen,
                router=no_alpha_router,
                router_alpha=None,
                static_scale=1.0,
            )
            no_alpha_nll = objective_for_routes(
                no_alpha_routes, frozen, weights
            ).total_loss
            values["no_alpha_nll"].append(float(no_alpha_nll))
            for name, objective in objectives.items():
                values[f"{name}_preference_loss"].append(
                    float(objective.preference_loss)
                )
                values[f"{name}_nll"].append(
                    float(objective.quality_nll.total_loss)
                )
                values[f"{name}_weighted_margin"].append(
                    float(objective.weighted_preference_margin)
                )
            for response_index in (0, 1):
                correct = routes_by_control["correct"][response_index].lambda_t
                correct_lambdas.append(correct.detach().cpu().reshape(-1))
                values["endpoint_alpha_lambda_delta"].extend(
                    float(value)
                    for value in (
                        endpoint_routes[0][response_index].lambda_t
                        - endpoint_routes[1][response_index].lambda_t
                    )
                    .abs()
                    .detach()
                    .cpu()
                    .reshape(-1)
                )
                for control in ("shuffled", "constant"):
                    changed = routes_by_control[control][response_index].lambda_t
                    values[f"correct_vs_{control}_lambda_delta"].extend(
                        float(value)
                        for value in (correct - changed)
                        .abs()
                        .detach()
                        .cpu()
                        .reshape(-1)
                    )
    if not correct_lambdas:
        raise ValueError("Preference pilot validation produced no tasks")
    lambdas = torch.cat(correct_lambdas).float()
    metrics = {name: summarize(items) for name, items in values.items()}
    metrics["correct_lambda"] = {
        "count": lambdas.numel(),
        "mean": float(lambdas.mean()),
        "std": float(lambdas.std(unbiased=False)),
        "min": float(lambdas.min()),
        "max": float(lambdas.max()),
    }
    metrics["tasks"] = len(values["correct_nll"])
    metrics["skipped_sample_ids"] = sorted(set(skipped))
    return metrics


def _protected_snapshot(
    training_config: ParmRouterTrainingConfig,
    pilot_config: AlphaPreferencePilotConfig,
) -> dict[str, Any]:
    return {
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "pblora": checkpoint_descriptor(
            project_path(training_config.parm_adapter_path)
        ),
        "initial_router_sha256": sha256_file(
            project_path(pilot_config.initial_router_checkpoint)
        ),
        "no_alpha_router_sha256": sha256_file(
            project_path(pilot_config.no_alpha_checkpoint)
        ),
        "data_manifest_sha256": sha256_file(
            project_path(training_config.data_root) / "manifest.json"
        ),
    }


def run_pilot(config: AlphaPreferencePilotConfig) -> dict[str, Any]:
    diagnostic = _require_diagnostic(config)
    training_config = ParmRouterTrainingConfig.load_json(
        project_path(config.base_training_config_path)
    )
    training_config = replace(
        training_config,
        device=config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    output_dir = project_path(config.output_dir)
    try:
        output_dir.relative_to(RESULTS_ROOT)
    except ValueError as error:
        raise ValueError("Pilot output must stay under results/parm_taro") from error
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite preference pilot: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    before = _protected_snapshot(training_config, config)
    runtime = load_training_runtime(training_config)
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    router, initial_payload = _load_router(
        project_path(config.initial_router_checkpoint),
        kind="smart",
        device=runtime.device,
        tokenizer_hash=tokenizer_hash,
    )
    no_alpha_router, _ = _load_router(
        project_path(config.no_alpha_checkpoint),
        kind="smart",
        device=runtime.device,
        tokenizer_hash=tokenizer_hash,
    )
    if not isinstance(router, SmartTokenRouter) or not router.config.use_preference:
        raise ValueError("Pilot initial checkpoint must use preference alpha")
    if not isinstance(no_alpha_router, SmartTokenRouter) or no_alpha_router.config.use_preference:
        raise ValueError("Pilot comparator must be the no-alpha Smart Router")
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    train_examples = deterministic_example_order(
        load_examples(
            project_path(training_config.data_root),
            "train",
            max_samples=config.max_train_samples,
        ),
        epoch=0,
        seed=config.seed,
    )
    validation_examples = load_examples(
        project_path(training_config.data_root),
        "validation",
        max_samples=config.max_validation_samples,
    )
    baseline = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        validation_examples,
    )
    router.train()
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    assert_optimizer_contains_only(optimizer, router.parameters())
    _assert_optimizer_fp32(optimizer)
    optimizer.zero_grad(set_to_none=True)
    training_values: dict[str, list[float]] = defaultdict(list)
    pending = 0
    optimizer_steps = 0
    router_gradient_nonzero = False
    preference_gradient_nonzero = False
    skipped: list[str] = []
    for epoch in range(config.num_epochs):
        order = deterministic_example_order(
            train_examples, epoch=epoch, seed=config.seed
        )
        for example_index, example in enumerate(order):
            alpha = sample_alpha(
                example.sample_id, epoch=epoch, seed=config.seed
            ).to(runtime.device)
            weights = response_objective_weights(
                alpha,
                better_response_id=example.better_response_id,
                safer_response_id=example.safer_response_id,
            )
            try:
                frozen = frozen_response_pair(
                    runtime, training_config, example, alpha
                )
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=alpha,
                static_scale=1.0,
            )
            objective = _objective(routes, frozen, weights, config)
            objective.total_loss.backward()
            _assert_no_frozen_gradients(runtime)
            pending += 1
            for name, value in (
                ("total_loss", objective.total_loss),
                ("preference_loss", objective.preference_loss),
                ("nll", objective.quality_nll.total_loss),
                ("strength_loss", objective.strength_loss),
                ("weighted_margin", objective.weighted_preference_margin),
                ("mean_lambda", objective.mean_lambda),
            ):
                training_values[name].append(float(value.detach()))
            if pending == config.gradient_accumulation_steps:
                for parameter in router.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                router_gradient_nonzero = router_gradient_nonzero or any(
                    parameter.grad is not None
                    and bool((parameter.grad != 0).any())
                    for parameter in router.parameters()
                )
                if router.preference_encoder is None:
                    raise RuntimeError("Preference encoder disappeared")
                preference_gradient_nonzero = preference_gradient_nonzero or any(
                    parameter.grad is not None
                    and bool((parameter.grad != 0).any())
                    for parameter in router.preference_encoder.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    router.parameters(), config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_steps += 1
                if optimizer_steps % config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "stage9_alpha_preference_pilot_progress",
                                "epoch": epoch,
                                "example_index": example_index,
                                "optimizer_step": optimizer_steps,
                                "preference_loss": training_values[
                                    "preference_loss"
                                ][-1],
                                "nll": training_values["nll"][-1],
                                "mean_lambda": training_values["mean_lambda"][-1],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            del frozen, routes, objective
    if pending:
        for parameter in router.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        router_gradient_nonzero = router_gradient_nonzero or any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in router.parameters()
        )
        if router.preference_encoder is None:
            raise RuntimeError("Preference encoder disappeared")
        preference_gradient_nonzero = preference_gradient_nonzero or any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in router.preference_encoder.parameters()
        )
        torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    after_metrics = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        validation_examples,
    )
    after = _protected_snapshot(training_config, config)
    _assert_no_frozen_gradients(runtime)
    checks = {
        "diagnostic_full_pass_and_objective_collapse": True,
        "correct_alpha_changes_lambda_materially": (
            after_metrics["endpoint_alpha_lambda_delta"]["mean"]
            >= config.min_control_lambda_delta
        ),
        "shuffled_alpha_changes_lambda_materially": (
            after_metrics["correct_vs_shuffled_lambda_delta"]["mean"]
            >= config.min_control_lambda_delta
        ),
        "constant_alpha_differs_materially": (
            after_metrics["correct_vs_constant_lambda_delta"]["mean"]
            >= config.min_control_lambda_delta
        ),
        "shuffled_preference_loss_degrades": (
            after_metrics["shuffled_preference_loss"]["mean"]
            > after_metrics["correct_preference_loss"]["mean"]
            if config.require_shuffled_preference_degradation
            else True
        ),
        "lambda_not_collapsed": (
            after_metrics["correct_lambda"]["mean"] >= config.min_mean_lambda
        ),
        "preference_loss_improved": (
            after_metrics["correct_preference_loss"]["mean"]
            < baseline["correct_preference_loss"]["mean"]
        ),
        "nll_reasonable_vs_no_alpha": (
            after_metrics["correct_nll"]["mean"]
            <= after_metrics["no_alpha_nll"]["mean"]
            + config.max_nll_degradation_vs_no_alpha
        ),
        "router_gradient_nonzero": router_gradient_nonzero,
        "preference_encoder_gradient_nonzero": preference_gradient_nonzero,
        "frozen_artifacts_unchanged": before == after,
        "test_split_unused": True,
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    metadata = {
        "schema_version": 1,
        "task": "pku_safe_rlhf_multi_objective_alpha_preference_pilot",
        "pilot_config": config.to_dict(),
        "initial_checkpoint": str(
            project_path(config.initial_router_checkpoint)
        ),
        "initial_checkpoint_metadata": initial_payload["metadata"],
        "tokenizer_semantic_sha256": tokenizer_hash,
        "objective": (
            "pairwise_preference + nll_quality + lambda_strength_floor"
        ),
        "full_training_authorized": False,
    }
    save_smart_checkpoint(
        router,
        output_dir / "pilot_final.pt",
        training_state={
            "optimizer_steps": optimizer_steps,
            "training_examples": len(training_values["total_loss"]),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        metadata=metadata,
    )
    report = {
        "schema_version": 1,
        "stage": 9,
        "method_label": config.method_label,
        "status": status,
        "checks": checks,
        "config": config.to_dict(),
        "diagnostic_conclusion": diagnostic["conclusion"],
        "diagnostic_sha256": sha256_file(
            project_path(config.required_diagnostic_path)
        ),
        "training": {
            "optimizer_steps": optimizer_steps,
            "samples_used": len(training_values["total_loss"]),
            "skipped_sample_ids": sorted(set(skipped)),
            "metrics": {
                name: summarize(items)
                for name, items in training_values.items()
            },
        },
        "validation_before": baseline,
        "validation_after": after_metrics,
        "protected_before": before,
        "protected_after": after,
        "checkpoint": {
            "path": str(output_dir / "pilot_final.pt"),
            "sha256": sha256_file(output_dir / "pilot_final.pt"),
        },
        "full_retraining_started": False,
    }
    write_json_atomic(output_dir / "pilot_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    args = parser.parse_args()
    config = AlphaPreferencePilotConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if overrides:
        config = replace(config, **overrides)
    report = run_pilot(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
