"""Run the isolated measured-design Stage 9 alpha-preference pilot V2."""

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
from PARM_TARO.training.alpha_pilot_v2 import (
    curriculum_objective,
    endpoint_sensitivity_loss,
    linear_lambda_floor_loss,
    preference_gain_objective,
    reset_lambda_head,
    router_parameter_groups,
    term_gradient_norms,
    transplant_no_alpha_router,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.constant_control import ConstantControlAccumulator
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
from PARM_TARO.training.pilot_v2_config import AlphaPreferencePilotV2Config
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_checkpoint import save_smart_checkpoint
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot_v2.json"
)


def _require_failure_diagnostic(
    config: AlphaPreferencePilotV2Config,
) -> dict[str, Any]:
    report = json.loads(
        project_path(config.required_failure_diagnostic_path).read_text(
            encoding="utf-8"
        )
    )
    recommendation = report.get("recommendation", {})
    allowed_initializations = {
        "no_alpha_transplant_fresh_preference_head",
        "fresh_alpha_target_gate",
    }
    checks = (
        report.get("status") == "PASS",
        report.get("protected_unchanged") is True,
        report.get("pairwise_direction_checks", {}).get("passed") is True,
        recommendation.get("initialization") in allowed_initializations,
        recommendation.get("full_retraining_authorized") is False,
    )
    if not all(checks):
        raise RuntimeError(
            "Pilot V2 requires a PASS failure diagnostic with a non-collapsed "
            "initialization and valid pairwise direction checks"
        )
    return report


def _protected_snapshot(
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV2Config,
) -> dict[str, Any]:
    paths = (
        config.first_pilot_report_path,
        config.first_pilot_checkpoint_path,
        config.alpha_checkpoint,
        config.no_alpha_checkpoint,
        config.required_failure_diagnostic_path,
    )
    return {
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "pblora": checkpoint_descriptor(
            project_path(training_config.parm_adapter_path)
        ),
        "data_manifest_sha256": sha256_file(
            project_path(training_config.data_root) / "manifest.json"
        ),
        "read_only_inputs": {
            name: sha256_file(project_path(name)) for name in paths
        },
    }


def _build_router(
    diagnostic: dict[str, Any],
    config: AlphaPreferencePilotV2Config,
    training_config: ParmRouterTrainingConfig,
    *,
    device: torch.device,
    tokenizer_hash: str,
) -> tuple[SmartTokenRouter, SmartTokenRouter, dict[str, Any]]:
    no_alpha, no_alpha_payload = _load_router(
        project_path(config.no_alpha_checkpoint),
        kind="smart",
        device=device,
        tokenizer_hash=tokenizer_hash,
    )
    if not isinstance(no_alpha, SmartTokenRouter) or no_alpha.config.use_preference:
        raise ValueError("Pilot V2 comparator must be a no-alpha Smart Router")
    recommendation = diagnostic["recommendation"]
    target = float(recommendation["target_initial_lambda"])
    weight_std = float(recommendation["head_weight_std"])
    alpha_config = SmartRouterConfig.load_json(
        project_path(training_config.router_config_path)
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if recommendation["initialization"] == (
        "no_alpha_transplant_fresh_preference_head"
    ):
        router = transplant_no_alpha_router(
            no_alpha,
            alpha_config,
            target_gate=target,
            head_weight_std=weight_std,
        )
    elif recommendation["initialization"] == "fresh_alpha_target_gate":
        router = SmartTokenRouter(alpha_config).to(device)
        reset_lambda_head(router, target_gate=target, weight_std=weight_std)
    else:
        raise ValueError("Diagnostic recommended an unsafe initialization")
    router = router.to(device)
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    router.eval()
    return router, no_alpha, no_alpha_payload


def _endpoint_routes(
    router: SmartTokenRouter,
    frozen: tuple[Any, Any],
    device: torch.device,
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    return tuple(
        route_response_pair(
            frozen,
            router=router,
            router_alpha=torch.tensor(
                [helpfulness, 1.0 - helpfulness],
                dtype=torch.float32,
                device=device,
            ),
            static_scale=1.0,
        )
        for helpfulness in (0.0, 1.0)
    )


def _calibrate_weights(
    router: SmartTokenRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV2Config,
    diagnostic: dict[str, Any],
    examples: list[Any],
) -> dict[str, Any]:
    recommendation = diagnostic["recommendation"]
    pairwise_scale = float(recommendation["pairwise_logit_scale"])
    lambda_floor = float(recommendation["lambda_floor"])
    sensitivity_delta = float(recommendation["sensitivity_minimum_delta"])
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    groups = router_parameter_groups(router)
    for index, example in enumerate(examples[: config.calibration_samples]):
        alpha = sample_alpha(example.sample_id, epoch=0, seed=config.seed).to(
            runtime.device
        )
        weights = response_objective_weights(
            alpha,
            better_response_id=example.better_response_id,
            safer_response_id=example.safer_response_id,
        )
        frozen = frozen_response_pair(runtime, training_config, example, alpha)
        routes = route_response_pair(
            frozen,
            router=router,
            router_alpha=alpha,
            static_scale=1.0,
        )
        endpoints = _endpoint_routes(router, frozen, runtime.device)
        preference = preference_gain_objective(
            routes,
            frozen,
            weights,
            pairwise_logit_scale=pairwise_scale,
        )
        strength, _ = linear_lambda_floor_loss(routes, floor=lambda_floor)
        sensitivity, _ = endpoint_sensitivity_loss(
            endpoints[0], endpoints[1], minimum_delta=sensitivity_delta
        )
        terms = (
            ("preference", preference.loss),
            ("nll", preference.quality_nll.total_loss),
            ("strength", strength),
            ("sensitivity", sensitivity),
        )
        for term_index, (name, term) in enumerate(terms):
            norms = term_gradient_norms(
                term,
                groups,
                retain_graph=term_index < len(terms) - 1,
            )
            for group, value in norms.items():
                values[name][group].append(value)
        del frozen, routes, endpoints, preference
    summaries = {
        term: {group: summarize(items) for group, items in groups.items()}
        for term, groups in values.items()
    }
    preference_norm = float(summaries["preference"]["all_router"]["mean"])
    nll_norm = float(summaries["nll"]["all_router"]["mean"])
    sensitivity_norm = float(summaries["sensitivity"]["all_router"]["mean"])
    if preference_norm <= 0.0 or nll_norm <= 0.0 or sensitivity_norm <= 0.0:
        raise RuntimeError("Pilot V2 calibration found a zero required gradient")
    nll_weight = min(
        config.max_nll_weight,
        config.nll_gradient_target_ratio * preference_norm / nll_norm,
    )
    sensitivity_weight = min(
        config.max_sensitivity_weight,
        config.sensitivity_gradient_target_ratio
        * preference_norm
        / sensitivity_norm,
    )
    return {
        "samples": min(config.calibration_samples, len(examples)),
        "raw_gradient_norms": summaries,
        "weights": {
            "preference": config.preference_weight,
            "nll_quality_phase": nll_weight,
            "nll_warmup_phase": 0.0,
            "strength": config.strength_weight,
            "sensitivity": sensitivity_weight,
        },
        "targets": {
            "nll_to_preference_gradient_ratio": config.nll_gradient_target_ratio,
            "sensitivity_to_preference_gradient_ratio": (
                config.sensitivity_gradient_target_ratio
            ),
        },
        "pairwise_logit_scale": pairwise_scale,
        "lambda_floor": lambda_floor,
        "sensitivity_minimum_delta": sensitivity_delta,
    }


def _evaluate(
    router: SmartTokenRouter,
    no_alpha_router: SmartTokenRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV2Config,
    diagnostic: dict[str, Any],
    examples: list[Any],
) -> dict[str, Any]:
    pairwise_scale = float(diagnostic["recommendation"]["pairwise_logit_scale"])
    tasks = list(iter_validation_tasks(examples, training_config.validation_alpha_grid))
    controls = build_alpha_controls(torch.stack([alpha for _, alpha in tasks]))
    values: dict[str, list[float]] = defaultdict(list)
    correct_lambdas: list[torch.Tensor] = []
    constant_control = ConstantControlAccumulator()
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
                frozen = frozen_response_pair(runtime, training_config, example, alpha)
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            router_alphas = {
                "correct": alpha,
                "shuffled": controls.shuffled[task_index].to(runtime.device),
                "constant": controls.constant[task_index].to(runtime.device),
            }
            routes = {
                name: route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=router_alpha,
                    static_scale=1.0,
                )
                for name, router_alpha in router_alphas.items()
            }
            endpoints = _endpoint_routes(router, frozen, runtime.device)
            objectives = {
                name: preference_gain_objective(
                    route,
                    frozen,
                    weights,
                    pairwise_logit_scale=pairwise_scale,
                )
                for name, route in routes.items()
            }
            no_alpha_routes = route_response_pair(
                frozen,
                router=no_alpha_router,
                router_alpha=None,
                static_scale=1.0,
            )
            values["no_alpha_nll"].append(
                float(objective_for_routes(no_alpha_routes, frozen, weights).total_loss)
            )
            for name, objective in objectives.items():
                values[f"{name}_preference_loss"].append(float(objective.loss))
                values[f"{name}_nll"].append(
                    float(objective.quality_nll.total_loss)
                )
                values[f"{name}_weighted_guidance_gain"].append(
                    float(objective.weighted_guidance_gain)
                )
            task_constant_deltas: list[torch.Tensor] = []
            for response_index in (0, 1):
                correct = routes["correct"][response_index].lambda_t
                correct_lambdas.append(correct.detach().cpu().reshape(-1))
                values["endpoint_alpha_lambda_delta"].extend(
                    float(value)
                    for value in (
                        endpoints[0][response_index].lambda_t
                        - endpoints[1][response_index].lambda_t
                    )
                    .abs()
                    .cpu()
                    .reshape(-1)
                )
                for control in ("shuffled", "constant"):
                    delta = (
                        correct - routes[control][response_index].lambda_t
                    ).abs().detach().cpu().reshape(-1)
                    values[f"correct_vs_{control}_lambda_delta"].extend(
                        float(value) for value in delta
                    )
                    if control == "constant":
                        task_constant_deltas.append(delta)
            constant_control.update(
                alpha,
                router_alphas["constant"],
                torch.cat(task_constant_deltas),
            )
    if not correct_lambdas:
        raise ValueError("Pilot V2 validation produced no tasks")
    lambdas = torch.cat(correct_lambdas).float()
    metrics = {name: summarize(items) for name, items in values.items()}
    metrics["correct_lambda"] = {
        "count": int(lambdas.numel()),
        "mean": float(lambdas.mean()),
        "std": float(lambdas.std(unbiased=False)),
        "min": float(lambdas.min()),
        "max": float(lambdas.max()),
    }
    metrics["tasks"] = len(values["correct_nll"])
    metrics["skipped_sample_ids"] = sorted(set(skipped))
    metrics["constant_control"] = constant_control.finalize(
        threshold=config.min_control_lambda_delta
    )
    return metrics


def _train_phase(
    *,
    name: str,
    epochs: int,
    nll_weight: float,
    router: SmartTokenRouter,
    optimizer: torch.optim.Optimizer,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV2Config,
    diagnostic: dict[str, Any],
    calibration: dict[str, Any],
    examples: list[Any],
    phase_index: int,
) -> dict[str, Any]:
    pairwise_scale = float(calibration["pairwise_logit_scale"])
    weights = calibration["weights"]
    values: dict[str, list[float]] = defaultdict(list)
    skipped: list[str] = []
    optimizer_steps = 0
    pending = 0
    router_gradient_nonzero = False
    preference_gradient_nonzero = False
    router.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        order = deterministic_example_order(
            examples,
            epoch=phase_index * 1000 + epoch,
            seed=config.seed,
        )
        for example_index, example in enumerate(order):
            alpha = sample_alpha(
                example.sample_id,
                epoch=phase_index * 1000 + epoch,
                seed=config.seed,
            ).to(runtime.device)
            response_weights = response_objective_weights(
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
            endpoints = _endpoint_routes(router, frozen, runtime.device)
            objective = curriculum_objective(
                routes,
                endpoints,
                frozen,
                response_weights,
                pairwise_logit_scale=pairwise_scale,
                preference_weight=float(weights["preference"]),
                nll_weight=nll_weight,
                strength_weight=float(weights["strength"]),
                sensitivity_weight=float(weights["sensitivity"]),
                lambda_floor=float(calibration["lambda_floor"]),
                sensitivity_minimum_delta=float(
                    calibration["sensitivity_minimum_delta"]
                ),
            )
            objective.total_loss.backward()
            _assert_no_frozen_gradients(runtime)
            pending += 1
            for metric, value in (
                ("total_loss", objective.total_loss),
                ("preference_loss", objective.preference_loss),
                ("nll", objective.quality_nll.total_loss),
                ("strength_loss", objective.strength_loss),
                ("sensitivity_loss", objective.sensitivity_loss),
                ("mean_lambda", objective.mean_lambda),
                ("endpoint_lambda_delta", objective.endpoint_lambda_delta),
                ("weighted_guidance_gain", objective.weighted_guidance_gain),
            ):
                values[metric].append(float(value.detach()))
            if pending == config.gradient_accumulation_steps:
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
                pending = 0
                if optimizer_steps % config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "stage9_alpha_preference_pilot_v2_progress",
                                "phase": name,
                                "epoch": epoch,
                                "example_index": example_index,
                                "optimizer_step": optimizer_steps,
                                "preference_loss": values["preference_loss"][-1],
                                "nll": values["nll"][-1],
                                "mean_lambda": values["mean_lambda"][-1],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            del frozen, routes, endpoints, objective
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
    return {
        "name": name,
        "epochs": epochs,
        "nll_weight": nll_weight,
        "optimizer_steps": optimizer_steps,
        "samples_used": len(values["total_loss"]),
        "skipped_sample_ids": sorted(set(skipped)),
        "router_gradient_nonzero": router_gradient_nonzero,
        "preference_encoder_gradient_nonzero": preference_gradient_nonzero,
        "metrics": {metric: summarize(items) for metric, items in values.items()},
    }


def run_pilot(config: AlphaPreferencePilotV2Config) -> dict[str, Any]:
    diagnostic = _require_failure_diagnostic(config)
    first_pilot = json.loads(
        project_path(config.first_pilot_report_path).read_text(encoding="utf-8")
    )
    if first_pilot.get("status") != "NOT_PASS":
        raise ValueError("Pilot V2 requires the preserved NOT_PASS pilot V1")
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
        raise ValueError("Pilot V2 output must stay under results/parm_taro") from error
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite alpha pilot V2: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot(training_config, config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    runtime = load_training_runtime(training_config)
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    router, no_alpha_router, no_alpha_payload = _build_router(
        diagnostic,
        config,
        training_config,
        device=runtime.device,
        tokenizer_hash=tokenizer_hash,
    )
    train_examples = load_examples(
        project_path(training_config.data_root),
        "train",
        max_samples=config.max_train_samples,
    )
    validation_examples = load_examples(
        project_path(training_config.data_root),
        "validation",
        max_samples=config.max_validation_samples,
    )
    calibration = _calibrate_weights(
        router,
        runtime,
        training_config,
        config,
        diagnostic,
        train_examples,
    )
    baseline = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        diagnostic,
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
    warmup = _train_phase(
        name="preference_warmup_without_nll",
        epochs=config.warmup_epochs,
        nll_weight=0.0,
        router=router,
        optimizer=optimizer,
        runtime=runtime,
        training_config=training_config,
        config=config,
        diagnostic=diagnostic,
        calibration=calibration,
        examples=train_examples,
        phase_index=0,
    )
    quality = _train_phase(
        name="preference_plus_nll_quality",
        epochs=config.quality_epochs,
        nll_weight=float(calibration["weights"]["nll_quality_phase"]),
        router=router,
        optimizer=optimizer,
        runtime=runtime,
        training_config=training_config,
        config=config,
        diagnostic=diagnostic,
        calibration=calibration,
        examples=train_examples,
        phase_index=1,
    )
    after_metrics = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        diagnostic,
        validation_examples,
    )
    after = _protected_snapshot(training_config, config)
    _assert_no_frozen_gradients(runtime)
    checks = {
        "failure_diagnostic_pass": True,
        "first_pilot_preserved_not_pass": first_pilot.get("status") == "NOT_PASS",
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
        "router_gradient_nonzero": (
            warmup["router_gradient_nonzero"] and quality["router_gradient_nonzero"]
        ),
        "preference_encoder_gradient_nonzero": (
            warmup["preference_encoder_gradient_nonzero"]
            and quality["preference_encoder_gradient_nonzero"]
        ),
        "frozen_artifacts_unchanged": before == after,
        "test_split_unused": True,
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    metadata = {
        "schema_version": 1,
        "task": "pku_safe_rlhf_multi_objective_alpha_preference_pilot_v2",
        "pilot_config": config.to_dict(),
        "diagnostic_sha256": sha256_file(
            project_path(config.required_failure_diagnostic_path)
        ),
        "initialization": diagnostic["recommendation"]["initialization"],
        "tokenizer_semantic_sha256": tokenizer_hash,
        "objective": (
            "base_relative_preference_gain + calibrated_nll + linear_lambda_floor "
            "+ endpoint_alpha_sensitivity"
        ),
        "no_alpha_checkpoint_metadata": no_alpha_payload["metadata"],
        "full_training_authorized": False,
    }
    checkpoint_path = output_dir / "pilot_final.pt"
    save_smart_checkpoint(
        router,
        checkpoint_path,
        training_state={
            "warmup_optimizer_steps": warmup["optimizer_steps"],
            "quality_optimizer_steps": quality["optimizer_steps"],
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
        "diagnostic_recommendation": diagnostic["recommendation"],
        "calibration": calibration,
        "validation_before": baseline,
        "training_phases": [warmup, quality],
        "validation_after": after_metrics,
        "protected_before": before,
        "protected_after": after,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
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
    config = AlphaPreferencePilotV2Config.load_json(args.config)
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
