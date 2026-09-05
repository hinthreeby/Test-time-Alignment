"""Run isolated Stage 9 Pilot V3 with normalized pre-sigmoid alpha residual."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.scripts.pilot_alpha_preference_v2 import (
    _endpoint_routes,
    _evaluate,
)
from PARM_TARO.training.alpha import response_objective_weights, sample_alpha
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.alpha_pilot_v2 import (
    endpoint_logit_sensitivity_loss,
    linear_lambda_floor_loss,
    preference_gain_objective,
    term_gradient_norms,
)
from PARM_TARO.training.alpha_residual_router import (
    AlphaResidualConfig,
    AlphaResidualSmartRouter,
    alpha_path_parameter_groups,
    build_from_v2_router,
    save_alpha_residual_checkpoint,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    SequenceTooLongError,
    deterministic_example_order,
    load_examples,
)
from PARM_TARO.training.engine import (
    _assert_no_frozen_gradients,
    _assert_optimizer_fp32,
    _load_router,
    frozen_response_pair,
    route_response_pair,
)
from PARM_TARO.training.pilot_v3_config import AlphaPreferencePilotV3Config
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_checkpoint import load_smart_checkpoint
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot_v3.json"
)


def _require_diagnostic(
    config: AlphaPreferencePilotV3Config,
) -> dict[str, Any]:
    report = json.loads(
        project_path(config.required_alpha_path_diagnostic).read_text(
            encoding="utf-8"
        )
    )
    recommendation = report.get("recommendation", {})
    checks = (
        report.get("status") == "PASS",
        report.get("protected_unchanged") is True,
        report.get("conclusion", {}).get(
            "sensitivity_loss_implementation_correct"
        )
        is True,
        recommendation.get("architecture")
        == "normalized_alpha_state_pre_sigmoid_residual",
        recommendation.get("freeze_state_router_during_alpha_warmup") is True,
        recommendation.get("min_control_lambda_delta") == 0.001,
        recommendation.get("full_retraining_authorized") is False,
    )
    if not all(checks):
        raise RuntimeError(
            "Pilot V3 requires a PASS alpha-path diagnostic with the measured "
            "pre-sigmoid residual recommendation"
        )
    return report


def _protected_snapshot(
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV3Config,
) -> dict[str, Any]:
    paths = (
        config.required_alpha_path_diagnostic,
        config.pilot_v2_report_path,
        config.pilot_v2_checkpoint_path,
        "results/parm_taro/training/v2_alpha_preference_pilot/pilot_report.json",
        "results/parm_taro/training/v2_alpha_preference_pilot/pilot_final.pt",
        config.no_alpha_checkpoint,
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
            path: sha256_file(project_path(path)) for path in paths
        },
    }


def _build_models(
    config: AlphaPreferencePilotV3Config,
    diagnostic: dict[str, Any],
    *,
    device: torch.device,
    tokenizer_hash: str,
) -> tuple[AlphaResidualSmartRouter, SmartTokenRouter, dict[str, Any]]:
    source, source_payload = load_smart_checkpoint(
        project_path(config.pilot_v2_checkpoint_path),
        map_location=device,
    )
    if source_payload["metadata"].get("tokenizer_semantic_sha256") != tokenizer_hash:
        raise ValueError("Pilot V2 source tokenizer hash mismatch")
    if sha256_file(project_path(config.pilot_v2_checkpoint_path)) != diagnostic.get(
        "pilot_v2_checkpoint_sha256"
    ):
        raise ValueError("Pilot V2 source checkpoint changed after diagnosis")
    residual_config = AlphaResidualConfig(
        preference_embedding_dim=source.config.preference_embedding_dim,
        state_bottleneck_dim=source.config.fusion_bottleneck_dim,
        residual_hidden_dim=config.residual_hidden_dim,
        residual_output_weight_std=config.residual_output_weight_std,
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    router = build_from_v2_router(source, residual_config).to(device)
    no_alpha, _ = _load_router(
        project_path(config.no_alpha_checkpoint),
        kind="smart",
        device=device,
        tokenizer_hash=tokenizer_hash,
    )
    if not isinstance(no_alpha, SmartTokenRouter) or no_alpha.config.use_preference:
        raise ValueError("Pilot V3 comparator must be no-alpha Smart Router")
    return router, no_alpha, source_payload


def _objective_terms(
    router: AlphaResidualSmartRouter,
    runtime: Any,
    frozen: tuple[Any, Any],
    alpha: torch.Tensor,
    response_weights: torch.Tensor,
    *,
    pairwise_scale: float,
    lambda_floor: float,
    minimum_logit_delta: float,
) -> tuple[Any, Any, Any, Any, Any]:
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
        response_weights,
        pairwise_logit_scale=pairwise_scale,
    )
    strength, mean_lambda = linear_lambda_floor_loss(routes, floor=lambda_floor)
    logit_sensitivity, logit_delta, lambda_delta = (
        endpoint_logit_sensitivity_loss(
            endpoints[0],
            endpoints[1],
            minimum_logit_delta=minimum_logit_delta,
        )
    )
    return (
        routes,
        preference,
        strength,
        mean_lambda,
        (logit_sensitivity, logit_delta, lambda_delta),
    )


def _calibrate(
    router: AlphaResidualSmartRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV3Config,
    diagnostic: dict[str, Any],
    pilot_v2_report: dict[str, Any],
    examples: list[Any],
) -> dict[str, Any]:
    pairwise_scale = float(
        pilot_v2_report["diagnostic_recommendation"]["pairwise_logit_scale"]
    )
    lambda_floor = float(
        pilot_v2_report["diagnostic_recommendation"]["lambda_floor"]
    )
    minimum_logit_delta = float(diagnostic["recommendation"]["minimum_logit_delta"])
    groups = alpha_path_parameter_groups(router)
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples[: config.calibration_samples]:
        alpha = sample_alpha(example.sample_id, epoch=0, seed=config.seed).to(
            runtime.device
        )
        response_weights = response_objective_weights(
            alpha,
            better_response_id=example.better_response_id,
            safer_response_id=example.safer_response_id,
        )
        frozen = frozen_response_pair(runtime, training_config, example, alpha)
        routes, preference, strength, _, sensitivity_values = _objective_terms(
            router,
            runtime,
            frozen,
            alpha,
            response_weights,
            pairwise_scale=pairwise_scale,
            lambda_floor=lambda_floor,
            minimum_logit_delta=minimum_logit_delta,
        )
        sensitivity = sensitivity_values[0]
        terms = (
            ("preference", preference.loss),
            ("nll", preference.quality_nll.total_loss),
            ("strength", strength),
            ("logit_sensitivity", sensitivity),
        )
        for index, (name, term) in enumerate(terms):
            norms = term_gradient_norms(
                term,
                groups,
                retain_graph=index < len(terms) - 1,
            )
            for group, value in norms.items():
                values[name][group].append(value)
        del frozen, routes, preference
    summaries = {
        term: {group: summarize(items) for group, items in groups.items()}
        for term, groups in values.items()
    }
    preference_norm = float(summaries["preference"]["all_router"]["mean"])
    nll_norm = float(summaries["nll"]["all_router"]["mean"])
    sensitivity_norm = float(
        summaries["logit_sensitivity"]["all_router"]["mean"]
    )
    if min(preference_norm, nll_norm, sensitivity_norm) <= 0.0:
        raise RuntimeError("Pilot V3 calibration found a zero required gradient")
    nll_weight = min(
        config.max_nll_weight,
        config.nll_gradient_target_ratio * preference_norm / nll_norm,
    )
    preference_alpha_path_norm = math.hypot(
        float(summaries["preference"]["preference_encoder"]["mean"]),
        float(summaries["preference"]["alpha_residual"]["mean"]),
    )
    sensitivity_alpha_path_norm = math.hypot(
        float(
            summaries["logit_sensitivity"]["preference_encoder"]["mean"]
        ),
        float(summaries["logit_sensitivity"]["alpha_residual"]["mean"]),
    )
    if min(preference_alpha_path_norm, sensitivity_alpha_path_norm) <= 0.0:
        raise RuntimeError("Pilot V3 alpha-path calibration found a zero gradient")
    sensitivity_weight = min(
        config.max_logit_sensitivity_weight,
        config.logit_sensitivity_gradient_target_ratio
        * preference_alpha_path_norm
        / sensitivity_alpha_path_norm,
    )
    return {
        "samples": min(config.calibration_samples, len(examples)),
        "raw_gradient_norms": summaries,
        "weights": {
            "preference": config.preference_weight,
            "nll_warmup_phase": 0.0,
            "nll_quality_phase": nll_weight,
            "strength": config.strength_weight,
            "logit_sensitivity": sensitivity_weight,
        },
        "alpha_path_gradient_norms_used_for_sensitivity_calibration": {
            "preference": preference_alpha_path_norm,
            "logit_sensitivity": sensitivity_alpha_path_norm,
        },
        "pairwise_logit_scale": pairwise_scale,
        "lambda_floor": lambda_floor,
        "minimum_logit_delta": minimum_logit_delta,
        "sensitivity_weight_hit_cap": (
            sensitivity_weight >= config.max_logit_sensitivity_weight
        ),
    }


def _set_trainable_phase(
    router: AlphaResidualSmartRouter,
    *,
    alpha_only: bool,
) -> tuple[torch.nn.Parameter, ...]:
    for parameter in router.parameters():
        parameter.requires_grad_(not alpha_only)
        parameter.grad = None
    if alpha_only:
        for parameter in router.alpha_residual_parameters():
            parameter.requires_grad_(True)
    parameters = tuple(
        parameter for parameter in router.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("Pilot V3 phase has no trainable Router parameters")
    return parameters


def _train_phase(
    *,
    name: str,
    alpha_only: bool,
    epochs: int,
    nll_weight: float,
    router: AlphaResidualSmartRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferencePilotV3Config,
    calibration: dict[str, Any],
    examples: list[Any],
    phase_index: int,
) -> dict[str, Any]:
    trainable = _set_trainable_phase(router, alpha_only=alpha_only)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    assert_optimizer_contains_only(optimizer, router.parameters())
    _assert_optimizer_fp32(optimizer)
    optimizer.zero_grad(set_to_none=True)
    values: dict[str, list[float]] = defaultdict(list)
    pending = 0
    optimizer_steps = 0
    skipped: list[str] = []
    router_gradient_nonzero = False
    preference_gradient_nonzero = False
    residual_gradient_nonzero = False
    router.train()
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
            routes, preference, strength, mean_lambda, sensitivity_values = (
                _objective_terms(
                    router,
                    runtime,
                    frozen,
                    alpha,
                    response_weights,
                    pairwise_scale=float(calibration["pairwise_logit_scale"]),
                    lambda_floor=float(calibration["lambda_floor"]),
                    minimum_logit_delta=float(calibration["minimum_logit_delta"]),
                )
            )
            sensitivity, logit_delta, lambda_delta = sensitivity_values
            weights = calibration["weights"]
            total = (
                float(weights["preference"]) * preference.loss
                + nll_weight * preference.quality_nll.total_loss
                + float(weights["strength"]) * strength
                + float(weights["logit_sensitivity"]) * sensitivity
            )
            total.backward()
            _assert_no_frozen_gradients(runtime)
            pending += 1
            for metric, value in (
                ("total_loss", total),
                ("preference_loss", preference.loss),
                ("nll", preference.quality_nll.total_loss),
                ("strength_loss", strength),
                ("logit_sensitivity_loss", sensitivity),
                ("mean_lambda", mean_lambda),
                ("endpoint_logit_delta", logit_delta),
                ("endpoint_lambda_delta", lambda_delta),
                ("weighted_guidance_gain", preference.weighted_guidance_gain),
            ):
                values[metric].append(float(value.detach()))
            if pending == config.gradient_accumulation_steps:
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                router_gradient_nonzero = router_gradient_nonzero or any(
                    parameter.grad is not None and bool((parameter.grad != 0).any())
                    for parameter in trainable
                )
                if router.preference_encoder is None:
                    raise RuntimeError("Preference encoder disappeared")
                preference_gradient_nonzero = preference_gradient_nonzero or any(
                    parameter.grad is not None and bool((parameter.grad != 0).any())
                    for parameter in router.preference_encoder.parameters()
                )
                residual_gradient_nonzero = residual_gradient_nonzero or any(
                    parameter.grad is not None and bool((parameter.grad != 0).any())
                    for parameter in router.alpha_state_residual.parameters()
                )
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_steps += 1
                if optimizer_steps % config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "stage9_alpha_preference_pilot_v3_progress",
                                "phase": name,
                                "epoch": epoch,
                                "example_index": example_index,
                                "optimizer_step": optimizer_steps,
                                "preference_loss": values["preference_loss"][-1],
                                "mean_lambda": values["mean_lambda"][-1],
                                "endpoint_lambda_delta": values[
                                    "endpoint_lambda_delta"
                                ][-1],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            del frozen, routes, preference
    if pending:
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        router_gradient_nonzero = router_gradient_nonzero or any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in trainable
        )
        if router.preference_encoder is None:
            raise RuntimeError("Preference encoder disappeared")
        preference_gradient_nonzero = preference_gradient_nonzero or any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in router.preference_encoder.parameters()
        )
        residual_gradient_nonzero = residual_gradient_nonzero or any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in router.alpha_state_residual.parameters()
        )
        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "name": name,
        "alpha_path_only": alpha_only,
        "nll_weight": nll_weight,
        "optimizer_steps": optimizer_steps,
        "samples_used": len(values["total_loss"]),
        "skipped_sample_ids": sorted(set(skipped)),
        "router_gradient_nonzero": router_gradient_nonzero,
        "preference_encoder_gradient_nonzero": preference_gradient_nonzero,
        "alpha_residual_gradient_nonzero": residual_gradient_nonzero,
        "metrics": {metric: summarize(items) for metric, items in values.items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }


def run_pilot(config: AlphaPreferencePilotV3Config) -> dict[str, Any]:
    diagnostic = _require_diagnostic(config)
    pilot_v2_report = json.loads(
        project_path(config.pilot_v2_report_path).read_text(encoding="utf-8")
    )
    if pilot_v2_report.get("status") != "NOT_PASS":
        raise ValueError("Pilot V3 requires preserved NOT_PASS Pilot V2")
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
        raise ValueError("Pilot V3 output must stay under results/parm_taro") from error
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite alpha Pilot V3: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot(training_config, config)
    runtime = load_training_runtime(training_config)
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    router, no_alpha_router, source_payload = _build_models(
        config,
        diagnostic,
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
    calibration = _calibrate(
        router,
        runtime,
        training_config,
        config,
        diagnostic,
        pilot_v2_report,
        train_examples,
    )
    evaluation_diagnostic = {
        "recommendation": {
            "pairwise_logit_scale": calibration["pairwise_logit_scale"]
        }
    }
    baseline = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        evaluation_diagnostic,
        validation_examples,
    )
    warmup = _train_phase(
        name="normalized_alpha_residual_warmup",
        alpha_only=True,
        epochs=config.warmup_epochs,
        nll_weight=0.0,
        router=router,
        runtime=runtime,
        training_config=training_config,
        config=config,
        calibration=calibration,
        examples=train_examples,
        phase_index=0,
    )
    quality = _train_phase(
        name="alpha_residual_plus_nll_quality",
        alpha_only=False,
        epochs=config.quality_epochs,
        nll_weight=float(calibration["weights"]["nll_quality_phase"]),
        router=router,
        runtime=runtime,
        training_config=training_config,
        config=config,
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
        evaluation_diagnostic,
        validation_examples,
    )
    after = _protected_snapshot(training_config, config)
    _assert_no_frozen_gradients(runtime)
    checks = {
        "alpha_path_diagnostic_pass": True,
        "pilot_v2_preserved_not_pass": pilot_v2_report.get("status") == "NOT_PASS",
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
        "alpha_residual_gradient_nonzero": (
            warmup["alpha_residual_gradient_nonzero"]
            and quality["alpha_residual_gradient_nonzero"]
        ),
        "frozen_artifacts_unchanged": before == after,
        "test_split_unused": True,
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    checkpoint_path = output_dir / "pilot_final.pt"
    warmup_optimizer_state = warmup.pop("optimizer_state_dict")
    quality_optimizer_state = quality.pop("optimizer_state_dict")
    save_alpha_residual_checkpoint(
        router,
        checkpoint_path,
        training_state={
            "warmup_optimizer_steps": warmup["optimizer_steps"],
            "quality_optimizer_steps": quality["optimizer_steps"],
            "warmup_optimizer_state_dict": warmup_optimizer_state,
            "quality_optimizer_state_dict": quality_optimizer_state,
        },
        metadata={
            "schema_version": 1,
            "task": "pku_safe_rlhf_multi_objective_alpha_preference_pilot_v3",
            "pilot_config": config.to_dict(),
            "source_checkpoint_sha256": sha256_file(
                project_path(config.pilot_v2_checkpoint_path)
            ),
            "source_checkpoint_metadata": source_payload["metadata"],
            "alpha_path_diagnostic_sha256": sha256_file(
                project_path(config.required_alpha_path_diagnostic)
            ),
            "tokenizer_semantic_sha256": tokenizer_hash,
            "full_training_authorized": False,
        },
    )
    report = {
        "schema_version": 1,
        "stage": 9,
        "method_label": config.method_label,
        "status": status,
        "checks": checks,
        "config": config.to_dict(),
        "architecture": {
            "base": "Pilot V2 SmartTokenRouter",
            "alpha_path": "normalized_alpha_state_pre_sigmoid_residual",
            "hard_coded_alpha_to_lambda": False,
        },
        "diagnostic_conclusion": diagnostic["conclusion"],
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
    config = AlphaPreferencePilotV3Config.load_json(args.config)
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
