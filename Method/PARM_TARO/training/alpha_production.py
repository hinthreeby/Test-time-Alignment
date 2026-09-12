"""Resumable Stage 9 production training for the validated V3 alpha Router."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from PARM_TARO.scripts.pilot_alpha_preference_v2 import _evaluate
from PARM_TARO.scripts.pilot_alpha_preference_v3 import (
    _build_models,
    _calibrate,
    _objective_terms,
    _set_trainable_phase,
)
from PARM_TARO.training.alpha import response_objective_weights, sample_alpha
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.alpha_residual_router import (
    AlphaResidualSmartRouter,
    load_alpha_residual_checkpoint,
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
    _frozen_snapshot,
    _load_router,
    frozen_response_pair,
)
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree


PRODUCTION_CHECKPOINT_SCHEMA = 1
PHASES = (
    ("normalized_alpha_residual_warmup", True, "warmup_epochs", 0.0),
    ("alpha_residual_plus_nll_quality", False, "quality_epochs", None),
)
REQUIRED_PILOT_CHECKS = (
    "correct_alpha_changes_lambda_materially",
    "constant_alpha_differs_materially",
    "shuffled_alpha_changes_lambda_materially",
    "lambda_not_collapsed",
    "preference_loss_improved",
    "nll_reasonable_vs_no_alpha",
    "shuffled_preference_loss_degrades",
    "frozen_artifacts_unchanged",
    "test_split_unused",
)
REQUIRED_PRODUCTION_CHECKS = (
    "own_metrics_finite",
    "router_lambda_nonconstant",
    "lambda_not_collapsed",
    "correct_alpha_changes_lambda_materially",
    "constant_alpha_differs_materially",
    "shuffled_alpha_changes_lambda_materially",
    "preference_loss_improved",
    "shuffled_preference_loss_degrades",
    "nll_reasonable_vs_no_alpha",
    "frozen_model_gradients_none",
    "optimizer_router_only",
    "optimizer_parameters_fp32",
    "frozen_artifacts_unchanged",
    "test_split_unused",
)


def require_pilot_v3_pass(
    config: AlphaPreferenceProductionConfig,
) -> dict[str, Any]:
    """Authorize production only from the measured, preserved V3 PASS."""

    report_path = project_path(config.required_pilot_v3_report_path)
    checkpoint_path = project_path(config.required_pilot_v3_checkpoint_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = [
        name for name in REQUIRED_PILOT_CHECKS if report.get("checks", {}).get(name) is not True
    ]
    if report.get("status") != "PASS" or failed:
        raise RuntimeError(f"Production requires Pilot V3 PASS; failed={failed}")
    if report.get("method_label") != "PARM_TARO_ALPHA_PREFERENCE_PILOT_V3":
        raise ValueError("Pilot V3 method label mismatch")
    architecture = report.get("architecture", {})
    if architecture.get("alpha_path") != "normalized_alpha_state_pre_sigmoid_residual":
        raise ValueError("Pilot V3 did not validate the production alpha path")
    if architecture.get("hard_coded_alpha_to_lambda") is not False:
        raise ValueError("Pilot V3 architecture hard-codes alpha to lambda")
    checkpoint = report.get("checkpoint", {})
    if checkpoint.get("sha256") != sha256_file(checkpoint_path):
        raise ValueError("Pilot V3 checkpoint hash does not match its report")
    pilot_config = report.get("config", {})
    locked = (
        "seed",
        "calibration_samples",
        "warmup_epochs",
        "quality_epochs",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "residual_hidden_dim",
        "residual_output_weight_std",
        "preference_weight",
        "max_nll_weight",
        "nll_gradient_target_ratio",
        "strength_weight",
        "logit_sensitivity_gradient_target_ratio",
        "max_logit_sensitivity_weight",
        "freeze_state_router_during_warmup",
        "min_control_lambda_delta",
        "min_mean_lambda",
        "max_nll_degradation_vs_no_alpha",
        "require_shuffled_preference_degradation",
        "pilot_v2_report_path",
        "pilot_v2_checkpoint_path",
        "no_alpha_checkpoint",
        "required_alpha_path_diagnostic",
    )
    changed = [
        name
        for name in locked
        if pilot_config.get(name) != getattr(config, name)
    ]
    if changed:
        raise ValueError(f"Production diverges from validated Pilot V3: {changed}")
    _, payload = load_alpha_residual_checkpoint(checkpoint_path)
    if payload["metadata"].get("full_training_authorized") is not False:
        raise ValueError("Pilot V3 checkpoint authorization metadata changed")
    return report


def _production_snapshot(
    config: AlphaPreferenceProductionConfig,
    training_config: ParmRouterTrainingConfig,
    runtime: Any,
) -> dict[str, Any]:
    protected_runs = (
        "taro",
        "v2_no_alpha",
        "v2_alpha",
        "v2_alpha_preference_pilot",
        "v2_alpha_preference_pilot_v2",
        "v2_alpha_preference_pilot_v3",
    )
    training_root = PROJECT_ROOT / "results/parm_taro/training"
    return {
        "frozen_models_and_data": _frozen_snapshot(training_config, runtime),
        "protected_training_runs": {
            name: hash_tree(training_root / name) for name in protected_runs
        },
        "authorization": {
            "pilot_v3_report_sha256": sha256_file(
                project_path(config.required_pilot_v3_report_path)
            ),
            "pilot_v3_checkpoint_sha256": sha256_file(
                project_path(config.required_pilot_v3_checkpoint_path)
            ),
        },
    }


def _phase_spec(
    config: AlphaPreferenceProductionConfig, phase_index: int
) -> tuple[str, bool, int, float]:
    name, alpha_only, epochs_name, nll_weight = PHASES[phase_index]
    epochs = int(getattr(config, epochs_name))
    return name, alpha_only, epochs, float(nll_weight or 0.0)


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def production_checks(
    config: AlphaPreferenceProductionConfig,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    validation_before: Mapping[str, Any],
    validation_after: Mapping[str, Any],
    phase_reports: list[Mapping[str, Any]],
) -> dict[str, bool]:
    constant_control = validation_after.get("constant_control")
    if constant_control is None:
        constant_control_pass = (
            float(validation_after["correct_vs_constant_lambda_delta"]["mean"])
            >= config.min_control_lambda_delta
        )
    else:
        constant_control_pass = bool(constant_control["gate"]["pass"])
    checks = {
        "own_metrics_finite": _all_finite(validation_after)
        and all(_all_finite(report.get("metrics", {})) for report in phase_reports),
        "router_lambda_nonconstant": (
            float(validation_after["correct_lambda"]["std"])
            > config.min_router_lambda_std
        ),
        "lambda_not_collapsed": (
            float(validation_after["correct_lambda"]["mean"])
            >= config.min_mean_lambda
        ),
        "correct_alpha_changes_lambda_materially": (
            float(validation_after["endpoint_alpha_lambda_delta"]["mean"])
            >= config.min_control_lambda_delta
        ),
        "constant_alpha_differs_materially": constant_control_pass,
        "shuffled_alpha_changes_lambda_materially": (
            float(validation_after["correct_vs_shuffled_lambda_delta"]["mean"])
            >= config.min_control_lambda_delta
        ),
        "preference_loss_improved": (
            float(validation_after["correct_preference_loss"]["mean"])
            < float(validation_before["correct_preference_loss"]["mean"])
        ),
        "shuffled_preference_loss_degrades": (
            float(validation_after["shuffled_preference_loss"]["mean"])
            > float(validation_after["correct_preference_loss"]["mean"])
            if config.require_shuffled_preference_degradation
            else True
        ),
        "nll_reasonable_vs_no_alpha": (
            float(validation_after["correct_nll"]["mean"])
            <= float(validation_after["no_alpha_nll"]["mean"])
            + config.max_nll_degradation_vs_no_alpha
        ),
        "frozen_model_gradients_none": True,
        "optimizer_router_only": all(
            report.get("optimizer_router_only") is True for report in phase_reports
        ),
        "optimizer_parameters_fp32": all(
            report.get("optimizer_parameters_fp32") is True
            for report in phase_reports
        ),
        "frozen_artifacts_unchanged": before == after,
        "test_split_unused": True,
    }
    return checks


def _checkpoint(
    router: AlphaResidualSmartRouter,
    path: Path,
    *,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> None:
    save_alpha_residual_checkpoint(
        router,
        path,
        training_state={
            "production_checkpoint_schema": PRODUCTION_CHECKPOINT_SCHEMA,
            **dict(state),
        },
        metadata=metadata,
        overwrite=overwrite,
    )


def _gradient_flags(
    router: AlphaResidualSmartRouter,
    trainable: tuple[torch.nn.Parameter, ...],
) -> dict[str, bool]:
    def nonzero(parameters: Any) -> bool:
        return any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in parameters
        )

    if router.preference_encoder is None:
        raise RuntimeError("Preference encoder disappeared")
    return {
        "router_gradient_nonzero": nonzero(trainable),
        "preference_encoder_gradient_nonzero": nonzero(
            router.preference_encoder.parameters()
        ),
        "alpha_residual_gradient_nonzero": nonzero(
            router.alpha_state_residual.parameters()
        ),
    }


def _run_phase(
    *,
    router: AlphaResidualSmartRouter,
    runtime: Any,
    training_config: ParmRouterTrainingConfig,
    config: AlphaPreferenceProductionConfig,
    calibration: Mapping[str, Any],
    examples: list[Any],
    phase_index: int,
    global_optimizer_step: int,
    resume_state: Mapping[str, Any] | None,
    latest_path: Path,
    metadata: Mapping[str, Any],
    completed_phases: list[dict[str, Any]],
    persistent_state: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    name, alpha_only, epochs, fixed_nll_weight = _phase_spec(config, phase_index)
    nll_weight = (
        fixed_nll_weight
        if phase_index == 0
        else float(calibration["weights"]["nll_quality_phase"])
    )
    trainable = _set_trainable_phase(router, alpha_only=alpha_only)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    assert_optimizer_contains_only(optimizer, router.parameters())
    _assert_optimizer_fp32(optimizer)
    start_epoch = 0
    next_example_index = 0
    phase_optimizer_step = 0
    metrics: dict[str, list[float]] = defaultdict(list)
    skipped: list[str] = []
    flags = {
        "router_gradient_nonzero": False,
        "preference_encoder_gradient_nonzero": False,
        "alpha_residual_gradient_nonzero": False,
    }
    if resume_state is not None:
        start_epoch = int(resume_state["epoch"])
        next_example_index = int(resume_state["next_example_index"])
        phase_optimizer_step = int(resume_state["phase_optimizer_step"])
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        metrics.update(
            {name: list(values) for name, values in resume_state["metrics"].items()}
        )
        skipped.extend(resume_state["skipped_sample_ids"])
        flags.update(resume_state["gradient_checks"])
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    router.train()
    for epoch in range(start_epoch, epochs):
        order = deterministic_example_order(
            examples, epoch=phase_index * 1000 + epoch, seed=config.seed
        )
        for example_index, example in enumerate(order):
            if epoch == start_epoch and example_index < next_example_index:
                continue
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
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Non-finite production loss for {example.sample_id}")
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
                metrics[metric].append(float(value.detach()))
            if pending == config.gradient_accumulation_steps:
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                current = _gradient_flags(router, trainable)
                flags = {key: flags[key] or value for key, value in current.items()}
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                phase_optimizer_step += 1
                global_optimizer_step += 1
                if global_optimizer_step % config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "stage9_alpha_preference_production_progress",
                                "phase": name,
                                "epoch": epoch,
                                "example_index": example_index,
                                "global_optimizer_step": global_optimizer_step,
                                "preference_loss": metrics["preference_loss"][-1],
                                "nll": metrics["nll"][-1],
                                "mean_lambda": metrics["mean_lambda"][-1],
                                "endpoint_lambda_delta": metrics[
                                    "endpoint_lambda_delta"
                                ][-1],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if (
                    global_optimizer_step
                    % config.checkpoint_every_optimizer_steps
                    == 0
                ):
                    _checkpoint(
                        router,
                        latest_path,
                        state={
                            "phase_index": phase_index,
                            "epoch": epoch,
                            "next_example_index": example_index + 1,
                            "global_optimizer_step": global_optimizer_step,
                            "phase_optimizer_step": phase_optimizer_step,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": dict(metrics),
                            "skipped_sample_ids": skipped,
                            "gradient_checks": flags,
                            "completed_phases": completed_phases,
                            **dict(persistent_state),
                        },
                        metadata=metadata,
                        overwrite=True,
                    )
            del frozen, routes, preference
        if pending:
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            current = _gradient_flags(router, trainable)
            flags = {key: flags[key] or value for key, value in current.items()}
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            phase_optimizer_step += 1
            global_optimizer_step += 1
        next_example_index = 0
    if not metrics["total_loss"]:
        raise ValueError(f"Production phase {name} used no train examples")
    report = {
        "name": name,
        "alpha_path_only": alpha_only,
        "epochs": epochs,
        "nll_weight": nll_weight,
        "optimizer_steps": phase_optimizer_step,
        "samples_used": len(metrics["total_loss"]),
        "skipped_sample_ids": sorted(set(skipped)),
        **flags,
        "optimizer_router_only": True,
        "optimizer_parameters_fp32": True,
        "metrics": {metric: summarize(values) for metric, values in metrics.items()},
    }
    return report, global_optimizer_step


def train_alpha_preference_production(
    config: AlphaPreferenceProductionConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    pilot_v3_report = require_pilot_v3_pass(config)
    diagnostic = json.loads(
        project_path(config.required_alpha_path_diagnostic).read_text(encoding="utf-8")
    )
    pilot_v2_report = json.loads(
        project_path(config.pilot_v2_report_path).read_text(encoding="utf-8")
    )
    training_config = ParmRouterTrainingConfig.load_json(
        project_path(config.base_training_config_path)
    )
    training_config = replace(
        training_config,
        device=config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
        output_dir=config.output_dir,
        max_train_samples=config.expected_train_samples,
        max_validation_samples=config.expected_validation_samples,
    )
    output_dir = project_path(config.output_dir)
    try:
        output_dir.relative_to(RESULTS_ROOT)
    except ValueError as error:
        raise ValueError("Production output must stay under results/parm_taro") from error
    latest_path = output_dir / "latest.pt"
    status_path = output_dir / "run_status.json"
    if resume and status_path.is_file():
        return json.loads(status_path.read_text(encoding="utf-8"))
    if resume and not latest_path.is_file():
        raise FileNotFoundError(f"Missing production resume checkpoint: {latest_path}")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite production run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    runtime = load_training_runtime(training_config)
    train_examples = load_examples(
        project_path(training_config.data_root),
        "train",
        max_samples=config.expected_train_samples,
    )
    validation_examples = load_examples(
        project_path(training_config.data_root),
        "validation",
        max_samples=config.expected_validation_samples,
    )
    if len(train_examples) != config.expected_train_samples:
        raise ValueError("Production train split is not exactly 8,000 examples")
    if len(validation_examples) != config.expected_validation_samples:
        raise ValueError("Production validation split is not exactly 500 examples")
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    current_snapshot = _production_snapshot(config, training_config, runtime)
    if resume:
        router, payload = load_alpha_residual_checkpoint(
            latest_path, map_location=runtime.device
        )
        metadata = payload["metadata"]
        state = payload["training_state"]
        if state.get("production_checkpoint_schema") != PRODUCTION_CHECKPOINT_SCHEMA:
            raise ValueError("Incompatible production resume checkpoint")
        if metadata.get("production_config") != config.to_dict():
            raise ValueError("Production resume config mismatch")
        if metadata.get("tokenizer_semantic_sha256") != tokenizer_hash:
            raise ValueError("Production resume tokenizer mismatch")
        if state.get("protected_before") != current_snapshot:
            raise ValueError("A protected production input changed before resume")
        no_alpha_router, _ = _load_router(
            project_path(config.no_alpha_checkpoint),
            kind="smart",
            device=runtime.device,
            tokenizer_hash=tokenizer_hash,
        )
        calibration = state["calibration"]
        validation_before = state["validation_before"]
        phase_index = int(state["phase_index"])
        global_optimizer_step = int(state["global_optimizer_step"])
        completed_phases = list(state["completed_phases"])
        resume_state: Mapping[str, Any] | None = state
        protected_before = state["protected_before"]
    else:
        router, no_alpha_router, source_payload = _build_models(
            config,
            diagnostic,
            device=runtime.device,
            tokenizer_hash=tokenizer_hash,
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
        validation_before = _evaluate(
            router,
            no_alpha_router,
            runtime,
            training_config,
            config,
            {"recommendation": {"pairwise_logit_scale": calibration["pairwise_logit_scale"]}},
            validation_examples,
        )
        protected_before = current_snapshot
        phase_index = 0
        global_optimizer_step = 0
        completed_phases = []
        resume_state = None
        metadata = {
            "schema_version": 1,
            "task": "pku_safe_rlhf_multi_objective_alpha_preference_production",
            "stage": 9,
            "method_label": config.method_label,
            "production_config": config.to_dict(),
            "tokenizer_semantic_sha256": tokenizer_hash,
            "source_checkpoint_sha256": sha256_file(
                project_path(config.pilot_v2_checkpoint_path)
            ),
            "source_checkpoint_metadata": source_payload["metadata"],
            "pilot_v3_report_sha256": sha256_file(
                project_path(config.required_pilot_v3_report_path)
            ),
            "pilot_v3_checkpoint_sha256": sha256_file(
                project_path(config.required_pilot_v3_checkpoint_path)
            ),
            "pilot_v3_validation_evidence": pilot_v3_report["validation_after"],
            "initialization": "fresh_deterministic_v3_residual_from_pilot_v2_source",
            "full_training_authorized": True,
            "test_split_used": False,
        }
        initial_state = {
            "phase_index": 0,
            "epoch": 0,
            "next_example_index": 0,
            "global_optimizer_step": 0,
            "phase_optimizer_step": 0,
            "optimizer_state_dict": None,
            "metrics": {},
            "skipped_sample_ids": [],
            "gradient_checks": {},
            "completed_phases": [],
            "calibration": calibration,
            "validation_before": validation_before,
            "protected_before": protected_before,
        }
        _checkpoint(
            router,
            latest_path,
            state=initial_state,
            metadata=metadata,
            overwrite=False,
        )
        write_json_atomic(output_dir / "calibration.json", calibration)
        write_json_atomic(output_dir / "validation_before.json", validation_before)
        write_json_atomic(output_dir / "resolved_config.json", config.to_dict())
    while phase_index < len(PHASES):
        phase_report, global_optimizer_step = _run_phase(
            router=router,
            runtime=runtime,
            training_config=training_config,
            config=config,
            calibration=calibration,
            examples=train_examples,
            phase_index=phase_index,
            global_optimizer_step=global_optimizer_step,
            resume_state=resume_state,
            latest_path=latest_path,
            metadata=metadata,
            completed_phases=completed_phases,
            persistent_state={
                "calibration": calibration,
                "validation_before": validation_before,
                "protected_before": protected_before,
            },
        )
        completed_phases.append(phase_report)
        phase_path = output_dir / f"phase_{phase_index}.json"
        write_json_atomic(
            phase_path,
            phase_report,
            overwrite=resume and phase_path.exists(),
        )
        phase_index += 1
        resume_state = None
        transition_state = {
            "phase_index": phase_index,
            "epoch": 0,
            "next_example_index": 0,
            "global_optimizer_step": global_optimizer_step,
            "phase_optimizer_step": 0,
            "optimizer_state_dict": None,
            "metrics": {},
            "skipped_sample_ids": [],
            "gradient_checks": {},
            "completed_phases": completed_phases,
            "calibration": calibration,
            "validation_before": validation_before,
            "protected_before": protected_before,
        }
        _checkpoint(
            router,
            latest_path,
            state=transition_state,
            metadata=metadata,
            overwrite=True,
        )
    validation_after = _evaluate(
        router,
        no_alpha_router,
        runtime,
        training_config,
        config,
        {"recommendation": {"pairwise_logit_scale": calibration["pairwise_logit_scale"]}},
        validation_examples,
    )
    _assert_no_frozen_gradients(runtime)
    protected_after = _production_snapshot(config, training_config, runtime)
    checks = production_checks(
        config,
        protected_before,
        protected_after,
        validation_before,
        validation_after,
        completed_phases,
    )
    status = "PASS" if all(checks[name] for name in REQUIRED_PRODUCTION_CHECKS) else "NOT_PASS"
    final_path = output_dir / "final.pt"
    final_state = {
        "phase_index": len(PHASES),
        "global_optimizer_step": global_optimizer_step,
        "completed_phases": completed_phases,
        "calibration": calibration,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "checks": checks,
        "status": status,
    }
    _checkpoint(
        router,
        final_path,
        state=final_state,
        metadata=metadata,
        overwrite=resume and final_path.exists(),
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
        "initialization": "fresh_deterministic_v3_residual_from_pilot_v2_source",
        "split_counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": 0,
        },
        "calibration": calibration,
        "validation_before": validation_before,
        "training_phases": completed_phases,
        "validation_after": validation_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "checkpoint": {"path": str(final_path), "sha256": sha256_file(final_path)},
        "resume_checkpoint": {"path": str(latest_path), "sha256": sha256_file(latest_path)},
        "global_optimizer_steps": global_optimizer_step,
        "test_split_used": False,
    }
    write_json_atomic(status_path, report)
    return report
