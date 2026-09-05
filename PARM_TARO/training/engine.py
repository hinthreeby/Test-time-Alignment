"""Stage 9 training, comparisons, alpha controls, and frozen audits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from PARM_TARO.training.alpha import (
    build_alpha_controls,
    is_near_tie,
    response_objective_weights,
    sample_alpha,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    MultiObjectiveExample,
    SequenceTooLongError,
    deterministic_example_order,
    iter_validation_tasks,
    load_examples,
    tokenize_response,
)
from PARM_TARO.training.objective import (
    DualResponseObjective,
    dual_response_objective,
)
from PARM_TARO.training.online import (
    FrozenSequenceDistributions,
    compute_frozen_sequence_distributions,
)
from PARM_TARO.training.routing import (
    ParmRoute,
    router_parm_route,
    static_parm_route,
)
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    ParmTrainingRuntime,
    audit_training_prerequisites,
    load_training_runtime,
    project_path,
    resolved_paths,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.checkpoint import (
    load_taro_checkpoint,
    save_taro_checkpoint,
)
from router_v2.config import TARORouterConfig
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.model import TAROTokenRouter
from router_v2.smart_checkpoint import (
    load_smart_checkpoint,
    save_smart_checkpoint,
)
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import (
    assert_optimizer_contains_only,
    hash_tree,
)


Router = TAROTokenRouter | SmartTokenRouter


@dataclass
class MethodAccumulator:
    losses: list[float] = field(default_factory=list)
    lambdas: list[torch.Tensor] = field(default_factory=list)
    alpha_helpfulness: list[float] = field(default_factory=list)
    mean_lambda_per_task: list[float] = field(default_factory=list)

    def update(
        self,
        objective: DualResponseObjective,
        routes: tuple[ParmRoute, ParmRoute],
        alpha: torch.Tensor,
    ) -> None:
        self.losses.append(float(objective.total_loss.detach()))
        values = torch.cat(
            [route.lambda_t.detach().cpu().reshape(-1) for route in routes]
        ).float()
        self.lambdas.append(values)
        self.alpha_helpfulness.append(float(alpha[0]))
        self.mean_lambda_per_task.append(float(values.mean()))

    def finalize(self) -> dict[str, Any]:
        if not self.losses or not self.lambdas:
            raise ValueError("Cannot finalize empty comparison metrics")
        values = torch.cat(self.lambdas)
        losses = torch.tensor(self.losses, dtype=torch.float64)
        alpha = torch.tensor(self.alpha_helpfulness, dtype=torch.float64)
        means = torch.tensor(self.mean_lambda_per_task, dtype=torch.float64)
        correlation = None
        if alpha.numel() >= 2:
            left = alpha - alpha.mean()
            right = means - means.mean()
            denominator = torch.sqrt(left.square().sum() * right.square().sum())
            if float(denominator) > 0.0:
                correlation = float((left * right).sum() / denominator)
        return {
            "tasks": len(self.losses),
            "dual_response_token_nll": float(losses.mean()),
            "loss_std": float(losses.std(unbiased=False)),
            "lambda": {
                "count": int(values.numel()),
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
            },
            "lambda_alpha_helpfulness_correlation": correlation,
            "finite": bool(torch.isfinite(losses).all() and torch.isfinite(values).all()),
        }


def _assert_no_frozen_gradients(runtime: ParmTrainingRuntime) -> None:
    offenders = [
        f"{model_name}.{parameter_name}"
        for model_name, model in (
            ("base", runtime.base_model),
            ("parm_pblora", runtime.guide_model),
        )
        for parameter_name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(
            f"Frozen Stage 9 model parameters received gradients: {offenders[:5]}"
        )


def _assert_optimizer_fp32(optimizer: torch.optim.Optimizer) -> None:
    non_fp32 = [
        str(parameter.dtype)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.dtype != torch.float32
    ]
    if non_fp32:
        raise ValueError(f"Stage 9 Router optimizer parameters must be FP32: {non_fp32[:5]}")


def _router_name(config: ParmRouterTrainingConfig) -> str:
    return config.stage


def build_router(
    config: ParmRouterTrainingConfig,
    *,
    vocab_size: int,
    device: torch.device,
) -> Router:
    path = project_path(config.router_config_path)
    if config.router_kind == "taro":
        router_config = TARORouterConfig.load_json(path)
        if router_config.vocab_size != vocab_size:
            raise ValueError("TARO config vocabulary does not match tokenizer")
        return TAROTokenRouter(router_config).to(device)
    router_config = SmartRouterConfig.load_json(path)
    expected_variant = {
        "v2_no_alpha": "v2_topk_state_history",
        "v2_alpha": "v2_topk_state_history_alpha",
    }[config.stage]
    if router_config.variant != expected_variant:
        raise ValueError("Smart Router variant does not match Stage 9 stage")
    if router_config.vocab_size != vocab_size:
        raise ValueError("Smart Router vocabulary does not match tokenizer")
    expected_max_position = config.max_continuation_tokens + int(
        config.include_eos_target
    ) - 1
    if router_config.max_position < expected_max_position:
        raise ValueError("Smart Router max_position is too small")
    if router_config.preference_dim != 2:
        raise ValueError("Stage 9 Smart Router requires preference_dim=2")
    return SmartTokenRouter(router_config).to(device)


def _save_router(
    model: Router,
    path: Path,
    *,
    training_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> None:
    if isinstance(model, TAROTokenRouter):
        save_taro_checkpoint(
            model,
            path,
            training_state=training_state,
            metadata=metadata,
            overwrite=overwrite,
        )
    else:
        save_smart_checkpoint(
            model,
            path,
            training_state=training_state,
            metadata=metadata,
            overwrite=overwrite,
        )


def _load_router(
    path: Path,
    *,
    kind: str,
    device: torch.device,
    tokenizer_hash: str,
) -> tuple[Router, dict[str, Any]]:
    if kind == "taro":
        model, payload = load_taro_checkpoint(path, map_location=device)
    else:
        model, payload = load_smart_checkpoint(path, map_location=device)
    metadata = payload["metadata"]
    if metadata.get("tokenizer_semantic_sha256") != tokenizer_hash:
        raise ValueError(f"Comparator tokenizer hash mismatch: {path}")
    if metadata.get("task") != "pku_safe_rlhf_multi_objective":
        raise ValueError(f"Comparator task metadata mismatch: {path}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _tokenized_pair(
    runtime: ParmTrainingRuntime,
    config: ParmRouterTrainingConfig,
    example: MultiObjectiveExample,
) -> tuple[Any, Any]:
    return tuple(
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


def frozen_response_pair(
    runtime: ParmTrainingRuntime,
    config: ParmRouterTrainingConfig,
    example: MultiObjectiveExample,
    alpha: torch.Tensor,
) -> tuple[FrozenSequenceDistributions, FrozenSequenceDistributions]:
    tokenized = _tokenized_pair(runtime, config, example)
    return tuple(
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


def route_response_pair(
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    *,
    router: Router | None,
    router_alpha: torch.Tensor | None,
    static_scale: float,
) -> tuple[ParmRoute, ParmRoute]:
    if router is None:
        return tuple(static_parm_route(item, static_scale) for item in frozen)
    return tuple(
        router_parm_route(router, item, router_alpha=router_alpha)
        for item in frozen
    )


def objective_for_routes(
    routes: tuple[ParmRoute, ParmRoute],
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    weights: torch.Tensor,
) -> DualResponseObjective:
    return dual_response_objective(
        (routes[0].guided_logprobs, routes[1].guided_logprobs),
        (frozen[0].gold_token_ids, frozen[1].gold_token_ids),
        weights,
    )


def training_example_objective(
    router: Router,
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    example: MultiObjectiveExample,
    alpha: torch.Tensor,
) -> tuple[DualResponseObjective, tuple[ParmRoute, ParmRoute], torch.Tensor]:
    weights = response_objective_weights(
        alpha,
        better_response_id=example.better_response_id,
        safer_response_id=example.safer_response_id,
    )
    router_alpha = (
        alpha
        if isinstance(router, SmartTokenRouter) and router.config.use_preference
        else None
    )
    routes = route_response_pair(
        frozen,
        router=router,
        router_alpha=router_alpha,
        static_scale=1.0,
    )
    return objective_for_routes(routes, frozen, weights), routes, weights


def _comparison_routers(
    config: ParmRouterTrainingConfig,
    own_model: Router,
    runtime: ParmTrainingRuntime,
) -> dict[str, Router]:
    output_parent = project_path(config.output_dir).parent
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    routers: dict[str, Router] = {}
    if config.stage == "taro":
        routers["taro"] = own_model
        return routers
    taro, _ = _load_router(
        output_parent / "taro" / "best.pt",
        kind="taro",
        device=runtime.device,
        tokenizer_hash=tokenizer_hash,
    )
    routers["taro"] = taro
    if config.stage == "v2_no_alpha":
        routers["v2_no_alpha"] = own_model
        return routers
    no_alpha, _ = _load_router(
        output_parent / "v2_no_alpha" / "best.pt",
        kind="smart",
        device=runtime.device,
        tokenizer_hash=tokenizer_hash,
    )
    routers["v2_no_alpha"] = no_alpha
    routers["v2_alpha"] = own_model
    return routers


def evaluate_comparisons(
    config: ParmRouterTrainingConfig,
    runtime: ParmTrainingRuntime,
    examples: list[MultiObjectiveExample],
    routers: Mapping[str, Router],
) -> dict[str, Any]:
    tasks = list(iter_validation_tasks(examples, config.validation_alpha_grid))
    if len(tasks) < 2:
        raise ValueError("Stage 9 validation requires at least two tasks")
    controls = build_alpha_controls(torch.stack([alpha for _, alpha in tasks]))
    method_names = ["static_parm", *routers]
    if "v2_alpha" in routers:
        method_names.extend(("v2_alpha_shuffled", "v2_alpha_constant"))
    accumulators = {name: MethodAccumulator() for name in method_names}
    skipped: list[str] = []
    near_ties = 0
    shuffled_delta_sum = 0.0
    constant_delta_sum = 0.0
    control_token_count = 0
    with torch.no_grad():
        for task_index, (example, alpha) in enumerate(tasks):
            alpha = alpha.to(runtime.device)
            weights = response_objective_weights(
                alpha,
                better_response_id=example.better_response_id,
                safer_response_id=example.safer_response_id,
            )
            near_ties += int(is_near_tie(weights, config.near_tie_threshold))
            try:
                frozen = frozen_response_pair(runtime, config, example, alpha)
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            routes_by_method: dict[str, tuple[ParmRoute, ParmRoute]] = {}
            routes_by_method["static_parm"] = route_response_pair(
                frozen,
                router=None,
                router_alpha=None,
                static_scale=config.static_reference_scale,
            )
            for name, router in routers.items():
                router_alpha = alpha if name == "v2_alpha" else None
                routes_by_method[name] = route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=router_alpha,
                    static_scale=config.static_reference_scale,
                )
            if "v2_alpha" in routers:
                alpha_router = routers["v2_alpha"]
                routes_by_method["v2_alpha_shuffled"] = route_response_pair(
                    frozen,
                    router=alpha_router,
                    router_alpha=controls.shuffled[task_index].to(runtime.device),
                    static_scale=config.static_reference_scale,
                )
                routes_by_method["v2_alpha_constant"] = route_response_pair(
                    frozen,
                    router=alpha_router,
                    router_alpha=controls.constant[task_index].to(runtime.device),
                    static_scale=config.static_reference_scale,
                )
                correct_routes = routes_by_method["v2_alpha"]
                for response_index in (0, 1):
                    correct_lambda = correct_routes[response_index].lambda_t.detach()
                    shuffled_lambda = routes_by_method[
                        "v2_alpha_shuffled"
                    ][response_index].lambda_t.detach()
                    constant_lambda = routes_by_method[
                        "v2_alpha_constant"
                    ][response_index].lambda_t.detach()
                    shuffled_delta_sum += float(
                        (correct_lambda - shuffled_lambda).abs().sum()
                    )
                    constant_delta_sum += float(
                        (correct_lambda - constant_lambda).abs().sum()
                    )
                    control_token_count += correct_lambda.numel()
            for name, routes in routes_by_method.items():
                objective = objective_for_routes(routes, frozen, weights)
                accumulators[name].update(objective, routes, alpha)
    metrics = {name: value.finalize() for name, value in accumulators.items()}
    checks = {
        "required_methods_present": set(method_names) == set(metrics),
        "all_metrics_finite": all(value["finite"] for value in metrics.values()),
        "validation_not_test": True,
    }
    control_report = None
    if "v2_alpha" in routers:
        if control_token_count <= 0:
            raise ValueError("No alpha control tokens were evaluated")
        shuffled_delta = shuffled_delta_sum / control_token_count
        constant_delta = constant_delta_sum / control_token_count
        shuffled_loss_delta = (
            metrics["v2_alpha_shuffled"]["dual_response_token_nll"]
            - metrics["v2_alpha"]["dual_response_token_nll"]
        )
        control_checks = {
            "shuffled_changes_lambda": (
                shuffled_delta >= config.min_control_lambda_delta
            ),
            "constant_changes_lambda": (
                constant_delta >= config.min_control_lambda_delta
            ),
            "shuffled_degrades_loss": (
                shuffled_loss_delta > 0.0
                if config.require_shuffled_loss_degradation
                else True
            ),
            "guide_alpha_held_correct_for_all_router_controls": True,
        }
        control_report = {
            "mean_abs_lambda_delta_correct_vs_shuffled": shuffled_delta,
            "mean_abs_lambda_delta_correct_vs_constant": constant_delta,
            "shuffled_minus_correct_loss": shuffled_loss_delta,
            "checks": control_checks,
            "pass": all(control_checks.values()),
        }
        checks["preference_usage_controls_pass"] = control_report["pass"]
    return {
        "split": "validation",
        "tasks_requested": len(tasks),
        "tasks_skipped_too_long": len(skipped),
        "skipped_sample_ids": sorted(set(skipped)),
        "near_tie_tasks": near_ties,
        "alpha_grid_helpfulness": list(config.validation_alpha_grid),
        "methods": metrics,
        "alpha_controls": control_report,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _frozen_snapshot(
    config: ParmRouterTrainingConfig,
    runtime: ParmTrainingRuntime,
) -> dict[str, Any]:
    data_manifest = runtime.paths["data_root"] / "manifest.json"
    return {
        "base_model": checkpoint_descriptor(runtime.paths["base_model"]),
        "guide_base_model": checkpoint_descriptor(
            runtime.paths["guide_base_model"]
        ),
        "parm_adapter": checkpoint_descriptor(runtime.paths["parm_adapter"]),
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "data_manifest_sha256": sha256_file(data_manifest),
        "training_config": config.to_dict(),
    }


def _validate_stage_order(config: ParmRouterTrainingConfig, output_dir: Path) -> None:
    predecessor = config.predecessor_stage
    if predecessor is None or not config.enforce_stage_order:
        return
    status_path = output_dir.parent / predecessor / "run_status.json"
    if not status_path.is_file():
        raise ValueError(f"Missing predecessor status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        raise ValueError(f"Predecessor stage is not PASS: {status_path}")


def _checkpoint_state(
    *,
    epoch: int,
    next_example_index: int,
    optimizer_step: int,
    best_validation_loss: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "next_example_index": next_example_index,
        "optimizer_step": optimizer_step,
        "best_validation_loss": best_validation_loss,
        "optimizer_state_dict": optimizer.state_dict(),
    }


def train_stage(
    config: ParmRouterTrainingConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    prerequisite = audit_training_prerequisites(config)
    if not prerequisite["ready"]:
        failed = [name for name, passed in prerequisite["checks"].items() if not passed]
        raise RuntimeError(f"Stage 9 prerequisites are not ready: {failed}")
    paths = resolved_paths(config)
    output_dir = paths["output_dir"]
    latest_path = output_dir / "latest.pt"
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Missing resume checkpoint: {latest_path}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Stage 9 run: {output_dir}")
    _validate_stage_order(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    runtime = load_training_runtime(config)
    train_examples = load_examples(
        paths["data_root"], "train", max_samples=config.max_train_samples
    )
    validation_examples = load_examples(
        paths["data_root"],
        "validation",
        max_samples=config.max_validation_samples,
    )
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    metadata = {
        "schema_version": 1,
        "task": "pku_safe_rlhf_multi_objective",
        "stage": config.stage,
        "training_config": config.to_dict(),
        "tokenizer_semantic_sha256": tokenizer_hash,
        "alpha_order": ["helpfulness", "harmlessness"],
        "parm_pblora_order": ["harmlessness", "helpfulness"],
        "objective": config.scalarization_rule,
        "full_vocabulary_guidance": True,
        "base_parm_pblora_frozen": True,
        "data_manifest_sha256": sha256_file(paths["data_root"] / "manifest.json"),
    }
    frozen_before = _frozen_snapshot(config, runtime)
    write_json_atomic(
        output_dir / "frozen_audit_before.json",
        frozen_before,
        overwrite=resume,
    )
    write_json_atomic(
        output_dir / "resolved_training_config.json",
        metadata,
        overwrite=resume,
    )
    if resume:
        router, payload = _load_router(
            latest_path,
            kind=config.router_kind,
            device=runtime.device,
            tokenizer_hash=tokenizer_hash,
        )
        if payload["metadata"].get("training_config") != config.to_dict():
            raise ValueError("Resume config does not match checkpoint")
        state = payload["training_state"]
        start_epoch = int(state["epoch"])
        next_example_index = int(state["next_example_index"])
        optimizer_step = int(state["optimizer_step"])
        best_validation_loss = float(state["best_validation_loss"])
        for parameter in router.parameters():
            parameter.requires_grad_(True)
    else:
        router = build_router(
            config,
            vocab_size=runtime.alignment.base_vocab_size,
            device=runtime.device,
        )
        start_epoch = 0
        next_example_index = 0
        optimizer_step = 0
        best_validation_loss = float("inf")
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if resume:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    assert_optimizer_contains_only(optimizer, router.parameters())
    _assert_optimizer_fp32(optimizer)

    final_comparison = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.num_epochs):
        router.train()
        order = deterministic_example_order(
            train_examples, epoch=epoch, seed=config.seed
        )
        pending = 0
        loss_sum = 0.0
        trained = 0
        skipped: list[str] = []
        near_ties = 0
        lambda_values: list[torch.Tensor] = []
        for example_index, example in enumerate(order):
            if epoch == start_epoch and example_index < next_example_index:
                continue
            alpha = sample_alpha(
                example.sample_id, epoch=epoch, seed=config.seed
            ).to(runtime.device)
            try:
                frozen = frozen_response_pair(runtime, config, example, alpha)
            except SequenceTooLongError:
                skipped.append(example.sample_id)
                continue
            objective, routes, weights = training_example_objective(
                router, frozen, example, alpha
            )
            objective.total_loss.backward()
            if trained == 0:
                _assert_no_frozen_gradients(runtime)
            pending += 1
            trained += 1
            loss_sum += float(objective.total_loss.detach())
            near_ties += int(is_near_tie(weights, config.near_tie_threshold))
            lambda_values.extend(
                route.lambda_t.detach().cpu().reshape(-1) for route in routes
            )
            if pending == config.gradient_accumulation_steps:
                for parameter in router.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_step += 1
                if optimizer_step % config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "stage9_train_progress",
                                "stage": config.stage,
                                "epoch": epoch,
                                "example_index": example_index,
                                "optimizer_step": optimizer_step,
                                "running_loss": loss_sum / trained,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if optimizer_step % config.checkpoint_every_optimizer_steps == 0:
                    _save_router(
                        router,
                        latest_path,
                        training_state=_checkpoint_state(
                            epoch=epoch,
                            next_example_index=example_index + 1,
                            optimizer_step=optimizer_step,
                            best_validation_loss=best_validation_loss,
                            optimizer=optimizer,
                        ),
                        metadata=metadata,
                        overwrite=latest_path.exists(),
                    )
        if pending:
            for parameter in router.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        if trained == 0 or not lambda_values:
            raise ValueError("No Stage 9 training examples were usable")
        router.eval()
        comparison_routers = _comparison_routers(config, router, runtime)
        final_comparison = evaluate_comparisons(
            config, runtime, validation_examples, comparison_routers
        )
        own_metrics = final_comparison["methods"][_router_name(config)]
        validation_loss = float(own_metrics["dual_response_token_nll"])
        values = torch.cat(lambda_values)
        epoch_report = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train": {
                "examples": trained,
                "skipped_too_long": len(skipped),
                "skipped_sample_ids": skipped,
                "near_ties": near_ties,
                "mean_loss": loss_sum / trained,
                "lambda_mean": float(values.mean()),
                "lambda_std": float(values.std(unbiased=False)),
            },
            "validation_comparison": final_comparison,
        }
        epoch_path = output_dir / f"epoch_{epoch:03d}.json"
        write_json_atomic(
            epoch_path,
            epoch_report,
            overwrite=resume and epoch_path.exists(),
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            _save_router(
                router,
                output_dir / "best.pt",
                training_state=_checkpoint_state(
                    epoch=epoch + 1,
                    next_example_index=0,
                    optimizer_step=optimizer_step,
                    best_validation_loss=best_validation_loss,
                    optimizer=optimizer,
                ),
                metadata=metadata,
                overwrite=(output_dir / "best.pt").exists(),
            )
        _save_router(
            router,
            latest_path,
            training_state=_checkpoint_state(
                epoch=epoch + 1,
                next_example_index=0,
                optimizer_step=optimizer_step,
                best_validation_loss=best_validation_loss,
                optimizer=optimizer,
            ),
            metadata=metadata,
            overwrite=latest_path.exists(),
        )
        next_example_index = 0
    if final_comparison is None:
        raise ValueError("No Stage 9 epoch was executed")
    frozen_after = _frozen_snapshot(config, runtime)
    _assert_no_frozen_gradients(runtime)
    frozen_unchanged = frozen_before == frozen_after
    frozen_audit = {
        "before": frozen_before,
        "after": frozen_after,
        "unchanged": frozen_unchanged,
    }
    write_json_atomic(output_dir / "frozen_audit_after.json", frozen_audit)
    own = final_comparison["methods"][_router_name(config)]
    checks = {
        "comparison_pass": bool(final_comparison["pass"]),
        "own_metrics_finite": bool(own["finite"]),
        "router_lambda_nonconstant": float(own["lambda"]["std"]) > 1e-6,
        "frozen_artifacts_unchanged": frozen_unchanged,
        "optimizer_router_only": True,
        "optimizer_parameters_fp32": True,
        "frozen_model_gradients_none": True,
        "test_split_unused": True,
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    run_status = {
        "schema_version": 1,
        "stage": config.stage,
        "status": status,
        "checks": checks,
        "best_validation_loss": best_validation_loss,
        "comparison": final_comparison,
        "best_checkpoint": str(output_dir / "best.pt"),
        "frozen_audit": frozen_audit,
    }
    write_json_atomic(output_dir / "run_status.json", run_status)
    return run_status
