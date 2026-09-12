"""Stage 5 training engine with frozen online logits and strict audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from router_v2.cache.features import compute_confidence_disagreement
from router_v2.cache.io import (
    require_path_within,
    sha256_file,
    write_json_atomic,
)
from router_v2.checkpoint import load_taro_checkpoint, save_taro_checkpoint
from router_v2.config import TARORouterConfig
from router_v2.device import resolve_device
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.provenance import tokenizer_descriptor
from router_v2.guide_model.runtime import (
    configure_deterministic_inference,
    load_compatible_tokenizer_pair,
    load_frozen_causal_model_pair,
)
from router_v2.model import TAROTokenRouter
from router_v2.smart_checkpoint import (
    load_smart_checkpoint,
    save_smart_checkpoint,
)
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import (
    SmartRouterBatch,
    SmartTokenRouter,
)
from router_v2.training.audit import (
    assert_frozen_models,
    assert_optimizer_contains_only,
    compare_frozen_snapshots,
    snapshot_frozen_artifacts,
)
from router_v2.training.config import RouterTrainingConfig
from router_v2.training.data import CachedRouterBatch, ShardedSequenceCache
from router_v2.training.diagnostics import LambdaDiagnostics
from router_v2.training.objective import (
    RouterTrainingObjective,
    compute_router_training_objective,
)
from router_v2.training.online import (
    OnlineLogitBatch,
    compute_online_logit_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "router_v2"


@dataclass(frozen=True)
class RouterForward:
    gate: torch.Tensor
    lambda_t: torch.Tensor
    guided_logits: torch.Tensor
    lambda_max: float


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _build_router(
    training_config: RouterTrainingConfig,
    cache: ShardedSequenceCache,
    device: torch.device,
) -> TAROTokenRouter | SmartTokenRouter:
    config_path = _project_path(training_config.router_config_path)
    if training_config.router_kind == "taro":
        config = TARORouterConfig.load_json(config_path)
        if config.vocab_size != cache.vocab_size or config.top_k != cache.top_k:
            raise ValueError("TARO config is incompatible with Stage 3 cache")
        if config.entropy_weight != training_config.entropy_weight:
            raise ValueError(
                "TARO router and training entropy weights must match"
            )
        return TAROTokenRouter(config).to(device)

    config = SmartRouterConfig.load_json(config_path)
    expected_variants = {
        "state": "v2_topk_confidence_position",
        "history": "v2_topk_state_history",
        "alpha": "v2_topk_state_history_alpha",
    }
    if config.variant != expected_variants[training_config.stage]:
        raise ValueError(
            f"Stage {training_config.stage!r} requires Smart variant "
            f"{expected_variants[training_config.stage]!r}"
        )
    if (
        config.vocab_size != cache.vocab_size
        or config.top_k != cache.top_k
        or config.max_position != cache.max_position
    ):
        raise ValueError("Smart Router config is incompatible with Stage 3 cache")
    return SmartTokenRouter(config).to(device)


def _router_forward(
    model: TAROTokenRouter | SmartTokenRouter,
    cached: CachedRouterBatch,
    online: OnlineLogitBatch,
) -> RouterForward:
    if isinstance(model, TAROTokenRouter):
        output = model(
            cached.topk,
            online.base_logits,
            online.guide_logits,
        )
        return RouterForward(
            gate=output.alpha,
            lambda_t=output.alpha,
            guided_logits=output.guided_logits,
            lambda_max=1.0,
        )
    output = model(
        SmartRouterBatch(
            topk=cached.topk,
            position=cached.position,
            selected_score=online.base_selected_logprob,
            preference=cached.preference,
        ),
        online.base_logits,
        online.guide_logits,
    )
    if output.guided_logits is None:
        raise RuntimeError("Smart Router did not return full guided logits")
    return RouterForward(
        gate=output.gate,
        lambda_t=output.lambda_t,
        guided_logits=output.guided_logits,
        lambda_max=model.config.lambda_max,
    )


def _objective(
    training_config: RouterTrainingConfig,
    forward: RouterForward,
    cached: CachedRouterBatch,
) -> RouterTrainingObjective:
    return compute_router_training_objective(
        forward.guided_logits,
        cached.gold_token_ids,
        cached.valid_mask,
        forward.gate,
        forward.lambda_t,
        lambda_max=forward.lambda_max,
        entropy_weight=training_config.entropy_weight,
        smoothness_weight=training_config.smoothness_weight,
        strength_weight=training_config.strength_weight,
    )


def _update_diagnostics(
    diagnostics: LambdaDiagnostics,
    objective: RouterTrainingObjective,
    forward: RouterForward,
    cached: CachedRouterBatch,
    *,
    max_position: int,
) -> None:
    derived = compute_confidence_disagreement(
        cached.topk.base_token_ids,
        cached.topk.base_logits,
        cached.topk.reward_token_ids,
        cached.topk.reward_logits,
        cached.position,
        max_position=max_position,
    )
    diagnostics.update(
        objective,
        forward.lambda_t,
        cached.valid_mask,
        cached.position,
        derived.base_entropy,
        derived.js_divergence,
        cached.preference,
    )


def _reference_metrics(
    logits: torch.Tensor,
    gold: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, int, int]:
    losses = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        gold.reshape(-1),
        reduction="none",
    ).reshape(gold.shape)
    count = int(mask.sum())
    nll_sum = float((losses * mask).sum())
    correct = int((logits.argmax(dim=-1).eq(gold) & mask).sum())
    return nll_sum, correct, count


def _save_checkpoint(
    model: TAROTokenRouter | SmartTokenRouter,
    path: Path,
    *,
    training_state: dict[str, Any],
    metadata: dict[str, Any],
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


def _load_checkpoint(
    kind: str,
    path: Path,
    device: torch.device,
) -> tuple[TAROTokenRouter | SmartTokenRouter, dict[str, Any]]:
    if kind == "taro":
        return load_taro_checkpoint(path, map_location=device)
    return load_smart_checkpoint(path, map_location=device)


def _optimizer_step(
    model: TAROTokenRouter | SmartTokenRouter,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
    gradient_scale: float = 1.0,
) -> None:
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _validate_stage_order(
    config: RouterTrainingConfig,
    output_dir: Path,
) -> None:
    predecessor = config.predecessor_stage
    if predecessor is None or not config.enforce_stage_order:
        return
    status_path = output_dir.parent / predecessor / "run_status.json"
    if not status_path.exists():
        raise ValueError(
            f"Stage {config.stage} requires predecessor report: {status_path}"
        )
    status = _load_json(status_path)
    if status.get("status") != "PASS":
        raise ValueError(
            f"Predecessor stage {predecessor} is not PASS: {status_path}"
        )


def _online_batch(
    base_model: Any,
    guide_model: Any,
    tokenizer: Any,
    cached_cpu: CachedRouterBatch,
    *,
    device: torch.device,
    guide_config: SentimentGuideConfig,
    max_continuation_tokens: int,
) -> OnlineLogitBatch:
    return compute_online_logit_batch(
        base_model,
        guide_model,
        tokenizer,
        cached_cpu,
        device=device,
        max_length=guide_config.max_length,
        max_continuation_tokens=max_continuation_tokens,
    )


def evaluate_router(
    model: TAROTokenRouter | SmartTokenRouter,
    training_config: RouterTrainingConfig,
    cache: ShardedSequenceCache,
    base_model: Any,
    guide_model: Any,
    tokenizer: Any,
    guide_config: SentimentGuideConfig,
    device: torch.device,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    model.eval()
    lambda_max = (
        1.0
        if isinstance(model, TAROTokenRouter)
        else model.config.lambda_max
    )
    diagnostics = LambdaDiagnostics(
        lambda_max=lambda_max,
        near_boundary_fraction=training_config.near_boundary_fraction,
        alpha_bins=training_config.alpha_bins,
    )
    base_nll_sum = 0.0
    base_correct = 0
    base_count = 0
    with torch.no_grad():
        for cached_cpu in cache.iter_batches(
            batch_size=training_config.batch_size,
            epoch=0,
            seed=training_config.seed,
            shuffle=False,
            max_samples=training_config.max_validation_samples,
        ):
            online = _online_batch(
                base_model,
                guide_model,
                tokenizer,
                cached_cpu,
                device=device,
                guide_config=guide_config,
                max_continuation_tokens=int(
                    cache.manifest["extraction_parameters"][
                        "max_continuation_tokens"
                    ]
                ),
            )
            cached = cached_cpu.to(device)
            forward = _router_forward(model, cached, online)
            objective = _objective(training_config, forward, cached)
            _update_diagnostics(
                diagnostics,
                objective,
                forward,
                cached,
                max_position=cache.max_position,
            )
            nll_sum, correct, count = _reference_metrics(
                online.base_logits,
                cached.gold_token_ids,
                cached.valid_mask,
            )
            base_nll_sum += nll_sum
            base_correct += correct
            base_count += count
    adaptive = diagnostics.finalize()
    base_metrics = {
        "lambda": 0.0,
        "token_count": base_count,
        "nll": base_nll_sum / base_count,
        "token_accuracy": base_correct / base_count,
    }
    return adaptive, float(adaptive["lambda"]["mean"]), base_metrics


def evaluate_fixed_lambda(
    fixed_lambda: float,
    lambda_max: float,
    training_config: RouterTrainingConfig,
    cache: ShardedSequenceCache,
    base_model: Any,
    guide_model: Any,
    tokenizer: Any,
    guide_config: SentimentGuideConfig,
    device: torch.device,
) -> dict[str, Any]:
    if not 0.0 <= fixed_lambda <= lambda_max:
        raise ValueError("Fixed lambda is outside the router range")
    nll_sum = 0.0
    correct_count = 0
    token_count = 0
    with torch.no_grad():
        for cached_cpu in cache.iter_batches(
            batch_size=training_config.batch_size,
            epoch=0,
            seed=training_config.seed,
            shuffle=False,
            max_samples=training_config.max_validation_samples,
        ):
            online = _online_batch(
                base_model,
                guide_model,
                tokenizer,
                cached_cpu,
                device=device,
                guide_config=guide_config,
                max_continuation_tokens=int(
                    cache.manifest["extraction_parameters"][
                        "max_continuation_tokens"
                    ]
                ),
            )
            cached = cached_cpu.to(device)
            guided = online.base_logits + fixed_lambda * (
                online.guide_logits - online.base_logits
            )
            batch_nll, batch_correct, batch_count = _reference_metrics(
                guided,
                cached.gold_token_ids,
                cached.valid_mask,
            )
            nll_sum += batch_nll
            correct_count += batch_correct
            token_count += batch_count
    return {
        "lambda": fixed_lambda,
        "token_count": token_count,
        "nll": nll_sum / token_count,
        "token_accuracy": correct_count / token_count,
    }


def train_router(
    training_config: RouterTrainingConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = require_path_within(
        _project_path(training_config.output_dir),
        RESULTS_ROOT,
        label="Stage 5 output directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_stage_order(training_config, output_dir)
    latest_path = output_dir / "latest.pt"
    if resume and not latest_path.exists():
        raise FileNotFoundError(f"Resume checkpoint is absent: {latest_path}")
    if not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Stage 5 output directory is not empty: {output_dir}"
        )

    cache_root = _project_path(training_config.cache_root)
    train_cache = ShardedSequenceCache(cache_root, "train")
    validation_cache = ShardedSequenceCache(cache_root, "validation")
    if (
        train_cache.top_k != validation_cache.top_k
        or train_cache.vocab_size != validation_cache.vocab_size
        or train_cache.max_position != validation_cache.max_position
    ):
        raise ValueError("Train and validation cache contracts differ")
    cache_has_preferences = train_cache.has_preferences()
    validation_has_preferences = validation_cache.has_preferences()
    if training_config.preference_source == "cache":
        if not cache_has_preferences or not validation_has_preferences:
            raise ValueError(
                "Alpha stage requires real train/validation preference vectors"
            )
    elif cache_has_preferences or validation_has_preferences:
        raise ValueError("Non-alpha stage must not consume preference cache")

    device = resolve_device(
        training_config.device,
        allow_cpu_fallback=training_config.allow_cpu_fallback,
    )
    runtime = configure_deterministic_inference(
        seed=training_config.seed,
        device=device,
    )
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    base_path = Path(train_cache.manifest["base_model"]["path"])
    guide_path = Path(train_cache.manifest["guide_model"]["path"])
    frozen_before = snapshot_frozen_artifacts(
        base_model_path=base_path,
        guide_model_path=guide_path,
        parm_path=PROJECT_ROOT / "PARM",
    )
    for cache in (train_cache, validation_cache):
        if (
            cache.manifest["base_model"]["checkpoint_sha256"]
            != frozen_before["base_model"]["checkpoint_sha256"]
            or cache.manifest["guide_model"]["checkpoint_sha256"]
            != frozen_before["guide_model"]["checkpoint_sha256"]
        ):
            raise ValueError(
                f"Frozen model hash does not match {cache.split} cache provenance"
            )
    write_json_atomic(output_dir / "frozen_audit_before.json", frozen_before)
    tokenizer_pair = load_compatible_tokenizer_pair(base_path, guide_path)
    base_tokenizer = tokenizer_descriptor(tokenizer_pair.base, base_path)
    guide_tokenizer = tokenizer_descriptor(tokenizer_pair.guide, guide_path)
    for cache in (train_cache, validation_cache):
        if (
            cache.manifest["base_tokenizer"]["semantic_sha256"]
            != base_tokenizer["semantic_sha256"]
            or cache.manifest["guide_tokenizer"]["semantic_sha256"]
            != guide_tokenizer["semantic_sha256"]
        ):
            raise ValueError(
                f"Tokenizer hash does not match {cache.split} cache provenance"
            )
    model_pair = load_frozen_causal_model_pair(
        base_path,
        guide_path,
        device=device,
        precision="fp32",
    )
    assert_frozen_models(model_pair.base, model_pair.guide)
    guide_config = SentimentGuideConfig.load_json(
        PROJECT_ROOT / "router_v2" / "configs" / "sentiment_guide.json"
    )

    if resume:
        model, payload = _load_checkpoint(
            training_config.router_kind,
            latest_path,
            device,
        )
        state = payload["training_state"]
        if payload["metadata"].get("training_config") != training_config.to_dict():
            raise ValueError("Resume training config does not match checkpoint")
        start_epoch = int(state["epoch"])
        next_batch_index = int(state["next_batch_index"])
        optimizer_step = int(state["optimizer_step"])
        best_validation_nll = float(state["best_validation_nll"])
        diagnostics = LambdaDiagnostics.from_state_dict(
            state["diagnostics_state"]
        )
    else:
        model = _build_router(training_config, train_cache, device)
        start_epoch = 0
        next_batch_index = 0
        optimizer_step = 0
        best_validation_nll = float("inf")
        diagnostics = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    if resume:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    assert_optimizer_contains_only(optimizer, model.parameters())
    run_metadata = {
        "schema_version": 1,
        "training_config": training_config.to_dict(),
        "runtime": runtime,
        "train_manifest": str(train_cache.manifest_path),
        "train_manifest_sha256": sha256_file(train_cache.manifest_path),
        "validation_manifest": str(validation_cache.manifest_path),
        "validation_manifest_sha256": sha256_file(
            validation_cache.manifest_path
        ),
        "full_vocabulary_logits": "online_transient_fp32",
        "router_topk_source": "stage3_fp32_cache",
        "selected_score_source": training_config.selected_score_source,
        "current_gold_router_feature": False,
        "selected_score_causally_shifted": True,
        "history_semantics": (
            "selected_score[t] affects lambda[t+1] and later only"
        ),
    }
    if not resume:
        write_json_atomic(
            output_dir / "resolved_training_config.json",
            run_metadata,
        )

    epoch_reports = []
    final_validation = None
    final_base = None
    final_same_average = None
    for epoch in range(start_epoch, training_config.num_epochs):
        model.train()
        if diagnostics is None or epoch != start_epoch or next_batch_index == 0:
            lambda_max = (
                1.0
                if isinstance(model, TAROTokenRouter)
                else model.config.lambda_max
            )
            diagnostics = LambdaDiagnostics(
                lambda_max=lambda_max,
                near_boundary_fraction=(
                    training_config.near_boundary_fraction
                ),
                alpha_bins=training_config.alpha_bins,
            )
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation = 0
        last_batch_index = -1
        for batch_index, cached_cpu in enumerate(
            train_cache.iter_batches(
                batch_size=training_config.batch_size,
                epoch=epoch,
                seed=training_config.seed,
                shuffle=True,
                max_samples=training_config.max_train_samples,
            )
        ):
            last_batch_index = batch_index
            if epoch == start_epoch and batch_index < next_batch_index:
                continue
            online = _online_batch(
                model_pair.base,
                model_pair.guide,
                tokenizer_pair.base,
                cached_cpu,
                device=device,
                guide_config=guide_config,
                max_continuation_tokens=int(
                    train_cache.manifest["extraction_parameters"][
                        "max_continuation_tokens"
                    ]
                ),
            )
            cached = cached_cpu.to(device)
            forward = _router_forward(model, cached, online)
            objective = _objective(training_config, forward, cached)
            (
                objective.total_loss
                / training_config.gradient_accumulation_steps
            ).backward()
            pending_accumulation += 1
            _update_diagnostics(
                diagnostics,
                objective,
                forward,
                cached,
                max_position=train_cache.max_position,
            )
            if (
                pending_accumulation
                == training_config.gradient_accumulation_steps
            ):
                _optimizer_step(
                    model,
                    optimizer,
                    max_grad_norm=training_config.max_grad_norm,
                )
                pending_accumulation = 0
                optimizer_step += 1
                if optimizer_step % training_config.log_every_optimizer_steps == 0:
                    print(
                        json.dumps(
                            {
                                "event": "train_progress",
                                "stage": training_config.stage,
                                "epoch": epoch,
                                "batch_index": batch_index,
                                "optimizer_step": optimizer_step,
                                "running_nll": (
                                    diagnostics.nll_sum
                                    / diagnostics.token_count
                                ),
                                "batch_lambda_mean": float(
                                    forward.lambda_t.detach()[
                                        ..., 0
                                    ][cached.valid_mask].mean()
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if (
                    optimizer_step
                    % training_config.checkpoint_every_optimizer_steps
                    == 0
                ):
                    _save_checkpoint(
                        model,
                        latest_path,
                        training_state={
                            "epoch": epoch,
                            "next_batch_index": batch_index + 1,
                            "optimizer_step": optimizer_step,
                            "best_validation_nll": best_validation_nll,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "diagnostics_state": diagnostics.state_dict(),
                        },
                        metadata=run_metadata,
                        overwrite=latest_path.exists(),
                    )
        if pending_accumulation:
            _optimizer_step(
                model,
                optimizer,
                max_grad_norm=training_config.max_grad_norm,
                gradient_scale=(
                    training_config.gradient_accumulation_steps
                    / pending_accumulation
                ),
            )
            optimizer_step += 1
        if last_batch_index < 0:
            raise ValueError("Training cache iterator produced no batches")
        train_summary = diagnostics.finalize()
        validation_summary, adaptive_mean, base_metrics = evaluate_router(
            model,
            training_config,
            validation_cache,
            model_pair.base,
            model_pair.guide,
            tokenizer_pair.base,
            guide_config,
            device,
        )
        lambda_max = (
            1.0
            if isinstance(model, TAROTokenRouter)
            else model.config.lambda_max
        )
        same_average = None
        if training_config.same_average_baseline:
            same_average = evaluate_fixed_lambda(
                adaptive_mean,
                lambda_max,
                training_config,
                validation_cache,
                model_pair.base,
                model_pair.guide,
                tokenizer_pair.base,
                guide_config,
                device,
            )
        epoch_report = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train": train_summary,
            "validation": validation_summary,
            "base_baseline": base_metrics,
            "same_average_lambda_baseline": same_average,
        }
        epoch_report_path = output_dir / f"epoch_{epoch:03d}.json"
        write_json_atomic(
            epoch_report_path,
            epoch_report,
            overwrite=epoch_report_path.exists(),
        )
        epoch_reports.append(epoch_report)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "stage": training_config.stage,
                    "epoch": epoch,
                    "train_nll": train_summary["nll"],
                    "validation_nll": validation_summary["nll"],
                    "base_nll": base_metrics["nll"],
                    "same_average_nll": (
                        same_average["nll"]
                        if same_average is not None
                        else None
                    ),
                    "lambda_mean": adaptive_mean,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if float(validation_summary["nll"]) < best_validation_nll:
            best_validation_nll = float(validation_summary["nll"])
            _save_checkpoint(
                model,
                output_dir / "best.pt",
                training_state={
                    "epoch": epoch + 1,
                    "next_batch_index": 0,
                    "optimizer_step": optimizer_step,
                    "best_validation_nll": best_validation_nll,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "diagnostics_state": diagnostics.state_dict(),
                },
                metadata=run_metadata,
                overwrite=(output_dir / "best.pt").exists(),
            )
        _save_checkpoint(
            model,
            latest_path,
            training_state={
                "epoch": epoch + 1,
                "next_batch_index": 0,
                "optimizer_step": optimizer_step,
                "best_validation_nll": best_validation_nll,
                "optimizer_state_dict": optimizer.state_dict(),
                "diagnostics_state": diagnostics.state_dict(),
            },
            metadata=run_metadata,
            overwrite=latest_path.exists(),
        )
        next_batch_index = 0
        final_validation = validation_summary
        final_base = base_metrics
        final_same_average = same_average

    if final_validation is None:
        raise ValueError("No training epoch was executed")
    frozen_after = snapshot_frozen_artifacts(
        base_model_path=base_path,
        guide_model_path=guide_path,
        parm_path=PROJECT_ROOT / "PARM",
    )
    frozen_audit = compare_frozen_snapshots(frozen_before, frozen_after)
    write_json_atomic(output_dir / "frozen_audit_after.json", frozen_audit)
    lambda_std = float(final_validation["lambda"]["std"])
    improves_base = float(final_validation["nll"]) < float(final_base["nll"])
    same_average_pass = (
        final_same_average is not None
        and float(final_validation["nll"]) < float(final_same_average["nll"])
    )
    checks = {
        "finite_validation": bool(final_validation["finite"]),
        "validation_nll_improves_over_base": improves_base,
        "lambda_not_constant": lambda_std > 1e-6,
        "same_average_control_available": final_same_average is not None,
        "adaptive_beats_same_average_fixed": same_average_pass,
        "frozen_artifacts_unchanged": bool(frozen_audit["pass"]),
    }
    status = "PASS" if all(checks.values()) else "NOT_PASS"
    run_status = {
        "schema_version": 1,
        "status": status,
        "stage": training_config.stage,
        "router_kind": training_config.router_kind,
        "checks": checks,
        "best_validation_nll": best_validation_nll,
        "final_validation": final_validation,
        "base_baseline": final_base,
        "same_average_lambda_baseline": final_same_average,
        "frozen_audit": frozen_audit,
        "epoch_reports": [
            str(output_dir / f"epoch_{item['epoch']:03d}.json")
            for item in epoch_reports
        ],
        "best_checkpoint": str(output_dir / "best.pt"),
        "latest_checkpoint": str(latest_path),
    }
    write_json_atomic(
        output_dir / "run_status.json",
        run_status,
        overwrite=(output_dir / "run_status.json").exists(),
    )
    return run_status
