"""Run a bounded CUDA load/forward/backward smoke for Stage 9."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from PARM_TARO.adapters.parm_adapter import (
    named_preference_to_parm,
    set_parm_preference,
)
from PARM_TARO.training.alpha import sample_alpha
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    MultiObjectiveExample,
    SequenceTooLongError,
    load_examples,
    tokenize_response,
)
from PARM_TARO.training.engine import (
    _assert_no_frozen_gradients,
    _assert_optimizer_fp32,
    build_router,
    frozen_response_pair,
    training_example_objective,
)
from PARM_TARO.training.runtime import (
    RESULTS_ROOT,
    AdapterLayerState,
    FrozenBaseModelView,
    ParmTrainingRuntime,
    audit_training_prerequisites,
    capture_adapter_layer_state,
    load_training_runtime,
)
from PARM_TARO.training.online import _selected_logits
from router_v2.cache.io import require_path_within, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/parm_taro/smoke/stage9_cuda_taro.json"


def _state_summary(states: tuple[AdapterLayerState, ...]) -> dict[str, object]:
    return {
        "layer_count": len(states),
        "disabled_count": sum(state.disabled for state in states),
        "enabled_count": sum(not state.disabled for state in states),
        "active_adapter_sets": sorted(
            {",".join(state.active_adapters) for state in states}
        ),
    }


def _audit_adapter_restoration(
    runtime: ParmTrainingRuntime,
    config: ParmRouterTrainingConfig,
    example: MultiObjectiveExample,
    alpha: torch.Tensor,
) -> dict[str, object]:
    if not isinstance(runtime.base_model, FrozenBaseModelView):
        raise TypeError("Restoration audit requires a shared FrozenBaseModelView")
    tokenized = tokenize_response(
        runtime.tokenizer,
        example,
        0,
        max_length=config.max_length,
        max_continuation_tokens=config.max_continuation_tokens,
        include_eos_target=config.include_eos_target,
    )
    set_parm_preference(
        runtime.guide_model,
        named_preference_to_parm(alpha),
    )
    state_before = capture_adapter_layer_state(runtime.guide_model)
    guide_before = _selected_logits(
        runtime.guide_model, tokenized, device=runtime.device
    )
    base = _selected_logits(runtime.base_model, tokenized, device=runtime.device)
    transition = runtime.base_model.last_adapter_transition
    if transition is None:
        raise RuntimeError("Base view did not record its adapter transition")
    guide_after = _selected_logits(
        runtime.guide_model, tokenized, device=runtime.device
    )
    state_after = capture_adapter_layer_state(runtime.guide_model)
    guide_difference = float((guide_before - guide_after).abs().max())
    base_guide_difference = float((base - guide_before).abs().max())
    return {
        "state_before": _state_summary(state_before),
        "state_inside_disable_context": _state_summary(transition.inside),
        "state_after": _state_summary(state_after),
        "state_after_equals_state_before": state_after == state_before,
        "transition_after_equals_before": transition.after == transition.before,
        "all_layers_disabled_inside": bool(transition.inside)
        and all(state.disabled for state in transition.inside),
        "requires_grad_restored": transition.requires_grad_restored,
        "guide_before_after_exact_match": torch.equal(
            guide_before, guide_after
        ),
        "guide_before_after_max_abs_difference": guide_difference,
        "base_guide_max_abs_logit_difference": base_guide_difference,
        "base_and_guide_differ": base_guide_difference > 1e-7,
    }


def run_smoke(
    config: ParmRouterTrainingConfig,
    *,
    max_samples: int,
) -> dict[str, object]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 9 CUDA smoke requires an exposed CUDA device")
    audit = audit_training_prerequisites(config)
    if not audit["ready"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"Stage 9 prerequisites are not ready: {failed}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    parm_before = checkpoint_descriptor(
        PROJECT_ROOT / config.parm_adapter_path
    )
    tree_before = hash_tree(PROJECT_ROOT / "PARM")
    runtime = load_training_runtime(config)
    if runtime.model_load_info.get("physical_backbone_count") != 1:
        raise RuntimeError("CUDA smoke requires exactly one physical backbone")
    if not runtime.model_load_info.get("load_in_4bit"):
        raise RuntimeError("CUDA smoke requires the shared backbone in 4-bit")

    router = build_router(
        config,
        vocab_size=runtime.alignment.base_vocab_size,
        device=runtime.device,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate)
    assert_optimizer_contains_only(optimizer, router.parameters())
    _assert_optimizer_fp32(optimizer)
    optimizer.zero_grad(set_to_none=True)

    examples = load_examples(
        runtime.paths["data_root"], "train", max_samples=max_samples
    )
    restoration_audit = _audit_adapter_restoration(
        runtime,
        config,
        examples[0],
        sample_alpha(examples[0].sample_id, epoch=0, seed=config.seed).to(
            runtime.device
        ),
    )
    losses = []
    distribution_max_abs_differences = []
    used_samples = []
    skipped_samples = []
    for example in examples:
        alpha = sample_alpha(example.sample_id, epoch=0, seed=config.seed).to(
            runtime.device
        )
        try:
            frozen = frozen_response_pair(runtime, config, example, alpha)
        except SequenceTooLongError:
            skipped_samples.append(example.sample_id)
            continue
        distribution_max_abs_differences.extend(
            float((item.base_logprobs - item.guide_logprobs).abs().max())
            for item in frozen
        )
        objective, _, _ = training_example_objective(
            router, frozen, example, alpha
        )
        objective.total_loss.backward()
        _assert_no_frozen_gradients(runtime)
        losses.append(float(objective.total_loss.detach()))
        used_samples.append(example.sample_id)
        del objective, frozen
    if not used_samples:
        raise RuntimeError("CUDA smoke found no usable Stage 9 samples")

    router_gradient_nonzero = any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in router.parameters()
    )
    parm_after = checkpoint_descriptor(runtime.paths["parm_adapter"])
    tree_after = hash_tree(PROJECT_ROOT / "PARM")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    checks = {
        "one_physical_backbone": (
            runtime.model_load_info.get("physical_backbone_count") == 1
        ),
        "backbone_loaded_in_4bit": bool(
            runtime.model_load_info.get("load_in_4bit")
        ),
        "base_and_parm_distributions_differ": all(
            value > 1e-7 for value in distribution_max_abs_differences
        ),
        "adapter_restored_after_base_forward": all(
            bool(restoration_audit[name])
            for name in (
                "state_after_equals_state_before",
                "transition_after_equals_before",
                "all_layers_disabled_inside",
                "requires_grad_restored",
                "guide_before_after_exact_match",
                "base_and_guide_differ",
            )
        ),
        "router_gradient_nonzero": router_gradient_nonzero,
        "frozen_model_gradients_none": True,
        "pblora_checkpoint_unchanged": parm_before == parm_after,
        "parm_tree_unchanged": tree_before == tree_after,
        "losses_finite": bool(torch.isfinite(torch.tensor(losses)).all()),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "NOT_PASS",
        "checks": checks,
        "model_load_info": runtime.model_load_info,
        "samples_requested": max_samples,
        "samples_used": used_samples,
        "samples_skipped": skipped_samples,
        "losses": losses,
        "base_parm_max_abs_logprob_difference": (
            distribution_max_abs_differences
        ),
        "adapter_restoration_audit": restoration_audit,
        "cuda": {
            "device": torch.cuda.get_device_name(runtime.device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_allocated / 1024**3,
            "peak_reserved_gib": peak_reserved / 1024**3,
        },
        "pblora_checkpoint": parm_after,
        "parm_tree": tree_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = replace(
        ParmRouterTrainingConfig.load_json(args.config),
        device="cuda",
        allow_cpu_fallback=False,
    )
    output = require_path_within(
        args.output,
        RESULTS_ROOT,
        label="Stage 9 CUDA smoke output",
    )
    report = run_smoke(config, max_samples=args.max_samples)
    write_json_atomic(output, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
