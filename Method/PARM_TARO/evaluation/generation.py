"""Frozen shared-backbone generation for all seven Stage 10 methods."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.adapters.router_adapter import RouterSequenceState
from PARM_TARO.decoding.adaptive import ParmTaroDecoder
from PARM_TARO.evaluation.config import ParmTaroEvaluationConfig
from PARM_TARO.training.alpha_residual_router import load_alpha_residual_checkpoint
from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.checkpoint import load_taro_checkpoint
from router_v2.model import TAROTokenRouter
from router_v2.smart_checkpoint import load_smart_checkpoint
from router_v2.smart_model import SmartRouterBatch, SmartTokenRouter


class FixedLambdaProvider:
    def __init__(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Fixed Stage 10 lambda must be in [0, 1]")
        self.value = float(value)

    def predict(
        self,
        base_router_values: torch.Tensor,
        guide_router_values: torch.Tensor,
        *,
        position: int,
        preference: torch.Tensor,
        state: RouterSequenceState,
    ) -> torch.Tensor:
        del guide_router_values, position, preference, state
        return torch.full(
            base_router_values.shape[:-1] + (1,),
            self.value,
            dtype=base_router_values.dtype,
            device=base_router_values.device,
        )

    @staticmethod
    def observe_selected_base_score(state: RouterSequenceState, score: float) -> None:
        del state, score


class CheckpointLambdaProvider:
    """Inference-only adapter for TARO, Smart V2, and V3 residual routers."""

    def __init__(self, router: TAROTokenRouter | SmartTokenRouter) -> None:
        self.router = router.eval()
        for parameter in router.parameters():
            parameter.requires_grad_(False)

    def predict(
        self,
        base_router_values: torch.Tensor,
        guide_router_values: torch.Tensor,
        *,
        position: int,
        preference: torch.Tensor,
        state: RouterSequenceState,
    ) -> torch.Tensor:
        topk = select_taro_topk(
            base_router_values.detach(),
            guide_router_values.detach(),
            top_k=self.router.config.top_k,
        )
        if isinstance(self.router, TAROTokenRouter):
            return self.router.predict_alpha(topk).detach()
        router_preference = preference if self.router.config.use_preference else None
        if self.router.config.use_history:
            state.topk_history.append(topk)
            history = TAROTopKBatch(
                base_token_ids=torch.stack([item.base_token_ids for item in state.topk_history], dim=1),
                base_logits=torch.stack([item.base_logits for item in state.topk_history], dim=1),
                reward_token_ids=torch.stack([item.reward_token_ids for item in state.topk_history], dim=1),
                reward_logits=torch.stack([item.reward_logits for item in state.topk_history], dim=1),
            )
            selected_scores = torch.tensor(
                [state.selected_base_scores + [0.0]],
                dtype=history.base_logits.dtype,
                device=history.base_logits.device,
            )
            positions = torch.arange(
                len(state.topk_history), device=history.base_logits.device
            ).unsqueeze(0)
            output = self.router.predict_lambda(
                SmartRouterBatch(
                    topk=history,
                    position=positions if self.router.config.use_position else None,
                    selected_score=selected_scores,
                    preference=router_preference,
                )
            )
            return output.lambda_t[:, -1].detach()
        output = self.router.predict_lambda(
            SmartRouterBatch(
                topk=topk,
                position=(
                    torch.tensor([position], device=topk.base_logits.device)
                    if self.router.config.use_position
                    else None
                ),
                preference=router_preference,
            )
        )
        return output.lambda_t.detach()

    @staticmethod
    def observe_selected_base_score(state: RouterSequenceState, score: float) -> None:
        state.selected_base_scores.append(float(score))


def _validate_metadata(payload: dict[str, Any], tokenizer_hash: str, *, production: bool) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("tokenizer_semantic_sha256") != tokenizer_hash:
        raise ValueError("Stage 10 Router tokenizer provenance mismatch")
    expected_task = (
        "pku_safe_rlhf_multi_objective_alpha_preference_production"
        if production
        else "pku_safe_rlhf_multi_objective"
    )
    if metadata.get("task") != expected_task:
        raise ValueError("Stage 10 Router task provenance mismatch")
    if production and metadata.get("full_training_authorized") is not True:
        raise ValueError("Full-alpha checkpoint is not production-authorized")


def load_checkpoint_providers(
    config: ParmTaroEvaluationConfig,
    *,
    device: torch.device,
    tokenizer_hash: str,
) -> dict[str, Any]:
    taro, taro_payload = load_taro_checkpoint(config.taro_checkpoint, map_location=device)
    no_alpha, no_alpha_payload = load_smart_checkpoint(
        config.no_alpha_checkpoint, map_location=device
    )
    full_alpha, full_alpha_payload = load_alpha_residual_checkpoint(
        config.full_alpha_checkpoint, map_location=device
    )
    _validate_metadata(taro_payload, tokenizer_hash, production=False)
    _validate_metadata(no_alpha_payload, tokenizer_hash, production=False)
    _validate_metadata(full_alpha_payload, tokenizer_hash, production=True)
    if no_alpha.config.use_preference:
        raise ValueError("V2 no-alpha checkpoint unexpectedly enables preference input")
    if not full_alpha.config.use_preference or full_alpha.config.preference_dim != 2:
        raise ValueError("Production full-alpha checkpoint lacks two-dimensional alpha")
    return {
        "parm_static": FixedLambdaProvider(1.0),
        "parm_taro": CheckpointLambdaProvider(taro),
        "parm_v2_no_alpha": CheckpointLambdaProvider(no_alpha),
        "parm_v2_full_alpha": CheckpointLambdaProvider(full_alpha),
        "parm_v2_full_alpha_shuffled_alpha": CheckpointLambdaProvider(full_alpha),
        "parm_v2_full_alpha_fixed_alpha": CheckpointLambdaProvider(full_alpha),
    }


def make_decoder(runtime: Any, config: ParmTaroEvaluationConfig, provider: Any) -> ParmTaroDecoder:
    return ParmTaroDecoder(
        base_model=runtime.base_model,
        guide_model=runtime.guide_model,
        tokenizer=runtime.tokenizer,
        alignment=runtime.alignment,
        lambda_provider=provider,
        device=runtime.device,
        preference_dim=2,
        top_k=20,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        stop_on_eos=config.stop_on_eos,
    )


def generate_one(
    decoder: ParmTaroDecoder,
    *,
    prompt: str,
    requested_alpha: tuple[float, float],
    router_alpha: tuple[float, float] | None,
    seed: int,
) -> dict[str, Any]:
    preference = torch.tensor(requested_alpha, dtype=torch.float32, device=decoder.device)
    router_preference = (
        None
        if router_alpha is None
        else torch.tensor(router_alpha, dtype=torch.float32, device=decoder.device)
    )
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    started = time.perf_counter()
    output = decoder.generate(
        prompt,
        preference=preference,
        router_preference=router_preference,
        seed=seed,
    )
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    elapsed = time.perf_counter() - started
    return {**output.to_dict(), "latency_seconds": elapsed}
