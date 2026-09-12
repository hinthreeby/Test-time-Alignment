"""PARM-specific routing for training and static/comparator evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from PARM_TARO.training.alpha import validate_alpha
from PARM_TARO.training.online import FrozenSequenceDistributions
from router_v2.candidates import select_taro_topk
from router_v2.model import TAROTokenRouter
from router_v2.smart_model import SmartRouterBatch, SmartTokenRouter


@dataclass(frozen=True)
class ParmRoute:
    guided_logprobs: torch.Tensor
    gate: torch.Tensor
    lambda_t: torch.Tensor


def _guided_logprobs(
    frozen: FrozenSequenceDistributions,
    lambda_t: torch.Tensor,
) -> torch.Tensor:
    expected = frozen.base_logprobs.shape[:-1] + (1,)
    if lambda_t.shape != expected:
        raise ValueError(f"lambda_t must have shape {expected}")
    return F.log_softmax(
        frozen.base_logprobs.detach()
        + lambda_t * frozen.guide_logprobs.detach(),
        dim=-1,
    )


def static_parm_route(
    frozen: FrozenSequenceDistributions,
    scale: float,
) -> ParmRoute:
    if scale < 0.0:
        raise ValueError("Static PARM scale must be non-negative")
    shape = frozen.base_logprobs.shape[:-1] + (1,)
    lambda_t = torch.full(
        shape,
        float(scale),
        dtype=frozen.base_logprobs.dtype,
        device=frozen.base_logprobs.device,
    )
    return ParmRoute(
        guided_logprobs=_guided_logprobs(frozen, lambda_t),
        gate=lambda_t,
        lambda_t=lambda_t,
    )


def router_parm_route(
    router: TAROTokenRouter | SmartTokenRouter,
    frozen: FrozenSequenceDistributions,
    *,
    router_alpha: torch.Tensor | None,
) -> ParmRoute:
    topk = select_taro_topk(
        frozen.base_logprobs.detach(),
        frozen.guide_logprobs.detach(),
        top_k=router.config.top_k,
    )
    if isinstance(router, TAROTokenRouter):
        if router_alpha is not None:
            raise ValueError("TARO router does not accept preference alpha")
        lambda_t = router.predict_alpha(topk)
        gate = lambda_t
    else:
        if router.config.use_preference:
            if router_alpha is None:
                raise ValueError("Preference-aware router requires alpha")
            alpha = validate_alpha(router_alpha)
            if alpha.shape != (2,):
                raise ValueError("Router alpha must have shape [2]")
            preference = alpha.to(frozen.base_logprobs.device).unsqueeze(0)
        else:
            if router_alpha is not None:
                raise ValueError("No-alpha router must not receive alpha")
            preference = None
        output = router.predict_lambda(
            SmartRouterBatch(
                topk=topk,
                position=(
                    frozen.position
                    if router.config.use_position
                    else None
                ),
                selected_score=(
                    frozen.base_selected_logprobs
                    if router.config.use_history
                    else None
                ),
                preference=preference,
            )
        )
        lambda_t = output.lambda_t
        gate = output.gate
    return ParmRoute(
        guided_logprobs=_guided_logprobs(frozen, lambda_t),
        gate=gate,
        lambda_t=lambda_t,
    )
