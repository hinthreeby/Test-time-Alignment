"""Pure metrics for diagnosing preference-insensitive lambda collapse."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from PARM_TARO.training.objective import dual_response_objective
from PARM_TARO.training.online import FrozenSequenceDistributions
from PARM_TARO.training.routing import ParmRoute, static_parm_route
from router_v2.candidates import select_taro_topk
from router_v2.smart_model import SmartRouterBatch, SmartTokenRouter


FIXED_LAMBDAS = (
    0.0,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)


@dataclass(frozen=True)
class FixedLambdaTaskMetrics:
    dual_response_nll: float
    response_mean_log_likelihoods: tuple[float, float]
    response_weight_difference: float
    weighted_preference_margin: float
    preferred_vs_nonpreferred_margin: float | None


def response_mean_log_likelihood(
    route: ParmRoute,
    frozen: FrozenSequenceDistributions,
) -> torch.Tensor:
    gold = frozen.gold_token_ids.to(route.guided_logprobs.device)
    selected = route.guided_logprobs[0].gather(
        -1, gold.unsqueeze(-1)
    ).squeeze(-1)
    return selected.mean()


def fixed_lambda_task_metrics(
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    response_weights: torch.Tensor,
    scale: float,
) -> FixedLambdaTaskMetrics:
    routes = tuple(static_parm_route(item, scale) for item in frozen)
    objective = dual_response_objective(
        (routes[0].guided_logprobs, routes[1].guided_logprobs),
        (frozen[0].gold_token_ids, frozen[1].gold_token_ids),
        response_weights,
    )
    scores = torch.stack(
        [
            response_mean_log_likelihood(route, item)
            for route, item in zip(routes, frozen)
        ]
    )
    weight_difference = response_weights[0] - response_weights[1]
    score_difference = scores[0] - scores[1]
    weighted_margin = weight_difference * score_difference
    preferred_margin = None
    if float(weight_difference.abs()) > 1e-8:
        preferred_margin = float(
            torch.sign(weight_difference) * score_difference
        )
    return FixedLambdaTaskMetrics(
        dual_response_nll=float(objective.total_loss.detach()),
        response_mean_log_likelihoods=(float(scores[0]), float(scores[1])),
        response_weight_difference=float(weight_difference),
        weighted_preference_margin=float(weighted_margin),
        preferred_vs_nonpreferred_margin=preferred_margin,
    )


def endpoint_distribution_sensitivity(
    left: FrozenSequenceDistributions,
    right: FrozenSequenceDistributions,
) -> dict[str, float]:
    if left.guide_logprobs.shape != right.guide_logprobs.shape:
        raise ValueError("Endpoint guide distributions must have matching shapes")
    left_log = left.guide_logprobs.detach().float()
    right_log = right.guide_logprobs.detach().float()
    midpoint_log = torch.logaddexp(left_log, right_log) - math.log(2.0)
    js = 0.5 * (
        (left_log.exp() * (left_log - midpoint_log)).sum(dim=-1)
        + (right_log.exp() * (right_log - midpoint_log)).sum(dim=-1)
    )
    gold = left.gold_token_ids.to(left_log.device)
    left_gold = left_log[0].gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    right_gold = right_log[0].gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    return {
        "mean_abs_logprob_delta": float((left_log - right_log).abs().mean()),
        "max_abs_logprob_delta": float((left_log - right_log).abs().max()),
        "mean_js_divergence": float(js.mean()),
        "mean_abs_gold_logprob_delta": float(
            (left_gold - right_gold).abs().mean()
        ),
    }


def smart_router_batch(
    router: SmartTokenRouter,
    frozen: FrozenSequenceDistributions,
    alpha: torch.Tensor,
) -> SmartRouterBatch:
    topk = select_taro_topk(
        frozen.base_logprobs.detach(),
        frozen.guide_logprobs.detach(),
        top_k=router.config.top_k,
    )
    return SmartRouterBatch(
        topk=topk,
        position=frozen.position if router.config.use_position else None,
        selected_score=(
            frozen.base_selected_logprobs
            if router.config.use_history
            else None
        ),
        preference=(
            alpha.to(frozen.base_logprobs.device).unsqueeze(0)
            if router.config.use_preference
            else None
        ),
    )


def finite_difference_lambda_alpha(
    router: SmartTokenRouter,
    frozen: FrozenSequenceDistributions,
    helpfulness: float,
    *,
    epsilon: float = 0.01,
) -> dict[str, float]:
    if not router.config.use_preference:
        raise ValueError("Alpha derivative requires a preference-aware router")
    if not 0.0 <= helpfulness <= 1.0 or not 0.0 < epsilon < 0.5:
        raise ValueError("Invalid finite-difference inputs")
    lower = max(0.0, helpfulness - epsilon)
    upper = min(1.0, helpfulness + epsilon)
    if upper <= lower:
        raise ValueError("Finite-difference interval is empty")
    device = frozen.base_logprobs.device
    lower_alpha = torch.tensor([lower, 1.0 - lower], device=device)
    upper_alpha = torch.tensor([upper, 1.0 - upper], device=device)
    with torch.no_grad():
        lower_lambda = router.predict_lambda(
            smart_router_batch(router, frozen, lower_alpha)
        ).lambda_t
        upper_lambda = router.predict_lambda(
            smart_router_batch(router, frozen, upper_alpha)
        ).lambda_t
    derivative = (upper_lambda - lower_lambda) / (upper - lower)
    return {
        "mean_derivative": float(derivative.mean()),
        "mean_abs_derivative": float(derivative.abs().mean()),
        "max_abs_derivative": float(derivative.abs().max()),
    }


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize empty values")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }
