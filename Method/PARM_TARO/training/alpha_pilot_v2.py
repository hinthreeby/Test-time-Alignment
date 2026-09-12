"""Measured diagnostics and objectives for the second alpha-aware pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from PARM_TARO.training.objective import (
    DualResponseObjective,
    dual_response_objective,
)
from PARM_TARO.training.online import FrozenSequenceDistributions
from PARM_TARO.training.routing import ParmRoute
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter


SCORE_SENSITIVITY_LAMBDAS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
INITIALIZATION_NAMES = (
    "collapsed_v2_alpha",
    "no_alpha_transplant_fresh_preference_head",
    "fresh_alpha_target_gate",
)


@dataclass(frozen=True)
class PreferenceGainObjective:
    loss: torch.Tensor
    quality_nll: DualResponseObjective
    guided_response_scores: torch.Tensor
    base_response_scores: torch.Tensor
    response_weight_difference: torch.Tensor
    weighted_guidance_gain: torch.Tensor


@dataclass(frozen=True)
class CurriculumObjective:
    total_loss: torch.Tensor
    preference_loss: torch.Tensor
    quality_nll: DualResponseObjective
    strength_loss: torch.Tensor
    sensitivity_loss: torch.Tensor
    mean_lambda: torch.Tensor
    endpoint_lambda_delta: torch.Tensor
    weighted_guidance_gain: torch.Tensor


def _mean_gold_log_likelihood(
    logprobs: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    selected = logprobs[0].gather(
        -1,
        gold.to(logprobs.device).unsqueeze(-1),
    ).squeeze(-1)
    return selected.mean()


def preference_gain_objective(
    routes: tuple[ParmRoute, ParmRoute],
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    response_weights: torch.Tensor,
    *,
    pairwise_logit_scale: float,
) -> PreferenceGainObjective:
    """Rank the alpha-preferred response by guidance gain over the base LM."""

    if pairwise_logit_scale <= 0.0:
        raise ValueError("pairwise_logit_scale must be positive")
    quality = dual_response_objective(
        (routes[0].guided_logprobs, routes[1].guided_logprobs),
        (frozen[0].gold_token_ids, frozen[1].gold_token_ids),
        response_weights,
    )
    guided_scores = -quality.response_nll
    base_scores = torch.stack(
        [
            _mean_gold_log_likelihood(item.base_logprobs, item.gold_token_ids)
            for item in frozen
        ]
    )
    weight_difference = quality.response_weights[0] - quality.response_weights[1]
    guided_margin = guided_scores[0] - guided_scores[1]
    base_margin = base_scores[0] - base_scores[1]
    weighted_gain = weight_difference * (guided_margin - base_margin)
    loss = F.softplus(-pairwise_logit_scale * weighted_gain)
    return PreferenceGainObjective(
        loss=loss,
        quality_nll=quality,
        guided_response_scores=guided_scores,
        base_response_scores=base_scores,
        response_weight_difference=weight_difference,
        weighted_guidance_gain=weighted_gain,
    )


def endpoint_sensitivity_loss(
    left: tuple[ParmRoute, ParmRoute],
    right: tuple[ParmRoute, ParmRoute],
    *,
    minimum_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if minimum_delta <= 0.0:
        raise ValueError("minimum_delta must be positive")
    differences = torch.cat(
        [
            (left[index].lambda_t - right[index].lambda_t).abs().reshape(-1)
            for index in (0, 1)
        ]
    )
    return F.relu(minimum_delta - differences).mean(), differences.mean()


def endpoint_logit_sensitivity_loss(
    left: tuple[ParmRoute, ParmRoute],
    right: tuple[ParmRoute, ParmRoute],
    *,
    minimum_logit_delta: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce endpoint separation before sigmoid compression."""

    if minimum_logit_delta <= 0.0 or not 0.0 < eps < 0.5:
        raise ValueError("Invalid pre-sigmoid sensitivity controls")
    lambda_differences = []
    logit_differences = []
    for index in (0, 1):
        left_lambda = left[index].lambda_t.clamp(eps, 1.0 - eps)
        right_lambda = right[index].lambda_t.clamp(eps, 1.0 - eps)
        lambda_differences.append((left_lambda - right_lambda).abs().reshape(-1))
        logit_differences.append(
            (torch.logit(left_lambda) - torch.logit(right_lambda))
            .abs()
            .reshape(-1)
        )
    lambda_delta = torch.cat(lambda_differences)
    logit_delta = torch.cat(logit_differences)
    return (
        F.relu(minimum_logit_delta - logit_delta).mean(),
        logit_delta.mean(),
        lambda_delta.mean(),
    )


def lambda_delta_to_logit_delta(
    *,
    reference_lambda: float,
    lambda_delta: float,
    eps: float = 1e-6,
) -> float:
    """Map a symmetric lambda separation target to pre-sigmoid logit space."""

    if not 0.0 < reference_lambda < 1.0:
        raise ValueError("reference_lambda must be in (0, 1)")
    if lambda_delta <= 0.0:
        raise ValueError("lambda_delta must be positive")
    lower = max(eps, reference_lambda - lambda_delta / 2.0)
    upper = min(1.0 - eps, reference_lambda + lambda_delta / 2.0)
    if upper <= lower:
        raise ValueError("Lambda sensitivity interval is empty")
    lower_logit = math.log(lower / (1.0 - lower))
    upper_logit = math.log(upper / (1.0 - upper))
    return upper_logit - lower_logit


def linear_lambda_floor_loss(
    routes: tuple[ParmRoute, ParmRoute],
    *,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if floor < 0.0:
        raise ValueError("lambda floor must be non-negative")
    values = torch.cat([route.lambda_t.reshape(-1) for route in routes])
    return F.relu(floor - values).mean(), values.mean()


def curriculum_objective(
    routes: tuple[ParmRoute, ParmRoute],
    endpoint_routes: tuple[
        tuple[ParmRoute, ParmRoute],
        tuple[ParmRoute, ParmRoute],
    ],
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    response_weights: torch.Tensor,
    *,
    pairwise_logit_scale: float,
    preference_weight: float,
    nll_weight: float,
    strength_weight: float,
    sensitivity_weight: float,
    lambda_floor: float,
    sensitivity_minimum_delta: float,
) -> CurriculumObjective:
    weights = (
        preference_weight,
        nll_weight,
        strength_weight,
        sensitivity_weight,
    )
    if any(value < 0.0 for value in weights):
        raise ValueError("Curriculum weights must be non-negative")
    if not any(value > 0.0 for value in weights):
        raise ValueError("Curriculum must enable at least one objective")
    preference = preference_gain_objective(
        routes,
        frozen,
        response_weights,
        pairwise_logit_scale=pairwise_logit_scale,
    )
    strength, mean_lambda = linear_lambda_floor_loss(routes, floor=lambda_floor)
    sensitivity, endpoint_delta = endpoint_sensitivity_loss(
        endpoint_routes[0],
        endpoint_routes[1],
        minimum_delta=sensitivity_minimum_delta,
    )
    total = (
        preference_weight * preference.loss
        + nll_weight * preference.quality_nll.total_loss
        + strength_weight * strength
        + sensitivity_weight * sensitivity
    )
    return CurriculumObjective(
        total_loss=total,
        preference_loss=preference.loss,
        quality_nll=preference.quality_nll,
        strength_loss=strength,
        sensitivity_loss=sensitivity,
        mean_lambda=mean_lambda,
        endpoint_lambda_delta=endpoint_delta,
        weighted_guidance_gain=preference.weighted_guidance_gain,
    )


def exact_score_margin_sensitivity(
    frozen: tuple[FrozenSequenceDistributions, FrozenSequenceDistributions],
    scale: float,
) -> dict[str, float]:
    """Compute exact d(S0-S1)/d lambda for one shared fixed lambda."""

    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    scores = []
    derivatives = []
    for item in frozen:
        logits = item.base_logprobs + scale * item.guide_logprobs
        probabilities = F.softmax(logits, dim=-1)
        gold = item.gold_token_ids.to(logits.device)
        guided = F.log_softmax(logits, dim=-1)
        selected_score = guided[0].gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        selected_guide = item.guide_logprobs[0].gather(
            -1, gold.unsqueeze(-1)
        ).squeeze(-1)
        expected_guide = (
            probabilities * item.guide_logprobs
        ).sum(dim=-1)[0]
        scores.append(selected_score.mean())
        derivatives.append((selected_guide - expected_guide).mean())
    return {
        "lambda": float(scale),
        "score_margin": float((scores[0] - scores[1]).detach()),
        "d_score_margin_d_lambda": float(
            (derivatives[0] - derivatives[1]).detach()
        ),
    }


def router_parameter_groups(
    router: SmartTokenRouter,
) -> dict[str, tuple[nn.Parameter, ...]]:
    groups: dict[str, list[nn.Parameter]] = {
        "preference_encoder": [],
        "final_lambda_head": [],
        "router_rest": [],
    }
    for name, parameter in router.named_parameters():
        if name.startswith("preference_encoder."):
            group = "preference_encoder"
        elif name.startswith("fusion_mlp.5."):
            group = "final_lambda_head"
        else:
            group = "router_rest"
        groups[group].append(parameter)
    return {name: tuple(parameters) for name, parameters in groups.items()}


def term_gradient_norms(
    term: torch.Tensor,
    groups: Mapping[str, Sequence[nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, float]:
    parameters = tuple(parameter for values in groups.values() for parameter in values)
    gradients = torch.autograd.grad(
        term,
        parameters,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    output: dict[str, float] = {}
    offset = 0
    for name, values in groups.items():
        squared = term.new_zeros((), dtype=torch.float32)
        for gradient in gradients[offset : offset + len(values)]:
            if gradient is not None:
                squared = squared + gradient.float().square().sum()
        output[name] = float(squared.sqrt())
        offset += len(values)
    output["all_router"] = math.sqrt(sum(value * value for value in output.values()))
    return output


def gate_parameterization_stats(routes: Iterable[ParmRoute]) -> dict[str, float]:
    gates = torch.cat([route.gate.detach().float().reshape(-1) for route in routes])
    eps = torch.finfo(gates.dtype).eps
    bounded = gates.clamp(eps, 1.0 - eps)
    logits = torch.logit(bounded)
    derivatives = bounded * (1.0 - bounded)
    return {
        "count": int(gates.numel()),
        "mean_lambda": float(gates.mean()),
        "mean_pre_sigmoid_logit": float(logits.mean()),
        "min_pre_sigmoid_logit": float(logits.min()),
        "max_pre_sigmoid_logit": float(logits.max()),
        "mean_sigmoid_derivative": float(derivatives.mean()),
        "min_sigmoid_derivative": float(derivatives.min()),
        "max_sigmoid_derivative": float(derivatives.max()),
    }


def reset_lambda_head(
    router: SmartTokenRouter,
    *,
    target_gate: float,
    weight_std: float,
) -> None:
    if not 0.0 < target_gate < 1.0 or weight_std <= 0.0:
        raise ValueError("Invalid lambda-head initialization")
    head = router.fusion_mlp[-1]
    if not isinstance(head, nn.Linear):
        raise TypeError("Smart Router final lambda head must be linear")
    nn.init.normal_(head.weight, mean=0.0, std=weight_std)
    nn.init.constant_(head.bias, math.log(target_gate / (1.0 - target_gate)))


def transplant_no_alpha_router(
    source: SmartTokenRouter,
    alpha_config: SmartRouterConfig,
    *,
    target_gate: float,
    head_weight_std: float,
) -> SmartTokenRouter:
    if source.config.use_preference or not alpha_config.use_preference:
        raise ValueError("Expected no-alpha source and alpha-aware destination")
    destination = SmartTokenRouter(alpha_config).to(next(source.parameters()).device)
    if destination.feature_slices["preference"].start != source.feature_dim:
        raise ValueError("Preference features must be appended after shared features")
    for name in (
        "token_embedding",
        "base_candidate_encoder",
        "guide_candidate_encoder",
        "history_gru",
    ):
        source_module = getattr(source, name)
        destination_module = getattr(destination, name)
        if source_module is not None and destination_module is not None:
            destination_module.load_state_dict(source_module.state_dict())
    source_norm = source.fusion_mlp[0]
    destination_norm = destination.fusion_mlp[0]
    if isinstance(source_norm, nn.LayerNorm) and isinstance(
        destination_norm, nn.LayerNorm
    ):
        with torch.no_grad():
            destination_norm.weight[: source.feature_dim].copy_(source_norm.weight)
            destination_norm.bias[: source.feature_dim].copy_(source_norm.bias)
    source_first = source.fusion_mlp[1]
    destination_first = destination.fusion_mlp[1]
    if not isinstance(source_first, nn.Linear) or not isinstance(
        destination_first, nn.Linear
    ):
        raise TypeError("Smart Router fusion input must be linear")
    with torch.no_grad():
        destination_first.weight[:, : source.feature_dim].copy_(source_first.weight)
        destination_first.bias.copy_(source_first.bias)
    destination.fusion_mlp[3].load_state_dict(source.fusion_mlp[3].state_dict())
    reset_lambda_head(
        destination,
        target_gate=target_gate,
        weight_std=head_weight_std,
    )
    return destination
