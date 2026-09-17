"""Canonical Adaptive-PARM fusion in log-policy space."""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from torch import Tensor

from .utils import broadcast_weight, validate_log_policy_pair


def adaptive_parm_logits(
    base_logprobs: Tensor,
    parm_logprobs: Tensor,
    weight: Any,
    *,
    validate: bool = True,
) -> Tensor:
    """Interpolate Base and PARM log policies with a trust weight.

    Implements exactly

    ``z = (1 - w) * log(p_base) + w * log(p_PARM)``.

    No temperature, normalization, clipping, or vocabulary-dependent weighting
    is applied. The returned tensor is newly allocated and inputs are untouched.
    Scalar, per-batch, and prefix/token-level weights are supported.
    """
    if validate:
        validate_log_policy_pair(base_logprobs, parm_logprobs)
    w = broadcast_weight(weight, base_logprobs)
    return (1.0 - w) * base_logprobs + w * parm_logprobs


def adaptive_parm_logprobs(
    base_logprobs: Tensor,
    parm_logprobs: Tensor,
    weight: Any,
    *,
    probability_dtype: torch.dtype | None = torch.float32,
    validate: bool = True,
) -> Tensor:
    """Return normalized log probabilities from canonical fused logits."""
    logits = adaptive_parm_logits(
        base_logprobs, parm_logprobs, weight, validate=validate
    )
    working = logits if probability_dtype is None else logits.to(probability_dtype)
    return torch.log_softmax(working, dim=-1)


def adaptive_parm_distribution(
    base_logprobs: Tensor,
    parm_logprobs: Tensor,
    weight: Any,
    *,
    probability_dtype: torch.dtype | None = torch.float32,
    validate: bool = True,
) -> Tensor:
    """Return the normalized Adaptive-PARM probability distribution."""
    return adaptive_parm_logprobs(
        base_logprobs,
        parm_logprobs,
        weight,
        probability_dtype=probability_dtype,
        validate=validate,
    ).exp()


def lambda_to_weight(value: Real | Tensor) -> Tensor:
    """Map legacy non-negative ``lambda`` to the equivalent trust weight.

    This helper exists only for equation audits. Adaptive-PARM controllers
    should predict ``w`` directly.
    """
    if isinstance(value, Real):
        lam = torch.as_tensor(value, dtype=torch.float64)
    elif isinstance(value, Tensor):
        if not value.is_floating_point():
            value = value.float()
        lam = value
    else:
        raise TypeError("lambda must be a real scalar or torch.Tensor")
    if not bool(torch.isfinite(lam).all()) or bool((lam < 0).any()):
        raise ValueError("lambda must be finite and non-negative")
    return lam / (1.0 + lam)
