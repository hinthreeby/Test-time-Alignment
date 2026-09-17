"""Shape and validation utilities for Adaptive-PARM fusion."""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from torch import Tensor


def validate_log_policy_pair(base_logprobs: Tensor, parm_logprobs: Tensor) -> None:
    """Fail closed on inputs that could alter fusion semantics silently."""
    if not isinstance(base_logprobs, Tensor) or not isinstance(parm_logprobs, Tensor):
        raise TypeError("base_logprobs and parm_logprobs must be torch.Tensor values")
    if base_logprobs.shape != parm_logprobs.shape:
        raise ValueError(
            "Base and PARM tensors must have identical shapes; got "
            f"{tuple(base_logprobs.shape)} and {tuple(parm_logprobs.shape)}"
        )
    if base_logprobs.ndim < 1 or base_logprobs.shape[-1] < 1:
        raise ValueError("The final tensor dimension must be a non-empty vocabulary")
    if not base_logprobs.is_floating_point() or not parm_logprobs.is_floating_point():
        raise TypeError("Base and PARM log policies must use floating-point dtypes")
    if base_logprobs.dtype != parm_logprobs.dtype:
        raise TypeError("Base and PARM log policies must have the same dtype")
    if base_logprobs.device != parm_logprobs.device:
        raise ValueError("Base and PARM log policies must be on the same device")
    if not bool(torch.isfinite(base_logprobs).all()):
        raise ValueError("base_logprobs contains NaN or Inf")
    if not bool(torch.isfinite(parm_logprobs).all()):
        raise ValueError("parm_logprobs contains NaN or Inf")


def broadcast_weight(weight: Any, reference: Tensor) -> Tensor:
    """Return a vocabulary-invariant weight broadcastable to ``reference``.

    Supported forms are scalar, per-batch ``[B]``, prefix/token-level
    ``[B, ...]``, and an already expanded ``[B, ..., 1]`` tensor. A controller
    cannot provide a different trust value for individual vocabulary entries.
    """
    if isinstance(weight, Real):
        result = torch.as_tensor(weight, dtype=reference.dtype, device=reference.device)
    elif isinstance(weight, Tensor):
        if not weight.is_floating_point():
            weight = weight.float()
        result = weight.to(device=reference.device, dtype=reference.dtype)
    else:
        raise TypeError("weight must be a real scalar or floating-point torch.Tensor")

    prefix_shape = reference.shape[:-1]
    if result.ndim == 0:
        pass
    elif result.ndim == reference.ndim:
        if result.shape[-1] != 1:
            raise ValueError("weight must be vocabulary-invariant (last dimension must be 1)")
    elif result.ndim <= len(prefix_shape):
        for index, size in enumerate(result.shape):
            if size not in (1, prefix_shape[index]):
                raise ValueError(
                    f"weight shape {tuple(result.shape)} does not match log-policy "
                    f"prefix shape {tuple(prefix_shape)}"
                )
        result = result.reshape(*result.shape, *((1,) * (reference.ndim - result.ndim)))
    else:
        raise ValueError(
            f"weight rank {result.ndim} cannot broadcast to log-policy rank {reference.ndim}"
        )

    if not bool(torch.isfinite(result).all()):
        raise ValueError("weight contains NaN or Inf")
    if bool((result < 0).any()) or bool((result > 1).any()):
        raise ValueError("Adaptive-PARM trust weight must lie in [0, 1]")
    return result
