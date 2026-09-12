"""Losses and lambda mappings used by the PARM-TARO recovery experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ResidualLambdaConfig:
    """A bounded residual scale centred on the trusted static PARM policy."""

    lambda_0: float = 1.0
    rho: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_0) or self.lambda_0 <= 0.0:
            raise ValueError("lambda_0 must be finite and positive")
        if not math.isfinite(self.rho) or not 0.0 < self.rho < 1.0:
            raise ValueError("rho must be in (0, 1)")

    @property
    def lambda_min(self) -> float:
        return self.lambda_0 * (1.0 - self.rho)

    @property
    def lambda_max(self) -> float:
        return self.lambda_0 * (1.0 + self.rho)


def residual_lambda(raw_output: torch.Tensor, config: ResidualLambdaConfig) -> torch.Tensor:
    """Map an unconstrained residual to ``lambda_0 * (1 + rho*tanh(u))``.

    A zero-initialized output head is therefore exactly static PARM at
    initialization.  The range is selected on validation, never on test.
    """

    if not bool(torch.isfinite(raw_output).all()):
        raise FloatingPointError("raw router output must be finite")
    return config.lambda_0 * (1.0 + config.rho * torch.tanh(raw_output))


def parm_preservation_kl(
    static_logprobs: torch.Tensor,
    adaptive_logprobs: torch.Tensor,
    *,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Forward KL ``KL(PARM-static || adaptive)`` for a trust region."""

    if static_logprobs.shape != adaptive_logprobs.shape:
        raise ValueError("static and adaptive log-probabilities must match")
    if not bool(torch.isfinite(static_logprobs).all()) or not bool(
        torch.isfinite(adaptive_logprobs).all()
    ):
        raise FloatingPointError("trust-region inputs must be finite")
    return F.kl_div(
        adaptive_logprobs,
        static_logprobs.exp(),
        reduction=reduction,
        log_target=False,
    )

