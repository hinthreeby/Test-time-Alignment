"""Deterministic simplex sampling and alpha-only controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch


CANONICAL_ALPHA_ORDER = ("helpfulness", "harmlessness")


def validate_alpha(alpha: torch.Tensor) -> torch.Tensor:
    values = alpha.float()
    if values.shape[-1:] != (2,):
        raise ValueError("alpha must have final dimension two")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("alpha must be finite")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("alpha values must be in [0, 1]")
    sums = values.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0):
        raise ValueError("alpha must lie on the two-objective simplex")
    return values


def sample_alpha(sample_id: str, *, epoch: int, seed: int) -> torch.Tensor:
    """Sample U(0,1) reproducibly, independent of batching/resume order."""

    if not sample_id or epoch < 0 or seed < 0:
        raise ValueError("Invalid deterministic alpha sampling key")
    digest = hashlib.sha256(
        f"{seed}\0{epoch}\0{sample_id}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    helpfulness = (integer + 0.5) / float(2**64)
    return torch.tensor(
        [helpfulness, 1.0 - helpfulness], dtype=torch.float32
    )


def response_objective_weights(
    alpha: torch.Tensor,
    *,
    better_response_id: int,
    safer_response_id: int,
) -> torch.Tensor:
    """Return q_alpha(response_0), q_alpha(response_1) without hard choice."""

    values = validate_alpha(alpha)
    if values.ndim != 1:
        raise ValueError("Response weights require one alpha vector")
    if better_response_id not in (0, 1) or safer_response_id not in (0, 1):
        raise ValueError("Preference labels must be response IDs 0 or 1")
    weights = torch.zeros(2, dtype=values.dtype, device=values.device)
    weights[better_response_id] += values[0]
    weights[safer_response_id] += values[1]
    if not torch.allclose(weights.sum(), torch.tensor(1.0, device=values.device)):
        raise RuntimeError("Dual-response scalarization weights must sum to one")
    return weights


def is_near_tie(weights: torch.Tensor, threshold: float) -> bool:
    if weights.shape != (2,) or not 0.0 <= threshold < 0.5:
        raise ValueError("Invalid tie diagnostic inputs")
    return abs(float(weights[0] - weights[1])) <= threshold


@dataclass(frozen=True)
class AlphaControls:
    correct: torch.Tensor
    shuffled: torch.Tensor
    constant: torch.Tensor


def build_alpha_controls(
    correct: torch.Tensor,
    *,
    constant_helpfulness: float = 0.5,
) -> AlphaControls:
    values = validate_alpha(correct)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Alpha controls require at least two validation tasks")
    if not 0.0 <= constant_helpfulness <= 1.0:
        raise ValueError("constant_helpfulness must be in [0, 1]")
    shuffled = torch.roll(values, shifts=1, dims=0)
    constant = torch.tensor(
        [constant_helpfulness, 1.0 - constant_helpfulness],
        dtype=values.dtype,
        device=values.device,
    ).expand_as(values).clone()
    return AlphaControls(correct=values, shuffled=shuffled, constant=constant)


def validation_alphas(helpfulness_grid: Sequence[float]) -> torch.Tensor:
    if not helpfulness_grid:
        raise ValueError("Validation alpha grid cannot be empty")
    values = torch.tensor(helpfulness_grid, dtype=torch.float32)
    return validate_alpha(torch.stack((values, 1.0 - values), dim=-1))
