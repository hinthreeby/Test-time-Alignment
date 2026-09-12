"""Minimal TARO-like MLP router for RAD candidate scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from router.features import build_router_features


@dataclass(frozen=True)
class RADTokenRouterConfig:
    beta_max: float
    beta_init: float | None = None
    top_k: int = 20
    hidden_dim: int = 64
    eps: float = 1e-6


class RADTokenRouter(nn.Module):
    """Predict a per-token RAD beta from normalized base logits and reward scores."""

    def __init__(
        self,
        beta_max: float,
        beta_init: float | None = None,
        top_k: int = 20,
        hidden_dim: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert top_k == 20, f"Minimal v1 router expects top_k=20, got {top_k}"
        assert hidden_dim == 64, f"Minimal v1 router expects hidden_dim=64, got {hidden_dim}"
        assert beta_max > 0, f"beta_max must be positive, got {beta_max}"
        if beta_init is not None:
            assert 0 < beta_init < beta_max, "beta_init must satisfy 0 < beta_init < beta_max"

        self.config = RADTokenRouterConfig(
            beta_max=float(beta_max),
            beta_init=float(beta_init) if beta_init is not None else None,
            top_k=int(top_k),
            hidden_dim=int(hidden_dim),
            eps=float(eps),
        )
        self.beta_max = float(beta_max)
        self.top_k = int(top_k)
        self.eps = float(eps)
        self.mlp = nn.Sequential(
            nn.Linear(2 * top_k, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if beta_init is not None:
            initial_gate = float(beta_init) / float(beta_max)
            final_layer = self.mlp[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, math.log(initial_gate / (1.0 - initial_gate)))

    def router_features(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> torch.Tensor:
        return build_router_features(base_logits, reward_scores, top_k=self.top_k, eps=self.eps)

    def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.router_features(base_logits, reward_scores)
        expected_dim = 2 * self.top_k
        assert features.shape[-1] == expected_dim, f"Expected {expected_dim} router features, got {features.shape[-1]}"
        gate = torch.sigmoid(self.mlp(features))
        beta = self.beta_max * gate
        assert beta.shape == gate.shape
        assert beta.shape[-1] == 1
        assert torch.isfinite(beta).all(), "beta contains NaN or infinity"
        assert bool((beta >= 0).all() and (beta <= self.beta_max).all()), "beta is outside [0, beta_max]"
        return beta, gate
