"""Faithful Top-K TARO gate and full-vocabulary routing equations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.config import TARORouterConfig
from router_v2.device import resolve_device


@dataclass(frozen=True)
class TARORouterOutput:
    """Per-token TARO gate, routed logits, and diagnostic Top-K sets."""

    alpha: torch.Tensor
    guided_logits: torch.Tensor
    topk: TAROTopKBatch


class TAROTokenRouter(nn.Module):
    """Flattened independent Top-K features followed by a shallow TARO MLP."""

    def __init__(self, config: TARORouterConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = 2 * config.top_k * (config.token_embedding_dim + 1)
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.token_embedding_dim,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        final_layer = self.gate_mlp[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("TARO gate must end in a linear layer")
        nn.init.constant_(
            final_layer.bias,
            math.log(config.initial_alpha / (1.0 - config.initial_alpha)),
        )

    def build_topk_features(self, topk: TAROTopKBatch) -> torch.Tensor:
        """Build h_top-k = [base pairs; reward pairs] without pooling."""

        self._validate_topk(topk)
        base_features = self._build_stream_features(
            topk.base_token_ids,
            topk.base_logits,
        )
        reward_features = self._build_stream_features(
            topk.reward_token_ids,
            topk.reward_logits,
        )
        features = torch.cat(
            (
                torch.flatten(base_features, start_dim=-2),
                torch.flatten(reward_features, start_dim=-2),
            ),
            dim=-1,
        )
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                "Invalid TARO Top-K feature dimension: "
                f"expected {self.feature_dim}, got {features.shape[-1]}"
            )
        return features

    def predict_alpha(self, topk: TAROTopKBatch) -> torch.Tensor:
        features = self.build_topk_features(topk)
        alpha = torch.sigmoid(self.gate_mlp(features))
        return alpha.clamp(
            min=self.config.alpha_eps,
            max=1.0 - self.config.alpha_eps,
        )

    def forward(
        self,
        topk: TAROTopKBatch,
        base_logits: torch.Tensor,
        reward_logits: torch.Tensor,
    ) -> TARORouterOutput:
        """Predict alpha from Top-K sets and route complete vocabulary logits."""

        alpha = self.predict_alpha(topk)
        if topk.base_logits.shape[:-1] != base_logits.shape[:-1]:
            raise ValueError("Top-K and full logits must have matching leading shapes")
        guided_logits = self.route_logits(alpha, base_logits, reward_logits)
        return TARORouterOutput(
            alpha=alpha,
            guided_logits=guided_logits,
            topk=topk,
        )

    def forward_from_full_logits(
        self,
        base_logits: torch.Tensor,
        reward_logits: torch.Tensor,
    ) -> TARORouterOutput:
        """Select independent Top-K sets and route the full vocabulary."""

        topk = select_taro_topk(
            base_logits,
            reward_logits,
            top_k=self.config.top_k,
        )
        return self(topk, base_logits, reward_logits)

    @staticmethod
    def route_logits(
        alpha: torch.Tensor,
        base_logits: torch.Tensor,
        reward_logits: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape != reward_logits.shape:
            raise ValueError("Logits to route must have identical shapes")
        if alpha.shape != base_logits.shape[:-1] + (1,):
            raise ValueError("alpha must have shape (..., 1) for the routed logits")
        if alpha.device != base_logits.device or alpha.device != reward_logits.device:
            raise ValueError("alpha and routed logits must be on the same device")
        detached_base = base_logits.detach().to(dtype=alpha.dtype)
        detached_reward = reward_logits.detach().to(dtype=alpha.dtype)
        return (1.0 - alpha) * detached_base + alpha * detached_reward

    def _build_stream_features(
        self,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        token_features = self.token_embedding(token_ids.detach().to(dtype=torch.long))
        scalar_features = logits.detach().to(dtype=token_features.dtype).unsqueeze(-1)
        return torch.cat((scalar_features, token_features), dim=-1)

    def _validate_topk(self, topk: TAROTopKBatch) -> None:
        shapes = {
            topk.base_token_ids.shape,
            topk.base_logits.shape,
            topk.reward_token_ids.shape,
            topk.reward_logits.shape,
        }
        if len(shapes) != 1:
            raise ValueError("All base/reward Top-K tensors must have identical shapes")
        if topk.base_token_ids.ndim < 1:
            raise ValueError("Top-K tensors must have shape (..., top_k)")
        if topk.base_token_ids.shape[-1] != self.config.top_k:
            raise ValueError(f"Expected top_k={self.config.top_k} candidates")
        model_device = self.token_embedding.weight.device
        tensors = (
            topk.base_token_ids,
            topk.base_logits,
            topk.reward_token_ids,
            topk.reward_logits,
        )
        if any(tensor.device != model_device for tensor in tensors):
            raise ValueError("Top-K tensors and TARO router must use the same device")
        for name, token_ids in (
            ("base", topk.base_token_ids),
            ("reward", topk.reward_token_ids),
        ):
            if token_ids.dtype == torch.bool or token_ids.is_floating_point():
                raise TypeError(f"{name} token IDs must use an integer dtype")
            if bool(
                (
                    (token_ids < 0)
                    | (token_ids >= self.config.vocab_size)
                ).any()
            ):
                raise ValueError(f"{name} token IDs contain an out-of-vocabulary index")


def build_taro_router(config: TARORouterConfig) -> TAROTokenRouter:
    """Build the TARO router on the device selected by its config."""

    device = resolve_device(
        config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    return TAROTokenRouter(config).to(device)
