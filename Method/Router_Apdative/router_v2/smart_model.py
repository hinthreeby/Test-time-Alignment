"""Feature-conditioned Smart Router V2 with a universal token-level lambda."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from router_v2.cache.features import (
    ConfidenceDisagreementFeatures,
    compute_confidence_disagreement,
)
from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.device import resolve_device
from router_v2.smart_config import SmartRouterConfig


SMART_CONFIDENCE_FEATURE_NAMES = (
    "base_entropy",
    "guide_entropy",
    "base_margin",
    "guide_margin",
    "js_divergence",
    "top1_agreement",
    "topk_overlap",
    "base_std",
    "guide_std",
    "base_range",
    "guide_range",
)


@dataclass(frozen=True)
class SmartRouterBatch:
    """Target-independent inputs accepted by Smart Router V2."""

    topk: TAROTopKBatch
    position: torch.Tensor | None = None
    selected_score: torch.Tensor | None = None
    preference: torch.Tensor | None = None


@dataclass(frozen=True)
class SmartRouterOutput:
    """Universal routing strength and optional full-vocabulary output."""

    gate: torch.Tensor
    lambda_t: torch.Tensor
    topk: TAROTopKBatch
    router_state: torch.Tensor | None
    guided_logits: torch.Tensor | None


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class SmartTokenRouter(nn.Module):
    """Route using Top-K identity, current statistics, history, and preference."""

    history_input_dim = 4
    confidence_dim = len(SMART_CONFIDENCE_FEATURE_NAMES)

    def __init__(self, config: SmartRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding: nn.Embedding | None = None
        self.base_candidate_encoder: nn.Module | None = None
        self.guide_candidate_encoder: nn.Module | None = None
        if config.candidate_feature_mode != "disabled":
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.token_embedding_dim,
            )
        if config.candidate_feature_mode == "token_aware_mean_max":
            candidate_input_dim = config.token_embedding_dim + 1
            self.base_candidate_encoder = nn.Sequential(
                nn.Linear(candidate_input_dim, config.candidate_hidden_dim),
                _activation(config.activation),
            )
            self.guide_candidate_encoder = nn.Sequential(
                nn.Linear(candidate_input_dim, config.candidate_hidden_dim),
                _activation(config.activation),
            )

        self.history_gru: nn.GRUCell | None = None
        if config.use_history:
            self.history_gru = nn.GRUCell(
                self.history_input_dim,
                config.history_hidden_dim,
            )

        self.preference_encoder: nn.Module | None = None
        if config.use_preference:
            self.preference_encoder = nn.Sequential(
                nn.Linear(config.preference_dim, config.preference_hidden_dim),
                _activation(config.activation),
                nn.Linear(
                    config.preference_hidden_dim,
                    config.preference_embedding_dim,
                ),
                _activation(config.activation),
            )

        group_dimensions: list[tuple[str, int]] = []
        if config.candidate_feature_mode == "taro_flatten":
            group_dimensions.append(
                (
                    "candidate",
                    2 * config.top_k * (config.token_embedding_dim + 1),
                )
            )
        elif config.candidate_feature_mode == "token_aware_mean_max":
            group_dimensions.append(
                ("candidate", 4 * config.candidate_hidden_dim)
            )
        if config.use_confidence:
            group_dimensions.append(("confidence", self.confidence_dim))
        if config.use_position:
            group_dimensions.append(("position", 1))
        if config.use_history:
            group_dimensions.append(("history", config.history_hidden_dim))
        if config.use_preference:
            group_dimensions.append(
                ("preference", config.preference_embedding_dim)
            )

        self.feature_slices: dict[str, slice] = {}
        offset = 0
        for name, dimension in group_dimensions:
            self.feature_slices[name] = slice(offset, offset + dimension)
            offset += dimension
        self.feature_dim = offset
        if self.feature_dim <= 0:
            raise ValueError("Smart Router fusion requires at least one feature")

        normalizer: nn.Module
        if self.feature_dim == 1:
            normalizer = nn.Identity()
        else:
            normalizer = nn.LayerNorm(self.feature_dim)
        self.fusion_mlp = nn.Sequential(
            normalizer,
            nn.Linear(self.feature_dim, config.fusion_hidden_dim),
            _activation(config.activation),
            nn.Linear(
                config.fusion_hidden_dim,
                config.fusion_bottleneck_dim,
            ),
            _activation(config.activation),
            nn.Linear(config.fusion_bottleneck_dim, 1),
        )
        final_layer = self.fusion_mlp[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Smart Router fusion must end in a linear layer")
        nn.init.constant_(
            final_layer.bias,
            math.log(config.initial_gate / (1.0 - config.initial_gate)),
        )

    def build_static_feature_groups(
        self,
        batch: SmartRouterBatch,
    ) -> dict[str, torch.Tensor]:
        """Build enabled current-token groups; history is added causally later."""

        groups, _ = self._build_static_feature_groups(batch)
        return groups

    def predict_lambda(self, batch: SmartRouterBatch) -> SmartRouterOutput:
        groups, confidence = self._build_static_feature_groups(batch)
        leading_shape = batch.topk.base_logits.shape[:-1]
        if self.config.use_history:
            if len(leading_shape) != 2:
                raise ValueError(
                    "History-enabled Smart Router requires Top-K shape [B, T, K]"
                )
            if batch.selected_score is None:
                raise ValueError("History-enabled router requires selected_score")
            selected_score = batch.selected_score.detach()
            if selected_score.shape != leading_shape:
                raise ValueError("selected_score must have shape [B, T]")
            if not bool(torch.isfinite(selected_score).all()):
                raise ValueError("selected_score must be finite")
            gate, lambda_t, router_state = self._predict_sequence_with_history(
                groups,
                confidence,
                selected_score,
                leading_shape,
            )
        else:
            gate, lambda_t = self._fuse(groups)
            router_state = None
        return SmartRouterOutput(
            gate=gate,
            lambda_t=lambda_t,
            topk=batch.topk,
            router_state=router_state,
            guided_logits=None,
        )

    def forward(
        self,
        batch: SmartRouterBatch,
        base_logits: torch.Tensor | None = None,
        guide_logits: torch.Tensor | None = None,
    ) -> SmartRouterOutput:
        output = self.predict_lambda(batch)
        if (base_logits is None) != (guide_logits is None):
            raise ValueError("base_logits and guide_logits must be provided together")
        guided_logits = None
        if base_logits is not None and guide_logits is not None:
            if batch.topk.base_logits.shape[:-1] != base_logits.shape[:-1]:
                raise ValueError(
                    "Top-K and full logits must have matching leading shapes"
                )
            guided_logits = self.route_logits(
                output.lambda_t,
                base_logits,
                guide_logits,
            )
        return SmartRouterOutput(
            gate=output.gate,
            lambda_t=output.lambda_t,
            topk=output.topk,
            router_state=output.router_state,
            guided_logits=guided_logits,
        )

    def forward_from_full_logits(
        self,
        base_logits: torch.Tensor,
        guide_logits: torch.Tensor,
        *,
        position: torch.Tensor | None = None,
        selected_score: torch.Tensor | None = None,
        preference: torch.Tensor | None = None,
    ) -> SmartRouterOutput:
        topk = select_taro_topk(
            base_logits,
            guide_logits,
            top_k=self.config.top_k,
        )
        return self(
            SmartRouterBatch(
                topk=topk,
                position=position,
                selected_score=selected_score,
                preference=preference,
            ),
            base_logits,
            guide_logits,
        )

    @staticmethod
    def route_logits(
        lambda_t: torch.Tensor,
        base_logits: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the TARO/RAD interpolation to every vocabulary item.

        PARM-TARO must not reuse this equation: its separate integration uses
        PARM-specific log-probability guidance semantics.
        """

        if base_logits.shape != guide_logits.shape:
            raise ValueError("Base and guide logits must have identical shapes")
        if lambda_t.shape != base_logits.shape[:-1] + (1,):
            raise ValueError(
                "lambda_t must have shape (..., 1) for universal routing"
            )
        if (
            lambda_t.device != base_logits.device
            or lambda_t.device != guide_logits.device
        ):
            raise ValueError("lambda_t and full logits must use the same device")
        base = base_logits.detach().to(dtype=lambda_t.dtype)
        guide = guide_logits.detach().to(dtype=lambda_t.dtype)
        return base + lambda_t * (guide - base)

    def _build_static_feature_groups(
        self,
        batch: SmartRouterBatch,
    ) -> tuple[
        dict[str, torch.Tensor],
        ConfidenceDisagreementFeatures | None,
    ]:
        self._validate_topk(batch.topk)
        leading_shape = batch.topk.base_logits.shape[:-1]
        groups: dict[str, torch.Tensor] = {}
        if self.config.candidate_feature_mode != "disabled":
            groups["candidate"] = self._build_candidate_features(batch.topk)

        position = self._validated_position(batch.position, leading_shape)
        confidence = None
        if self.config.use_confidence:
            confidence_position = (
                position
                if position is not None
                else torch.zeros(
                    leading_shape,
                    dtype=torch.long,
                    device=batch.topk.base_logits.device,
                )
            )
            confidence = compute_confidence_disagreement(
                batch.topk.base_token_ids,
                batch.topk.base_logits,
                batch.topk.reward_token_ids,
                batch.topk.reward_logits,
                confidence_position,
                max_position=self.config.max_position,
            )
        if self.config.use_confidence:
            if confidence is None:
                raise RuntimeError("Confidence features were not constructed")
            groups["confidence"] = torch.stack(
                [
                    getattr(confidence, name)
                    for name in SMART_CONFIDENCE_FEATURE_NAMES
                ],
                dim=-1,
            ).to(dtype=self._model_dtype())
        if self.config.use_position:
            if position is None:
                raise ValueError("Position feature is enabled but position is absent")
            groups["position"] = (
                position.detach().to(dtype=self._model_dtype()).unsqueeze(-1)
                / float(self.config.max_position)
            )
        if self.config.use_preference:
            groups["preference"] = self._build_preference_features(
                batch.preference,
                leading_shape,
            )
        return groups, confidence

    def _build_candidate_features(self, topk: TAROTopKBatch) -> torch.Tensor:
        base_pairs = self._candidate_pairs(
            topk.base_token_ids,
            topk.base_logits,
        )
        guide_pairs = self._candidate_pairs(
            topk.reward_token_ids,
            topk.reward_logits,
        )
        if self.config.candidate_feature_mode == "taro_flatten":
            return torch.cat(
                (
                    torch.flatten(base_pairs, start_dim=-2),
                    torch.flatten(guide_pairs, start_dim=-2),
                ),
                dim=-1,
            )
        if self.config.candidate_feature_mode != "token_aware_mean_max":
            raise RuntimeError("Candidate encoder is disabled")
        if (
            self.base_candidate_encoder is None
            or self.guide_candidate_encoder is None
        ):
            raise RuntimeError("Candidate stream encoders were not initialized")
        base_encoded = self.base_candidate_encoder(base_pairs)
        guide_encoded = self.guide_candidate_encoder(guide_pairs)
        return torch.cat(
            (
                base_encoded.mean(dim=-2),
                base_encoded.max(dim=-2).values,
                guide_encoded.mean(dim=-2),
                guide_encoded.max(dim=-2).values,
            ),
            dim=-1,
        )

    def _candidate_pairs(
        self,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_embedding is None:
            raise RuntimeError("Token embedding is disabled")
        embedding = self.token_embedding(token_ids.detach().long())
        scalar = logits.detach().to(dtype=embedding.dtype).unsqueeze(-1)
        return torch.cat((scalar, embedding), dim=-1)

    def _build_preference_features(
        self,
        preference: torch.Tensor | None,
        leading_shape: torch.Size,
    ) -> torch.Tensor:
        if preference is None:
            raise ValueError("Preference feature is enabled but preference is absent")
        if self.preference_encoder is None:
            raise RuntimeError("Preference encoder was not initialized")
        expected_per_token = leading_shape + (self.config.preference_dim,)
        values = preference.detach()
        if values.shape == expected_per_token:
            expanded = values
        elif len(leading_shape) >= 2 and values.shape == (
            leading_shape[0],
            self.config.preference_dim,
        ):
            reshape = (leading_shape[0],) + (1,) * (len(leading_shape) - 1)
            expanded = values.reshape(
                reshape + (self.config.preference_dim,)
            ).expand(expected_per_token)
        else:
            raise ValueError(
                "preference must be per-token or one vector per sequence"
            )
        if not bool(torch.isfinite(expanded).all()):
            raise ValueError("preference must be finite")
        return self.preference_encoder(
            expanded.to(
                device=self._model_device(),
                dtype=self._model_dtype(),
            )
        )

    def _predict_sequence_with_history(
        self,
        groups: dict[str, torch.Tensor],
        confidence: ConfidenceDisagreementFeatures | None,
        selected_score: torch.Tensor,
        leading_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.history_gru is None:
            raise RuntimeError("History GRU is unavailable")
        batch_size, sequence_length = leading_shape
        dtype = self._model_dtype()
        device = self._model_device()
        hidden = torch.zeros(
            batch_size,
            self.config.history_hidden_dim,
            dtype=dtype,
            device=device,
        )
        previous_lambda = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        previous_score = torch.zeros_like(previous_lambda)
        previous_entropy = torch.zeros_like(previous_lambda)
        previous_js = torch.zeros_like(previous_lambda)
        gates = []
        lambdas = []
        states = []
        selected_score = selected_score.to(device=device, dtype=dtype)
        if confidence is None:
            base_entropy = torch.zeros(
                batch_size,
                sequence_length,
                dtype=dtype,
                device=device,
            )
            js_divergence = torch.zeros_like(base_entropy)
        else:
            base_entropy = confidence.base_entropy.to(device=device, dtype=dtype)
            js_divergence = confidence.js_divergence.to(
                device=device,
                dtype=dtype,
            )
        for index in range(sequence_length):
            history_input = torch.cat(
                (
                    previous_lambda,
                    previous_score,
                    previous_entropy,
                    previous_js,
                ),
                dim=-1,
            )
            hidden = self.history_gru(history_input, hidden)
            step_groups = {
                name: values[:, index]
                for name, values in groups.items()
            }
            step_groups["history"] = hidden
            gate, lambda_t = self._fuse(step_groups)
            gates.append(gate)
            lambdas.append(lambda_t)
            states.append(hidden)
            previous_lambda = lambda_t
            previous_score = selected_score[:, index].unsqueeze(-1)
            previous_entropy = base_entropy[:, index].unsqueeze(-1)
            previous_js = js_divergence[:, index].unsqueeze(-1)
        return (
            torch.stack(gates, dim=1),
            torch.stack(lambdas, dim=1),
            torch.stack(states, dim=1),
        )

    def _fuse(
        self,
        groups: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = tuple(self.feature_slices)
        if set(groups) != set(expected):
            raise ValueError(
                f"Fusion feature groups mismatch: expected={expected}, "
                f"got={tuple(groups)}"
            )
        features = torch.cat([groups[name] for name in expected], dim=-1)
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"Expected fusion dimension {self.feature_dim}, "
                f"got {features.shape[-1]}"
            )
        logits = self.fusion_mlp(features)
        gate = torch.sigmoid(logits).clamp(
            self.config.lambda_eps,
            1.0 - self.config.lambda_eps,
        )
        return gate, self.config.lambda_max * gate

    def _validated_position(
        self,
        position: torch.Tensor | None,
        leading_shape: torch.Size,
    ) -> torch.Tensor | None:
        if position is None:
            if self.config.use_position:
                raise ValueError("Position feature is enabled but position is absent")
            return None
        if position.shape != leading_shape:
            raise ValueError("position must match Top-K leading dimensions")
        if position.dtype == torch.bool or position.is_floating_point():
            raise TypeError("position must use an integer dtype")
        if bool(
            ((position < 0) | (position > self.config.max_position)).any()
        ):
            raise ValueError("position is outside the configured range")
        return position.detach().to(
            device=self._model_device(),
            dtype=torch.long,
        )

    def _validate_topk(self, topk: TAROTopKBatch) -> None:
        tensors = (
            topk.base_token_ids,
            topk.base_logits,
            topk.reward_token_ids,
            topk.reward_logits,
        )
        if len({tensor.shape for tensor in tensors}) != 1:
            raise ValueError("All independent Top-K tensors must share one shape")
        if topk.base_token_ids.ndim < 1:
            raise ValueError("Top-K tensors must have shape (..., K)")
        if topk.base_token_ids.shape[-1] != self.config.top_k:
            raise ValueError(f"Expected top_k={self.config.top_k}")
        if any(tensor.device != self._model_device() for tensor in tensors):
            raise ValueError("Top-K tensors and Smart Router must share a device")
        if not bool(torch.isfinite(topk.base_logits).all()) or not bool(
            torch.isfinite(topk.reward_logits).all()
        ):
            raise ValueError("Top-K logits must be finite")
        for name, token_ids in (
            ("base", topk.base_token_ids),
            ("guide", topk.reward_token_ids),
        ):
            if token_ids.dtype == torch.bool or token_ids.is_floating_point():
                raise TypeError(f"{name} token IDs must use an integer dtype")
            if bool(
                (
                    (token_ids < 0)
                    | (token_ids >= self.config.vocab_size)
                ).any()
            ):
                raise ValueError(f"{name} token IDs are out of vocabulary")

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _model_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def build_smart_router(config: SmartRouterConfig) -> SmartTokenRouter:
    device = resolve_device(
        config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    return SmartTokenRouter(config).to(device)
