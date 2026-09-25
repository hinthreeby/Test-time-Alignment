"""
TokenRouter — TARo-inspired binary gate for selective PARM intervention.

Predicts w_t ∈ {0, w_fixed} (binary) or w_t ∈ [0, 1] (continuous).
Uses only base-model-side features to enable compute savings (pre-PARM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    gate_prob: torch.Tensor    # (batch,) probability of w > 0
    w_t: torch.Tensor          # (batch,) selected weight
    uncertainty: torch.Tensor  # (batch,) uncertainty estimate


class TokenRouter(nn.Module):
    """
    Binary token-level intervention controller.

    Input features (same as ObjectiveTracker + objective deficits):
        - top_k log-probs        (top_k,)
        - entropy                (1,)
        - top-1/top-2 margin     (1,)
        - normalized position    (1,)
        - alpha_t                (K,)
        - objective_deficits     (K,)   how far each objective is behind
        - previous gate decision (1,)

    Output:
        - gate_prob: P(w_t > 0)
        - w_t: selected weight (binary: 0 or w_fixed; or continuous)
    """

    def __init__(
        self,
        num_objectives: int = 2,
        top_k: int = 32,
        hidden_dim: int = 64,
        w_fixed: float = 1.0,
        mode: str = "binary",   # "binary" | "continuous"
        dropout: float = 0.1,
        feature_schema: str = "full_v1",
    ):
        super().__init__()
        if num_objectives <= 0:
            raise ValueError("num_objectives must be positive")
        if top_k < 2:
            raise ValueError("top_k must be >= 2")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2")
        if not 0.0 <= w_fixed <= 1.0:
            raise ValueError("w_fixed must be in [0, 1]")
        if mode not in {"binary", "continuous"}:
            raise ValueError("mode must be 'binary' or 'continuous'")
        if feature_schema not in {"full_v1", "compact_v1"}:
            raise ValueError("feature_schema must be 'full_v1' or 'compact_v1'")
        self.num_objectives = num_objectives
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.w_fixed = w_fixed
        self.mode = mode
        self.dropout = dropout
        self.feature_schema = feature_schema

        # full_v1: top-k logprobs + entropy + logprob margin + position
        # compact_v1: entropy + top1 prob + probability margin + top-k mass + position
        distribution_dim = top_k + 2 if feature_schema == "full_v1" else 4
        input_dim = distribution_dim + 1 + num_objectives + num_objectives + 1
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden_dim // 2, 1)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 1)

    def build_features(
        self,
        base_logits: torch.Tensor,    # (batch, vocab)
        alpha_t: torch.Tensor,        # (K,) or (batch, K)
        objective_deficits: torch.Tensor,  # (K,) or (batch, K)
        step_index: int,
        max_steps: int = 256,
        prev_gate: Optional[torch.Tensor] = None,  # (batch,) or None
    ) -> torch.Tensor:
        batch = base_logits.shape[0]

        log_probs = F.log_softmax(base_logits, dim=-1)
        top_k_lp, _ = log_probs.topk(self.top_k, dim=-1)  # (batch, top_k)

        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        top2_lp = top_k_lp[:, :2]
        logprob_margin = top2_lp[:, 0:1] - top2_lp[:, 1:2]

        pos = torch.tensor([[step_index / max(max_steps - 1, 1)]],
                           dtype=base_logits.dtype, device=base_logits.device).expand(batch, 1)

        if alpha_t.dim() == 1:
            alpha_t = alpha_t.unsqueeze(0).expand(batch, -1)
        alpha_t = alpha_t.to(base_logits.dtype)

        if objective_deficits.dim() == 1:
            objective_deficits = objective_deficits.unsqueeze(0).expand(batch, -1)
        objective_deficits = objective_deficits.to(base_logits.dtype)

        if prev_gate is None:
            prev_gate_feat = torch.zeros(batch, 1, dtype=base_logits.dtype, device=base_logits.device)
        else:
            prev_gate_feat = prev_gate.unsqueeze(-1).to(base_logits.dtype)

        if self.feature_schema == "full_v1":
            distribution_features = [top_k_lp, entropy, logprob_margin]
        else:
            top_k_probs = top_k_lp.exp()
            top1_probability = top_k_probs[:, 0:1]
            probability_margin = top_k_probs[:, 0:1] - top_k_probs[:, 1:2]
            top_k_mass = top_k_probs.sum(dim=-1, keepdim=True)
            distribution_features = [
                entropy, top1_probability, probability_margin, top_k_mass
            ]

        features = torch.cat([
            *distribution_features, pos, alpha_t, objective_deficits, prev_gate_feat
        ], dim=-1)
        return features.to(dtype=self.net[0].weight.dtype)

    def gate_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid gate logits for precomputed training features."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}"
            )
        hidden = self.net(features)
        return self.gate_head(hidden).squeeze(-1)

    def forward_features(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> RouterOutput:
        """Run the router on features materialized by an offline oracle."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}"
            )
        hidden = self.net(features)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        uncertainty = self.uncertainty_head(hidden).squeeze(-1).exp()
        if self.mode == "binary":
            w_t = (gate_prob > threshold).to(gate_prob.dtype) * self.w_fixed
        else:
            w_t = gate_prob * self.w_fixed
        return RouterOutput(gate_prob=gate_prob, w_t=w_t, uncertainty=uncertainty)

    def forward(
        self,
        base_logits: torch.Tensor,
        alpha_t: torch.Tensor,
        objective_deficits: torch.Tensor,
        step_index: int,
        max_steps: int = 256,
        prev_gate: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> RouterOutput:
        features = self.build_features(
            base_logits, alpha_t, objective_deficits, step_index, max_steps, prev_gate
        )
        return self.forward_features(features, threshold)

    def save(self, path: str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "num_objectives": self.num_objectives,
                "top_k": self.top_k,
                "hidden_dim": self.hidden_dim,
                "w_fixed": self.w_fixed,
                "mode": self.mode,
                "dropout": self.dropout,
                "feature_schema": self.feature_schema,
            },
            "metadata": metadata or {},
        }, destination)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TokenRouter":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        cfg.update(kwargs)
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
