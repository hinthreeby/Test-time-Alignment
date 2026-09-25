"""
ObjectiveTracker — lightweight prefix-level per-objective reward estimator.

Takes base model features and outputs incremental reward vector per objective.
Used by DynamicPreferenceController to update alpha_t at each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class TrackerOutput:
    rewards: torch.Tensor     # (batch, K) incremental reward per objective
    uncertainty: torch.Tensor  # (batch,) scalar uncertainty estimate


class ObjectiveTracker(nn.Module):
    """
    Lightweight MLP that estimates incremental per-objective reward
    from base model features.

    Input features (concatenated):
        - top_k log-probs from base model  (top_k,)
        - entropy of base distribution     (1,)
        - top-1/top-2 margin               (1,)
        - normalized token position        (1,)
        - alpha_user                       (K,)

    Total input dim = top_k + 3 + K
    """

    def __init__(
        self,
        num_objectives: int = 2,
        top_k: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_objectives = num_objectives
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        input_dim = top_k + 3 + num_objectives  # logprobs + entropy + margin + position + alpha
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Reward head: K outputs
        self.reward_head = nn.Linear(hidden_dim // 2, num_objectives)
        # Uncertainty head: scalar (log-variance)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 1)
        self.register_buffer("target_mean", torch.zeros(num_objectives))
        self.register_buffer("target_std", torch.ones(num_objectives))

    def set_target_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.num_objectives,) or std.shape != (self.num_objectives,):
            raise ValueError("target normalization must have shape (num_objectives,)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("target normalization must be finite with positive std")
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std))

    def build_features(
        self,
        base_logits: torch.Tensor,   # (batch, vocab)
        alpha_user: torch.Tensor,     # (K,) or (batch, K)
        step_index: int,
        max_steps: int = 256,
    ) -> torch.Tensor:
        """Build feature vector from base model logits."""
        import torch.nn.functional as F

        batch = base_logits.shape[0]

        log_probs = F.log_softmax(base_logits, dim=-1)

        # Top-k log probs
        top_k_lp, _ = log_probs.topk(self.top_k, dim=-1)  # (batch, top_k)

        # Entropy
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # (batch, 1)

        # Top-1/top-2 margin
        top2_lp = top_k_lp[:, :2]
        margin = (top2_lp[:, 0:1] - top2_lp[:, 1:2])  # (batch, 1)

        # Normalized position
        pos = torch.tensor([[step_index / max(max_steps - 1, 1)]], dtype=base_logits.dtype, device=base_logits.device)
        pos = pos.expand(batch, 1)  # (batch, 1)

        # Alpha user
        if alpha_user.dim() == 1:
            alpha_user = alpha_user.unsqueeze(0).expand(batch, -1)
        alpha_user = alpha_user.to(base_logits.dtype)

        features = torch.cat([top_k_lp, entropy, margin, pos, alpha_user], dim=-1)
        return features.to(dtype=self.net[0].weight.dtype)

    def normalized_outputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized reward predictions and scalar log variance."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}"
            )
        hidden = self.net(features)
        return self.reward_head(hidden), self.uncertainty_head(hidden).squeeze(-1)

    def forward_features(self, features: torch.Tensor) -> TrackerOutput:
        normalized_rewards, log_var = self.normalized_outputs(features)
        rewards = normalized_rewards * self.target_std + self.target_mean
        uncertainty = torch.sigmoid(log_var)
        return TrackerOutput(rewards=rewards, uncertainty=uncertainty)

    def forward(
        self,
        base_logits: torch.Tensor,
        alpha_user: torch.Tensor,
        step_index: int,
        max_steps: int = 256,
    ) -> TrackerOutput:
        """
        Args:
            base_logits: (batch, vocab)
            alpha_user: (K,) or (batch, K)
            step_index: current decoding step
            max_steps: for position normalization

        Returns:
            TrackerOutput with rewards (batch, K) and uncertainty (batch,)
        """
        features = self.build_features(base_logits, alpha_user, step_index, max_steps)
        return self.forward_features(features)

    def save(self, path: str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": {
            "num_objectives": self.num_objectives,
            "top_k": self.top_k,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }, "metadata": metadata or {}}, destination)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ObjectiveTracker":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        cfg.update(kwargs)
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
