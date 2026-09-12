from __future__ import annotations

import torch
from torch import nn


class MultiSignalController(nn.Module):
    """Small token-level router over normalized reward signals."""

    def __init__(self, num_signals: int, hidden_dim: int = 64, signal_costs=None):
        super().__init__()
        self.num_signals = int(num_signals)
        costs = signal_costs if signal_costs is not None else [1.0] * self.num_signals
        self.register_buffer("signal_costs", torch.tensor(costs, dtype=torch.float32))

        # LM statistics (mean/std/max/entropy/margin), per-signal mean/std,
        # and one cross-signal disagreement feature.
        feature_dim = 6 + 2 * self.num_signals
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(hidden_dim, self.num_signals)
        self.strength_head = nn.Linear(hidden_dim, 1)
        self.trust_head = nn.Linear(hidden_dim, 1)

    def forward(self, lm_logits: torch.Tensor, signal_scores: torch.Tensor):
        lm = lm_logits.float()
        signals = signal_scores.float()
        probabilities = torch.softmax(lm, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        top_two = torch.topk(lm, k=min(2, lm.size(-1)), dim=-1).values
        margin = top_two[:, 0] - top_two[:, -1]

        features = torch.cat(
            [
                lm.mean(dim=-1, keepdim=True),
                lm.std(dim=-1, keepdim=True, unbiased=False),
                lm.max(dim=-1, keepdim=True).values,
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                signals.mean(dim=1),
                signals.std(dim=1, unbiased=False),
                signals.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        hidden = self.encoder(features)
        weight_logits = self.weight_head(hidden) - 0.05 * self.signal_costs
        weights = torch.softmax(weight_logits, dim=-1)
        trust = torch.sigmoid(self.trust_head(hidden).squeeze(-1))
        strength = 2.0 * torch.sigmoid(self.strength_head(hidden).squeeze(-1)) * trust
        return {"weights": weights, "strength": strength, "trust": trust}
