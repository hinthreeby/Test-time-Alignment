from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SignalOutput:
    name: str
    objective: str
    scores: torch.Tensor
    available: bool
    source: str
    cost_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, batch_size: int, top_k: int) -> None:
        if tuple(self.scores.shape) != (batch_size, top_k):
            raise ValueError(
                f"{self.name}: expected score shape {(batch_size, top_k)}, "
                f"got {tuple(self.scores.shape)}"
            )
        if not torch.isfinite(self.scores).all():
            raise ValueError(f"{self.name}: scores contain NaN or infinity")
