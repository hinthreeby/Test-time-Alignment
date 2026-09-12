from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class PrefixScorer(nn.Module):
    """GPT-2 backbone with a scalar value head for a text prefix."""

    def __init__(self, model_path: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            last_indices = torch.full(
                (input_ids.size(0),), input_ids.size(1) - 1,
                dtype=torch.long, device=input_ids.device,
            )
        else:
            last_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        last_hidden = hidden[
            torch.arange(hidden.size(0), device=hidden.device), last_indices
        ]
        return torch.sigmoid(self.value_head(last_hidden).squeeze(-1))
