"""Static PARM equation used as the adaptive implementation's oracle."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def parm_guided_logprobs(
    base_logits: torch.Tensor,
    guide_logits: torch.Tensor,
    guidance_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Normalize log pi_base + scale * log pi_PARM over full vocabulary."""

    if base_logits.shape != guide_logits.shape:
        raise ValueError("Base and PARM guide logits must have identical shapes")
    base = base_logits.detach().float()
    guide = guide_logits.detach().float()
    base_logprobs = F.log_softmax(base, dim=-1)
    guide_logprobs = F.log_softmax(guide, dim=-1)
    if isinstance(guidance_scale, torch.Tensor):
        scale = guidance_scale.detach().to(
            device=base.device, dtype=base.dtype
        )
        if scale.shape != base.shape[:-1] + (1,):
            raise ValueError("guidance_scale tensor must have shape [..., 1]")
    else:
        scale = torch.tensor(float(guidance_scale), device=base.device)
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError("guidance_scale must be finite and non-negative")
    return F.log_softmax(base_logprobs + scale * guide_logprobs, dim=-1)
