"""Controller-agnostic diagnostics for Base/PARM policy disagreement."""

from __future__ import annotations

import torch
from torch import Tensor

from .utils import validate_log_policy_pair


def policy_disagreement_features(
    base_logprobs: Tensor, parm_logprobs: Tensor
) -> dict[str, Tensor]:
    """Compute finite, prefix-local features without history or future leakage."""
    validate_log_policy_pair(base_logprobs, parm_logprobs)
    a = torch.log_softmax(base_logprobs.float(), dim=-1)
    b = torch.log_softmax(parm_logprobs.float(), dim=-1)
    p = a.exp()
    q = b.exp()
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.log()
    return {
        "base_entropy": -(p * a).sum(dim=-1),
        "parm_entropy": -(q * b).sum(dim=-1),
        "js_divergence": 0.5
        * ((p * (a - log_midpoint)).sum(dim=-1) + (q * (b - log_midpoint)).sum(dim=-1)),
        "top1_agreement": (a.argmax(dim=-1) == b.argmax(dim=-1)).float(),
    }
