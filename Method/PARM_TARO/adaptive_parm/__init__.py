"""Adaptive PARM Fusion: calibrated trust interpolation in log-policy space."""

from .config import AdaptivePARMFusionConfig
from .fusion import (
    adaptive_parm_distribution,
    adaptive_parm_logits,
    adaptive_parm_logprobs,
    lambda_to_weight,
)

__all__ = [
    "AdaptivePARMFusionConfig",
    "adaptive_parm_distribution",
    "adaptive_parm_logits",
    "adaptive_parm_logprobs",
    "lambda_to_weight",
]
