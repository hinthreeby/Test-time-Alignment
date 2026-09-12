"""Runtime device selection with an explicit CPU fallback policy."""

from __future__ import annotations

import warnings

import torch


def resolve_device(
    requested: str = "auto",
    *,
    allow_cpu_fallback: bool = True,
) -> torch.device:
    """Resolve auto/cuda/cpu while preferring CUDA whenever it is available."""

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda" and not allow_cpu_fallback:
        raise RuntimeError("CUDA was requested but is not available in this runtime")
    if requested == "cuda":
        warnings.warn(
            "CUDA is unavailable in this runtime; falling back to CPU",
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.device("cpu")
