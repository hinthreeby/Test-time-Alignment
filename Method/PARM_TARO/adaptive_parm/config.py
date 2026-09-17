"""Configuration for the frozen Adaptive-PARM fusion formulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptivePARMFusionConfig:
    """Numerical configuration; the fusion equation itself is not configurable."""

    validate_inputs: bool = True
    probability_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.probability_dtype not in {"input", "float32"}:
            raise ValueError("probability_dtype must be 'input' or 'float32'")
