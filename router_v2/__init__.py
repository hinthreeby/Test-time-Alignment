"""Independent TARO and feature-conditioned Smart Router V2 models."""

from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.config import TARORouterConfig
from router_v2.device import resolve_device
from router_v2.model import TARORouterOutput, TAROTokenRouter, build_taro_router
from router_v2.objective import TAROObjective, compute_taro_objective
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import (
    SmartRouterBatch,
    SmartRouterOutput,
    SmartTokenRouter,
    build_smart_router,
)

__all__ = [
    "TAROObjective",
    "TARORouterConfig",
    "TARORouterOutput",
    "TAROTokenRouter",
    "TAROTopKBatch",
    "SmartRouterBatch",
    "SmartRouterConfig",
    "SmartRouterOutput",
    "SmartTokenRouter",
    "build_smart_router",
    "build_taro_router",
    "compute_taro_objective",
    "resolve_device",
    "select_taro_topk",
]
