from __future__ import annotations

from typing import Dict, Callable, Any

from Method.MultiSignal.router.adapters.fallback_adapter import RADAdapter, GenARMAdapter, CDQAdapter, ARGSAdapter

_ADAPTERS: Dict[str, Callable[..., Any]] = {
    "rad": RADAdapter,
    "genarm": GenARMAdapter,
    "cdq": CDQAdapter,
    "args": ARGSAdapter,
}


def register_adapter(name: str):
    def decorator(cls):
        _ADAPTERS[name.lower()] = cls
        return cls
    return decorator


def get_adapter(name: str):
    key = name.lower()
    if key in _ADAPTERS:
        return _ADAPTERS[key]
    raise RuntimeError(
        f"No adapter registered for '{name}'. "
        "Add an adapter implementation under Method/MultiSignal/router/adapters/."
    )


def list_adapters():
    return sorted(_ADAPTERS)
