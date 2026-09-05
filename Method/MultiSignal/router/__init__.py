"""Local minimal router package for MultiSignal."""

from .adapters.registry import get_adapter

__all__ = ["get_adapter"]
