"""Checkpoint helpers for the minimal RAD token router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from router.model import RADTokenRouter


def save_router_checkpoint(router: RADTokenRouter, path: Path | str, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "class": "RADTokenRouter",
        "config": router.config.__dict__,
        "state_dict": router.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_router_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> RADTokenRouter:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["class"] == "RADTokenRouter"
    config = dict(payload["config"])
    router = RADTokenRouter(**config)
    router.load_state_dict(payload["state_dict"], strict=True)
    return router
