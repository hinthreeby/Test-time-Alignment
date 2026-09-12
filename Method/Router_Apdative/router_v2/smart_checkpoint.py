"""Separate atomic checkpoint format for Smart Router V2."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter


SMART_CHECKPOINT_FORMAT = "router_v2.smart_router_checkpoint"
SMART_CHECKPOINT_SCHEMA_VERSION = 1


def build_smart_checkpoint_payload(
    model: SmartTokenRouter,
    *,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = model.config
    return {
        "schema_version": SMART_CHECKPOINT_SCHEMA_VERSION,
        "format": SMART_CHECKPOINT_FORMAT,
        "method_label": config.method_label,
        "model_class": "SmartTokenRouter",
        "architecture": config.architecture,
        "variant": config.variant,
        "feature_toggles": config.feature_toggles,
        "lambda_max": config.lambda_max,
        "config": config.to_dict(),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "training_state": dict(training_state or {}),
        "metadata": dict(metadata or {}),
    }


def validate_smart_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    version = payload.get("schema_version")
    if version != SMART_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Incompatible Smart Router checkpoint schema_version: "
            f"expected {SMART_CHECKPOINT_SCHEMA_VERSION}, got {version}"
        )
    required = {
        "schema_version",
        "format",
        "method_label",
        "model_class",
        "architecture",
        "variant",
        "feature_toggles",
        "lambda_max",
        "config",
        "state_dict",
        "training_state",
        "metadata",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(
            f"Invalid Smart Router checkpoint keys; "
            f"missing={missing}, unknown={unknown}"
        )
    if payload["format"] != SMART_CHECKPOINT_FORMAT:
        raise ValueError("Invalid Smart Router checkpoint format")
    if payload["method_label"] != "SMART_ROUTER_V2":
        raise ValueError("Invalid Smart Router method_label")
    if payload["model_class"] != "SmartTokenRouter":
        raise ValueError("Checkpoint does not contain SmartTokenRouter")
    for field in ("config", "state_dict", "training_state", "metadata"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"Checkpoint field {field!r} must be a mapping")
    config = SmartRouterConfig.from_dict(payload["config"])
    mirrored = {
        "architecture": config.architecture,
        "variant": config.variant,
        "feature_toggles": config.feature_toggles,
        "lambda_max": config.lambda_max,
    }
    for name, expected in mirrored.items():
        if payload[name] != expected:
            raise ValueError(
                f"Checkpoint field {name!r} does not match embedded config"
            )


def save_smart_checkpoint(
    model: SmartTokenRouter,
    path: str | Path,
    *,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite Smart Router checkpoint: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_smart_checkpoint_payload(
        model,
        training_state=training_state,
        metadata=metadata,
    )
    validate_smart_checkpoint_payload(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite Smart Router checkpoint: {output_path}"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_smart_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SmartTokenRouter, dict[str, Any]]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Smart Router checkpoint payload must be a mapping")
    validate_smart_checkpoint_payload(payload)
    target_device = torch.device(map_location)
    model = SmartTokenRouter(
        SmartRouterConfig.from_dict(payload["config"])
    ).to(target_device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
