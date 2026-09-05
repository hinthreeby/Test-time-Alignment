"""Versioned, no-overwrite checkpoint format for faithful TARO Top-K."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from router_v2.config import TARORouterConfig
from router_v2.model import TAROTokenRouter


CHECKPOINT_FORMAT = "router_v2.taro_topk_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 2


def build_checkpoint_payload(
    model: TAROTokenRouter,
    *,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = model.config
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": CHECKPOINT_FORMAT,
        "method_label": "TARO",
        "model_class": "TAROTokenRouter",
        "architecture": config.architecture,
        "mode": config.mode,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "token_embedding_dim": config.token_embedding_dim,
        "hidden_dim": config.hidden_dim,
        "entropy_weight": config.entropy_weight,
        "config": config.to_dict(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "training_state": dict(training_state or {}),
        "metadata": dict(metadata or {}),
    }


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    version = payload.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Incompatible TARO checkpoint schema_version: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, got {version}. "
            "Stage 2 schema-1 checkpoints use the obsolete pooled/GELU architecture."
        )

    required = {
        "schema_version",
        "format",
        "method_label",
        "model_class",
        "architecture",
        "mode",
        "vocab_size",
        "top_k",
        "token_embedding_dim",
        "hidden_dim",
        "entropy_weight",
        "config",
        "state_dict",
        "training_state",
        "metadata",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(f"Invalid checkpoint keys; missing={missing}, unknown={unknown}")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Invalid TARO Top-K checkpoint format")
    if payload["method_label"] != "TARO":
        raise ValueError("Checkpoint method_label must be exactly 'TARO'")
    if payload["model_class"] != "TAROTokenRouter":
        raise ValueError("Checkpoint does not contain a TAROTokenRouter")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    if not isinstance(payload["state_dict"], Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    if not isinstance(payload["training_state"], Mapping):
        raise ValueError("Checkpoint training_state must be a mapping")
    if not isinstance(payload["metadata"], Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")

    config = TARORouterConfig.from_dict(payload["config"])
    mirrored_fields = (
        "architecture",
        "mode",
        "vocab_size",
        "top_k",
        "token_embedding_dim",
        "hidden_dim",
        "entropy_weight",
    )
    for field in mirrored_fields:
        if payload[field] != getattr(config, field):
            raise ValueError(
                f"Checkpoint field {field!r} does not match embedded config"
            )


def save_taro_checkpoint(
    model: TAROTokenRouter,
    path: str | Path,
    *,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        model,
        training_state=training_state,
        metadata=metadata,
    )
    validate_checkpoint_payload(payload)
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
            raise FileExistsError(f"Refusing to overwrite checkpoint: {output_path}")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_taro_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TAROTokenRouter, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("TARO checkpoint payload must be a mapping")
    validate_checkpoint_payload(payload)
    target_device = torch.device(map_location)
    model = TAROTokenRouter(
        TARORouterConfig.from_dict(payload["config"])
    ).to(target_device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
