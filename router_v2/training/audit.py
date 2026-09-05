"""Frozen artifact and optimizer-boundary audits for Stage 5."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from router_v2.cache.io import sha256_file
from router_v2.guide_model.provenance import checkpoint_descriptor


def hash_tree(root: str | Path) -> dict[str, Any]:
    path_root = Path(root).resolve()
    records = []
    file_count = 0
    total_bytes = 0
    for path in sorted(
        (item for item in path_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path_root).as_posix(),
    ):
        relative = path.relative_to(path_root).as_posix()
        size = path.stat().st_size
        records.append(f"{relative}\t{size}\t{sha256_file(path)}\n")
        file_count += 1
        total_bytes += size
    return {
        "path": str(path_root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(
            "".join(records).encode("utf-8")
        ).hexdigest(),
    }


def snapshot_frozen_artifacts(
    *,
    base_model_path: str | Path,
    guide_model_path: str | Path,
    parm_path: str | Path,
) -> dict[str, Any]:
    return {
        "base_model": checkpoint_descriptor(Path(base_model_path)),
        "guide_model": checkpoint_descriptor(Path(guide_model_path)),
        "parm": hash_tree(parm_path),
    }


def compare_frozen_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "base_model_unchanged": before["base_model"] == after["base_model"],
        "guide_model_unchanged": before["guide_model"] == after["guide_model"],
        "parm_unchanged": before["parm"] == after["parm"],
    }
    return {
        "before": before,
        "after": after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def assert_frozen_models(*models: nn.Module) -> None:
    trainable = [
        name
        for model_index, model in enumerate(models)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise ValueError(f"Frozen model parameters require gradients: {trainable[:5]}")


def assert_optimizer_contains_only(
    optimizer: torch.optim.Optimizer,
    allowed_parameters: Iterable[nn.Parameter],
) -> None:
    allowed_ids = {id(parameter) for parameter in allowed_parameters}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if not optimizer_parameters:
        raise ValueError("Optimizer contains no parameters")
    unexpected = [
        parameter for parameter in optimizer_parameters if id(parameter) not in allowed_ids
    ]
    if unexpected:
        raise ValueError("Optimizer contains non-router parameters")
    if len({id(parameter) for parameter in optimizer_parameters}) != len(
        optimizer_parameters
    ):
        raise ValueError("Optimizer contains duplicate parameters")
