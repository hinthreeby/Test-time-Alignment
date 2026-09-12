"""Deterministic batching and atomic LoRA checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch


def cast_trainable_parameters_to_fp32(model: Any) -> list[torch.nn.Parameter]:
    """Keep frozen model weights untouched while promoting adapter weights."""

    trainable_parameters = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter.data = parameter.data.to(dtype=torch.float32)
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"Trainable parameter {parameter_name} is not FP32: "
                f"{parameter.dtype}"
            )
        trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters found")
    return trainable_parameters


def assert_optimizer_parameters_fp32(
    optimizer: torch.optim.Optimizer,
) -> None:
    non_fp32 = [
        parameter.dtype
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.dtype != torch.float32
    ]
    if non_fp32:
        raise TypeError(
            "Every optimizer parameter must be FP32; found "
            f"{sorted({str(dtype) for dtype in non_fp32})}"
        )


def canonical_config_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_epoch_batches(
    *,
    num_samples: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(num_samples, generator=generator).tolist()
    return [
        order[start : start + batch_size]
        for start in range(0, num_samples, batch_size)
    ]


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    checkpoints = []
    for path in root.glob("checkpoint-step-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]


def save_adapter_directory_atomic(
    *,
    model: Any,
    tokenizer: Any,
    destination: str | Path,
    training_state: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite guide checkpoint: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        torch.save(training_state, temporary / "training_state.pt")
        with (temporary / "training_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def cleanup_stale_checkpoint_temps(output_dir: str | Path) -> list[str]:
    root = Path(output_dir)
    removed = []
    if not root.exists():
        return removed
    for path in sorted(root.glob(".checkpoint-step-*.tmp")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path.name)
    for path in sorted(root.glob(".final_adapter.*.tmp")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path.name)
    return removed
