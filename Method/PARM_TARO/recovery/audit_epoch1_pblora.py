"""CPU-safe integrity audit for the recovered one-epoch PBLoRA artifact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).absolute().parents[2]
TRAINING = ROOT / "results/parm_taro/recovery/reproduced/pblora"
FINAL = TRAINING / "final_checkpoint"
OUT = ROOT / "results/parm_taro/recovery/pblora_repro/epoch1_validation"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    tree = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix()
        sha = file_hash(file)
        size = file.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": sha})
        tree.update(relative.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
    return rows, tree.hexdigest()


def state_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if (value.is_floating_point() or value.is_complex()) else True
    if isinstance(value, dict):
        return all(state_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(state_tensors_finite(item) for item in value)
    return True


def main() -> int:
    if not FINAL.is_dir():
        raise SystemExit(f"Final checkpoint directory is missing: {FINAL}")
    files, tree_sha = tree_inventory(FINAL)
    zero_files = [row["path"] for row in files if row["bytes"] == 0]
    config = json.loads((FINAL / "adapter_config.json").read_text(encoding="utf-8"))

    tensor_count = 0
    parameter_count = 0
    finite = True
    tensor_shapes: dict[str, list[int]] = {}
    weights = FINAL / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            parameter_count += tensor.numel()
            finite = finite and bool(torch.isfinite(tensor).all())
            tensor_shapes[key] = list(tensor.shape)

    states = []
    for directory in sorted(TRAINING.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        state_file = directory / "trainer_state.json"
        if not state_file.is_file():
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        required = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        states.append({
            "path": str(directory.resolve()),
            "global_step": int(state["global_step"]),
            "epoch": float(state["epoch"]),
            "max_steps": int(state["max_steps"]),
            "required_state_files_present": all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in required),
        })

    recovery = json.loads((TRAINING / "recovery_training_audit.json").read_text(encoding="utf-8"))
    preflight = json.loads((TRAINING / "recovery_preflight.json").read_text(encoding="utf-8"))
    completed_step = int(recovery["global_step"])
    expected_step = max(item["max_steps"] for item in states)
    final_epoch = completed_step / expected_step
    root_same_as_final = file_hash(TRAINING / "adapter_model.safetensors") == file_hash(weights)
    latest_periodic = Path(states[-1]["path"])
    optimizer_state = torch.load(latest_periodic / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler_state = torch.load(latest_periodic / "scheduler.pt", map_location="cpu", weights_only=True)
    # This is a locally generated trusted checkpoint. RNG state includes NumPy
    # objects that are intentionally outside torch's weights-only allowlist.
    rng_state = torch.load(latest_periodic / "rng_state.pth", map_location="cpu", weights_only=False)
    resumable_state_loadable = all(state_tensors_finite(value) for value in (optimizer_state, scheduler_state, rng_state))
    config_valid = (
        str(config.get("peft_type", "")).upper() == "PBLORA"
        and config.get("obj_num") == 2
        and config.get("r1") == 4
        and config.get("r2") == 4
        and float(config.get("lora_alpha")) == 8.0
        and float(config.get("lora_dropout")) == 0.05
        and set(config.get("target_modules", [])) == {"q_proj", "k_proj", "v_proj"}
    )
    checkpoint_ok = bool(
        files and not zero_files and finite and parameter_count == 6_296_064
        and tensor_count == 384 and config_valid and root_same_as_final
        and completed_step == expected_step and abs(final_epoch - 1.0) < 1e-12
        and states and all(item["required_state_files_present"] for item in states)
        and resumable_state_loadable
        and preflight.get("sampled_base_state_unchanged") is True
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if checkpoint_ok else "FAIL",
        "training_output": str(TRAINING.resolve()),
        "final_checkpoint": str(FINAL.resolve()),
        "latest_completed_global_step": completed_step,
        "expected_total_steps": expected_step,
        "final_epoch": final_epoch,
        "latest_resumable_checkpoint": states[-1]["path"],
        "latest_resumable_global_step": states[-1]["global_step"],
        "latest_resumable_state_loadable_and_finite": resumable_state_loadable,
        "periodic_checkpoints": states,
        "files": files,
        "final_tree_sha256": tree_sha,
        "zero_byte_files": zero_files,
        "adapter_config": config,
        "tensor_count": tensor_count,
        "trainable_parameter_count": parameter_count,
        "all_adapter_tensors_finite": finite,
        "structure_matches_expected": config_valid,
        "root_and_final_weights_identical": root_same_as_final,
        "base_frozen_during_training": preflight.get("all_non_pblora_parameters_frozen"),
        "sampled_base_state_unchanged": preflight.get("sampled_base_state_unchanged"),
        "adapter_payload_reload": "PASS_SAFE_TENSORS_CPU",
        "full_4bit_model_reload": "PENDING_ALPHA_PROBE",
        "note": "The final adapter is step 240. checkpoint-200 is only the latest resumable optimizer checkpoint because save_steps=50.",
    }
    atomic_text(OUT / "checkpoint_audit.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    hash_lines = [f"# tree_sha256 {tree_sha}  {FINAL}"]
    hash_lines += [f"{row['sha256']}  {row['bytes']:>10}  {row['path']}" for row in files]
    atomic_text(OUT / "checkpoint_hashes.txt", "\n".join(hash_lines) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "final_checkpoint", "latest_completed_global_step", "expected_total_steps", "final_epoch", "trainable_parameter_count", "final_tree_sha256")}, indent=2))
    return 0 if checkpoint_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
