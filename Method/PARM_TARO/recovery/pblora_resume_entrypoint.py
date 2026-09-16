"""Recovery-only launcher for the protected PARM PBLoRA trainer.

The historical trainer calls ``trainer.train()`` without a resume argument and
has no arbitrary subset switch. This launcher adds those operational features
at runtime; it never edits ``PARM/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    configured = os.environ.get("TTA_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / ".git").exists():
            return root
        raise RuntimeError(f"TTA_PROJECT_ROOT is not a project root: {root}")
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Cannot locate project root containing .git")


ROOT = find_project_root(Path(__file__).resolve().parent)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_symlink(target: Path, alias: Path) -> None:
    """Replace only the small alias; never delete a checkpoint directory."""
    alias.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias.with_name(f".{alias.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
    os.replace(temporary, alias)


def install_subset_hook(limit: int | None) -> None:
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("--train-subset must be positive")
    import datasets

    original = datasets.load_dataset

    def load_dataset_with_subset(*positional: Any, **keywords: Any) -> Any:
        result = original(*positional, **keywords)
        data_files = keywords.get("data_files")
        text = str(data_files)
        if "train.json" in text:
            return result.select(range(min(limit, len(result))))
        return result

    datasets.load_dataset = load_dataset_with_subset


def sampled_base_fingerprint(model: Any) -> str:
    """Cheap mutation sentinel over frozen non-PBLoRA tensors.

    This is a sampled in-memory fingerprint. Immutable base model files are
    verified independently by their recorded shard hashes.
    """
    digest = hashlib.sha256()
    base = [(name, value) for name, value in model.named_parameters() if "pblora_" not in name]
    if not base:
        return digest.hexdigest()
    stride = max(1, len(base) // 32)
    for name, value in base[::stride][:32]:
        flat = value.detach().reshape(-1)
        indexes = sorted({0, max(0, flat.numel() // 2), max(0, flat.numel() - 1)})
        sample = flat[indexes].float().cpu().numpy().tobytes() if flat.numel() else b""
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(sample)
    return digest.hexdigest()


def install_trainer_hooks(output_dir: Path, resume_checkpoint: str | None) -> None:
    import torch
    from transformers import Trainer, TrainerCallback

    original_train = Trainer.train
    original_save = Trainer._save_checkpoint

    class RecoveryAuditCallback(TrainerCallback):
        def __init__(self) -> None:
            self.losses: list[float] = []
            self.nonzero_gradient_observations = 0
            self.max_gradient_norm = 0.0

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            if logs and isinstance(logs.get("loss"), (int, float)):
                self.losses.append(float(logs["loss"]))
            self._write(state)

        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
        ) -> None:
            if model is not None:
                norms = []
                for name, parameter in model.named_parameters():
                    if "pblora_" in name and parameter.grad is not None:
                        value = float(torch.linalg.vector_norm(parameter.grad.detach().float()).cpu())
                        if value > 0 and torch.isfinite(torch.tensor(value)):
                            norms.append(value)
                if norms:
                    self.nonzero_gradient_observations += 1
                    self.max_gradient_norm = max(self.max_gradient_norm, max(norms))
            self._write(state)

        def _write(self, state: Any) -> None:
            atomic_json(
                output_dir / "recovery_training_audit.json",
                {
                    "global_step": int(state.global_step),
                    "losses": self.losses,
                    "losses_finite": all(torch.isfinite(torch.tensor(x)).item() for x in self.losses),
                    "loss_directional_decrease": len(self.losses) >= 2 and self.losses[-1] < self.losses[0],
                    "nonzero_gradient_observations": self.nonzero_gradient_observations,
                    "max_gradient_norm": self.max_gradient_norm,
                },
            )

    def train_with_recovery(self: Any, *positional: Any, **keywords: Any) -> Any:
        if resume_checkpoint:
            keywords.setdefault("resume_from_checkpoint", resume_checkpoint)

        trainable = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        unexpected = [name for name, _ in trainable if "pblora_" not in name]
        module_counts = {
            field: sum(field in name for name, _ in trainable)
            for field in ("pblora_A", "pblora_B", "pblora_W1", "pblora_W2")
        }
        base_fingerprint_before = sampled_base_fingerprint(self.model)
        atomic_json(
            output_dir / "recovery_preflight.json",
            {
                "status": "PASS" if trainable and not unexpected else "FAIL",
                "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
                "trainable_tensor_count": len(trainable),
                "expected_tulu_7b_r4_parameter_count": 6_296_064,
                "pblora_tensor_counts": module_counts,
                "unexpected_trainable_parameters": unexpected,
                "all_non_pblora_parameters_frozen": not unexpected,
                "sampled_base_fingerprint_before": base_fingerprint_before,
                "resume_from_checkpoint": resume_checkpoint,
            },
        )
        if not trainable or unexpected:
            raise RuntimeError("PBLoRA preflight failed: base is not fully frozen or adapter is absent")

        self.add_callback(RecoveryAuditCallback())
        result = original_train(self, *positional, **keywords)
        base_fingerprint_after = sampled_base_fingerprint(self.model)
        preflight_path = output_dir / "recovery_preflight.json"
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        payload["sampled_base_fingerprint_after"] = base_fingerprint_after
        payload["sampled_base_state_unchanged"] = base_fingerprint_before == base_fingerprint_after
        atomic_json(preflight_path, payload)
        return result

    def save_with_atomic_pointers(self: Any, *positional: Any, **keywords: Any) -> Any:
        result = original_save(self, *positional, **keywords)
        checkpoint = output_dir / f"checkpoint-{self.state.global_step}"
        if checkpoint.is_dir() and (checkpoint / "trainer_state.json").is_file():
            pointer = {"checkpoint": str(checkpoint.resolve()), "global_step": int(self.state.global_step)}
            atomic_json(output_dir / "last_checkpoint.json", pointer)
            atomic_symlink(checkpoint, output_dir / "last")

            eval_losses = [
                float(item["eval_loss"])
                for item in self.state.log_history
                if isinstance(item.get("eval_loss"), (int, float))
            ]
            if eval_losses:
                candidate = {**pointer, "metric": "eval_loss", "value": eval_losses[-1]}
                best_path = output_dir / "best_checkpoint.json"
                old = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else None
                if old is None or candidate["value"] < old["value"]:
                    atomic_json(best_path, candidate)
                    atomic_symlink(checkpoint, output_dir / "best")
        return result

    Trainer.train = train_with_recovery
    Trainer._save_checkpoint = save_with_atomic_pointers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--recovery-output-dir", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)

    training_dir = ROOT / "PARM/code/training"
    vendored_peft = ROOT / "PARM/peft/src"
    sys.path.insert(0, str(training_dir))
    sys.path.insert(0, str(vendored_peft))

    output_dir = Path(args.recovery_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    install_subset_hook(args.train_subset)
    install_trainer_hooks(output_dir, args.resume_from_checkpoint)

    script = training_dir / "train_pref_arm.py"
    sys.argv = [str(script), *trainer_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
