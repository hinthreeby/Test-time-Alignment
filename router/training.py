"""Training utilities for the minimal RAD token router."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

from router.features import compute_guided_scores
from router.model import RADTokenRouter

LOGGER = logging.getLogger("router_training")


@dataclass(frozen=True)
class RouterTrainingConfig:
    train_manifest: str
    validation_manifest: str
    output_dir: str
    beta_max: float
    beta_init: float | None
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    gradient_clip_norm: float
    early_stopping_patience: int
    seed: int
    fixed_beta_baselines: list[float]
    near_zero_fraction: float
    near_max_fraction: float
    device: str


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return output_dir / "last.pt", output_dir / "best.pt", output_dir / "training_log.json"


def optimizer_contains_only_router(optimizer: torch.optim.Optimizer, router: RADTokenRouter) -> bool:
    router_param_ids = {id(parameter) for parameter in router.parameters()}
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return optimizer_param_ids == router_param_ids


def assert_no_external_params_in_optimizer(
    optimizer: torch.optim.Optimizer,
    router: RADTokenRouter,
    external_models: Sequence[torch.nn.Module] | None = None,
) -> None:
    assert optimizer_contains_only_router(optimizer, router), "Optimizer must contain only router parameters"
    if external_models:
        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        external_param_ids = {
            id(parameter)
            for model in external_models
            for parameter in model.parameters()
        }
        assert optimizer_param_ids.isdisjoint(external_param_ids), "Optimizer contains frozen base/RM parameters"


class FeatureShardDataset:
    """Lazy iterable over tensor feature shards."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = load_json(manifest_path)
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"No shards listed in {manifest_path}")
        self.top_k = int(self.manifest.get("top_k", 20))
        if self.top_k != 20:
            raise ValueError(f"Expected top_k=20, got {self.top_k}")
        self.num_records = int(self.manifest.get("cached_token_steps", 0))
        if self.num_records <= 0:
            self.num_records = sum(int(shard["num_records"]) for shard in self.shards)
        for shard in self.shards:
            path = Path(shard["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing shard: {path}")
            expected_hash = shard.get("sha256")
            if expected_hash and sha256_file(path) != expected_hash:
                raise ValueError(f"Shard hash mismatch: {path}")

    def iter_batches(
        self,
        batch_size: int,
        device: torch.device,
        shuffle: bool,
        seed: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        shard_order = list(range(len(self.shards)))
        rng = random.Random(seed)
        if shuffle:
            rng.shuffle(shard_order)
        for shard_position in shard_order:
            shard = self.shards[shard_position]
            payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
            required = {"base_logits", "rad_reward_scores", "gold_index"}
            missing = required - set(payload)
            if missing:
                raise KeyError(f"Shard {shard['path']} missing required fields: {sorted(missing)}")
            base_logits = payload["base_logits"].float()
            reward_scores = payload["rad_reward_scores"].float()
            gold_index = payload["gold_index"].long()
            assert base_logits.shape == reward_scores.shape
            assert base_logits.shape[-1] == 20
            assert gold_index.ndim == 1
            row_indices = list(range(base_logits.shape[0]))
            if shuffle:
                rng.shuffle(row_indices)
            for start in range(0, len(row_indices), batch_size):
                selected = torch.tensor(row_indices[start : start + batch_size], dtype=torch.long)
                yield {
                    "base_logits": base_logits.index_select(0, selected).to(device),
                    "rad_reward_scores": reward_scores.index_select(0, selected).to(device),
                    "gold_index": gold_index.index_select(0, selected).to(device),
                }


def candidate_nll_from_scores(guided_scores: torch.Tensor, gold_index: torch.Tensor) -> torch.Tensor:
    assert guided_scores.ndim == 2 and guided_scores.shape[-1] == 20
    assert gold_index.ndim == 1 and gold_index.shape[0] == guided_scores.shape[0]
    return F.cross_entropy(guided_scores.float(), gold_index.long(), reduction="mean")


def fixed_beta_metrics(dataset: FeatureShardDataset, beta: float, batch_size: int, device: torch.device) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total = 0
    beta_tensor_cache: dict[int, torch.Tensor] = {}
    for batch in dataset.iter_batches(batch_size=batch_size, device=device, shuffle=False, seed=0):
        batch_size_actual = int(batch["gold_index"].shape[0])
        if batch_size_actual not in beta_tensor_cache:
            beta_tensor_cache[batch_size_actual] = torch.full((batch_size_actual, 1), float(beta), device=device)
        guided = compute_guided_scores(batch["base_logits"], batch["rad_reward_scores"], beta_tensor_cache[batch_size_actual])
        loss = candidate_nll_from_scores(guided, batch["gold_index"])
        total_loss += float(loss.item()) * batch_size_actual
        total_correct += int((guided.argmax(dim=-1) == batch["gold_index"]).sum().item())
        total += batch_size_actual
    return {"nll": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


def beta_summary(
    beta_values: list[float],
    gate_values: list[float],
    beta_max: float,
    near_zero_threshold: float,
    near_max_threshold: float,
) -> dict[str, float | None]:
    if not beta_values:
        return {
            "beta_mean": None,
            "beta_std": None,
            "beta_min": None,
            "beta_max": None,
            "beta_near_zero_fraction": None,
            "beta_near_max_fraction": None,
            "gate_mean": None,
            "gate_std": None,
        }
    beta_tensor = torch.tensor(beta_values, dtype=torch.float)
    gate_tensor = torch.tensor(gate_values, dtype=torch.float)
    return {
        "beta_mean": float(beta_tensor.mean().item()),
        "beta_std": float(beta_tensor.std(unbiased=False).item()),
        "beta_min": float(beta_tensor.min().item()),
        "beta_max": float(beta_tensor.max().item()),
        "beta_near_zero_fraction": float((beta_tensor < near_zero_threshold * beta_max).float().mean().item()),
        "beta_near_max_fraction": float((beta_tensor > near_max_threshold * beta_max).float().mean().item()),
        "gate_mean": float(gate_tensor.mean().item()),
        "gate_std": float(gate_tensor.std(unbiased=False).item()),
    }


def run_epoch(
    router: RADTokenRouter,
    dataset: FeatureShardDataset,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    seed: int,
    near_zero_threshold: float,
    near_max_threshold: float,
) -> dict[str, float | None]:
    training = optimizer is not None
    router.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    beta_values: list[float] = []
    gate_values: list[float] = []
    grad_norm_values: list[float] = []
    optimizer_steps = 0
    for batch in dataset.iter_batches(batch_size=batch_size, device=device, shuffle=training, seed=seed):
        if training:
            optimizer.zero_grad(set_to_none=True)
        beta, gate = router(batch["base_logits"], batch["rad_reward_scores"])
        guided = compute_guided_scores(batch["base_logits"], batch["rad_reward_scores"], beta)
        loss = candidate_nll_from_scores(guided, batch["gold_index"])
        if training:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), gradient_clip_norm)
            grad_norm_values.append(float(grad_norm.item()))
            optimizer.step()
            optimizer_steps += 1
        batch_count = int(batch["gold_index"].shape[0])
        total_loss += float(loss.item()) * batch_count
        total_correct += int((guided.argmax(dim=-1) == batch["gold_index"]).sum().item())
        total += batch_count
        beta_values.extend(float(value) for value in beta.detach().cpu().flatten().tolist())
        gate_values.extend(float(value) for value in gate.detach().cpu().flatten().tolist())
    metrics = {
        "nll": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "examples": total,
        "optimizer_steps": optimizer_steps,
        "gradient_norm": sum(grad_norm_values) / len(grad_norm_values) if grad_norm_values else None,
    }
    metrics.update(beta_summary(beta_values, gate_values, router.beta_max, near_zero_threshold, near_max_threshold))
    return metrics


def save_training_checkpoint(
    path: Path,
    router: RADTokenRouter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_nll: float,
    patience_bad_epochs: int,
    config: RouterTrainingConfig,
    dataset_manifest_hashes: dict[str, str],
    model_identifiers: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "router_state_dict": router.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_nll": best_validation_nll,
        "patience_bad_epochs": patience_bad_epochs,
        "config": asdict(config),
        "dataset_manifest_hashes": dataset_manifest_hashes,
        "model_identifiers": model_identifiers,
        "history": history,
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    router: RADTokenRouter,
    optimizer: torch.optim.Optimizer,
    expected_config: RouterTrainingConfig,
    expected_manifest_hashes: dict[str, str],
) -> tuple[int, int, float, int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_config = dict(payload["config"])
    current_config = asdict(expected_config)
    comparable_keys = sorted(set(stored_config) & set(current_config) - {"epochs"})
    mismatches = [
        key for key in comparable_keys
        if stored_config[key] != current_config[key]
    ]
    if mismatches:
        raise ValueError("Checkpoint config is incompatible with current training config")
    if int(current_config["epochs"]) < int(stored_config["epochs"]):
        raise ValueError("Resume epochs cannot be less than the checkpoint's original epoch budget")
    if payload["dataset_manifest_hashes"] != expected_manifest_hashes:
        raise ValueError("Checkpoint dataset manifest hashes differ from current inputs")
    router.load_state_dict(payload["router_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return (
        int(payload["epoch"]),
        int(payload["global_step"]),
        float(payload["best_validation_nll"]),
        int(payload.get("patience_bad_epochs", 0)),
        list(payload.get("history", [])),
    )


def train_router(config: RouterTrainingConfig, resume: bool = False) -> dict[str, Any]:
    set_deterministic_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    last_path, best_path, log_path = checkpoint_paths(output_dir)
    train_manifest_path = Path(config.train_manifest)
    validation_manifest_path = Path(config.validation_manifest)
    dataset_hashes = manifest_hashes([train_manifest_path, validation_manifest_path])
    train_dataset = FeatureShardDataset(train_manifest_path)
    validation_dataset = FeatureShardDataset(validation_manifest_path)

    router = RADTokenRouter(beta_max=config.beta_max, beta_init=config.beta_init).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    assert_no_external_params_in_optimizer(optimizer, router)

    start_epoch = 0
    global_step = 0
    best_validation_nll = float("inf")
    patience_bad_epochs = 0
    history: list[dict[str, Any]] = []
    model_identifiers = {
        "router": "RADTokenRouter",
        "train_feature_manifest": str(train_manifest_path),
        "validation_feature_manifest": str(validation_manifest_path),
    }
    if resume and last_path.exists():
        start_epoch, global_step, best_validation_nll, patience_bad_epochs, history = load_training_checkpoint(
            last_path,
            router,
            optimizer,
            config,
            dataset_hashes,
        )
        start_epoch += 1

    fixed_baselines = {
        str(beta): fixed_beta_metrics(validation_dataset, beta, config.batch_size, device)
        for beta in config.fixed_beta_baselines
    }
    with torch.inference_mode():
        resume_start_validation_metrics = run_epoch(
            router,
            validation_dataset,
            config.batch_size,
            device,
            optimizer=None,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed,
            near_zero_threshold=config.near_zero_fraction,
            near_max_threshold=config.near_max_fraction,
        )
    if resume and history and "initial_validation" in history[0]:
        initial_validation_metrics = dict(history[0]["initial_validation"])
    else:
        initial_validation_metrics = resume_start_validation_metrics

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            router,
            train_dataset,
            config.batch_size,
            device,
            optimizer,
            config.gradient_clip_norm,
            config.seed + epoch,
            config.near_zero_fraction,
            config.near_max_fraction,
        )
        global_step += int(train_metrics["optimizer_steps"])
        with torch.inference_mode():
            validation_metrics = run_epoch(
                router,
                validation_dataset,
                config.batch_size,
                device,
                optimizer=None,
                gradient_clip_norm=config.gradient_clip_norm,
                seed=config.seed,
                near_zero_threshold=config.near_zero_fraction,
                near_max_threshold=config.near_max_fraction,
            )
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "fixed_beta_validation_baselines": fixed_baselines,
            "initial_validation": initial_validation_metrics,
            "resume_start_validation": resume_start_validation_metrics if resume else None,
            "patience_bad_epochs": patience_bad_epochs,
            "learning_rate": lr,
            "epoch_duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(epoch_record)
        validation_nll = float(validation_metrics["nll"])
        improved = validation_nll < best_validation_nll - 1e-12
        if improved:
            best_validation_nll = validation_nll
            patience_bad_epochs = 0
            save_training_checkpoint(
                best_path,
                router,
                optimizer,
                epoch,
                global_step,
                best_validation_nll,
                patience_bad_epochs,
                config,
                dataset_hashes,
                model_identifiers,
                history,
            )
        else:
            patience_bad_epochs += 1
        epoch_record["patience_bad_epochs"] = patience_bad_epochs
        save_training_checkpoint(
            last_path,
            router,
            optimizer,
            epoch,
            global_step,
            best_validation_nll,
            patience_bad_epochs,
            config,
            dataset_hashes,
            model_identifiers,
            history,
        )
        write_log = {
            "config": asdict(config),
            "dataset_manifest_hashes": dataset_hashes,
            "best_validation_nll": best_validation_nll,
            "initial_validation_nll": initial_validation_metrics["nll"],
            "resume_start_validation_nll": resume_start_validation_metrics["nll"] if resume else None,
            "patience_bad_epochs": patience_bad_epochs,
            "history": history,
            "early_stopped": False,
        }
        if patience_bad_epochs >= config.early_stopping_patience:
            write_log["early_stopped"] = True
            write_log["stopped_epoch"] = epoch
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(write_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            break
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(write_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "training_log": str(log_path),
        "best_validation_nll": best_validation_nll,
        "history": history,
        "initial_validation": initial_validation_metrics,
        "resume_start_validation": resume_start_validation_metrics if resume else None,
        "fixed_beta_validation_baselines": fixed_baselines,
    }


def parse_fixed_betas(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal RAD token router from offline feature shards.")
    parser.add_argument("--train-manifest", type=Path, default=Path("cache/router_features/train/manifest.json"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("cache/router_features/validation/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/router"))
    parser.add_argument("--beta-max", type=float, default=30.0)
    parser.add_argument("--beta-init", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-beta-baselines", default="0,1,3,10,30")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    beta_init = None if args.beta_init <= 0 else float(args.beta_init)
    config = RouterTrainingConfig(
        train_manifest=str(args.train_manifest),
        validation_manifest=str(args.validation_manifest),
        output_dir=str(args.output_dir),
        beta_max=float(args.beta_max),
        beta_init=beta_init,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        gradient_clip_norm=float(args.gradient_clip_norm),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        fixed_beta_baselines=parse_fixed_betas(args.fixed_beta_baselines),
        near_zero_fraction=0.05,
        near_max_fraction=0.95,
        device=str(args.device),
    )
    result = train_router(config, resume=args.resume)
    print(json.dumps({key: result[key] for key in ["best_validation_nll", "last_checkpoint", "best_checkpoint", "training_log"]}, indent=2))


if __name__ == "__main__":
    main()
