#!/usr/bin/env python3
"""Train ObjectiveTracker from cache-correct dynamic-alpha oracle probes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


CAP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP_DIR))
from src.models.objective_tracker import ObjectiveTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-data", type=Path, required=True)
    parser.add_argument("--ckpt-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        parser.error("epochs, batch-size and patience must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    return args


def load_examples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = []
    for row in prompts:
        if row.get("schema_version") != 2 or row.get("probe_cache_policy") != "full_prefix_replay_per_alpha":
            raise ValueError(
                "ObjectiveTracker requires schema v2 full-prefix probes. "
                "Rebuild with build_dynamic_alpha_oracle.py; legacy cache-reuse data is invalid."
            )
        prompt_id = str(row.get("sample_id") or row.get("prompt"))
        for trace in row.get("trace", []):
            if "tracker_features" in trace and "reward_targets" in trace:
                examples.append({
                    "sample_id": prompt_id,
                    "features": trace["tracker_features"],
                    "targets": trace["reward_targets"],
                })
    if not examples:
        raise ValueError("No tracker_features/reward_targets found in oracle data")
    return examples


def split_groups(rows: list[dict[str, Any]], fraction: float, seed: int):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two prompts for grouped validation")
    random.Random(seed).shuffle(keys)
    count = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
    validation = set(keys[:count])
    return (
        [row for key in keys if key not in validation for row in groups[key]],
        [row for key in keys if key in validation for row in groups[key]],
        sorted(validation),
    )


def as_tensors(rows: list[dict[str, Any]]):
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["targets"] for row in rows], dtype=torch.float32)
    if features.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Invalid tracker feature/target shapes")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Tracker dataset contains NaN or Inf")
    return features, targets


def evaluate(model, features, raw_targets, normalized_targets, device):
    model.eval()
    with torch.no_grad():
        predicted_normalized, log_var = model.normalized_outputs(features.to(device))
        predicted = predicted_normalized * model.target_std + model.target_mean
        mae = (predicted.cpu() - raw_targets).abs().mean(dim=0)
        error = predicted_normalized - normalized_targets.to(device)
        nll = 0.5 * (error.square().mean(dim=-1) * (-log_var).exp() + log_var)
    return {"loss": float(nll.mean()), "mae_by_objective": mae.tolist()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    rows = load_examples(args.oracle_data)
    train_rows, validation_rows, validation_groups = split_groups(
        rows, args.validation_fraction, args.seed
    )
    train_x, train_y = as_tensors(train_rows)
    validation_x, validation_y = as_tensors(validation_rows)
    top_k = train_x.shape[1] - 5
    if top_k < 2:
        raise ValueError(f"Cannot infer a valid top_k from feature dimension {train_x.shape[1]}")
    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_y_normalized = (train_y - mean) / std
    validation_y_normalized = (validation_y - mean) / std

    model = ObjectiveTracker(top_k=top_k, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    model.set_target_normalization(mean.to(device), std.to(device))
    loader = DataLoader(
        TensorDataset(train_x, train_y_normalized), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None; best_loss = math.inf; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, log_var = model.normalized_outputs(features)
            error = prediction - targets
            loss = 0.5 * (error.square().mean(dim=-1) * (-log_var).exp() + log_var)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_x, validation_y, validation_y_normalized, device)
        metrics["epoch"] = epoch; history.append(metrics)
        if metrics["loss"] < best_loss - 1e-8:
            best_loss = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_x, train_y, train_y_normalized, device)
    validation_metrics = evaluate(model, validation_x, validation_y, validation_y_normalized, device)
    metadata = {
        "schema_version": 2, "seed": args.seed,
        "oracle_data": str(args.oracle_data.resolve()),
        "train_examples": len(train_rows), "validation_examples": len(validation_rows),
        "validation_groups": validation_groups, "epochs_completed": len(history),
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
    }
    model.cpu().save(str(args.ckpt_out), metadata=metadata)
    report = args.ckpt_out.with_suffix(args.ckpt_out.suffix + ".metrics.json")
    report.write_text(json.dumps({**metadata, "history": history}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.ckpt_out), **metadata}, indent=2))


if __name__ == "__main__":
    main()
