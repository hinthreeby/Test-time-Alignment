#!/usr/bin/env python3
"""Train the binary DynaCAP-PARM TokenRouter on a scored w-oracle JSONL."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
CAP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP_DIR))

from src.models.token_router import TokenRouter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-data", type=Path, required=True)
    parser.add_argument("--ckpt-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--w-fixed", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        parser.error("epochs, batch-size and patience must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    return args


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Oracle dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Oracle dataset is empty: {path}")
    if "trace" in rows[0] and "feature_values" not in rows[0]:
        raise ValueError(
            "This is a dynamic-alpha objective-probe dataset (prompt/response/trace), "
            "not a w-oracle router dataset. Run scripts/build_w_oracle.py first."
        )
    required = {"feature_values", "gate_label", "sample_id", "feature_schema", "top_k"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing required fields: {sorted(missing)}")
    return rows


def grouped_split(rows: list[dict[str, Any]], fraction: float, seed: int):
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sample_id"])].append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two unique sample_id groups for leakage-free validation")
    random.Random(seed).shuffle(keys)
    validation_count = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
    validation_keys = set(keys[:validation_count])
    train = [row for key in keys if key not in validation_keys for row in groups[key]]
    validation = [row for key in keys if key in validation_keys for row in groups[key]]
    return train, validation, sorted(validation_keys)


def tensors(rows: list[dict[str, Any]], expected_dim: int):
    features = torch.tensor([row["feature_values"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([float(row["gate_label"]) for row in rows], dtype=torch.float32)
    if features.dim() != 2 or features.shape[1] != expected_dim:
        raise ValueError(f"Expected feature matrix (*, {expected_dim}), got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("Oracle features contain NaN or Inf")
    if not set(labels.tolist()) <= {0.0, 1.0}:
        raise ValueError("gate_label must contain only 0 or 1")
    return features, labels


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.sigmoid()
    predictions = probabilities >= 0.5
    truth = labels.bool()
    tp = int((predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    positives = max(1, int(truth.sum()))
    negatives = max(1, int((~truth).sum()))
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
        "accuracy": float((predictions == truth).float().mean()),
        "balanced_accuracy": 0.5 * (tp / positives + tn / negatives),
        "positive_rate": float(predictions.float().mean()),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    rows = load_rows(args.oracle_data)
    schemas = {str(row["feature_schema"]) for row in rows}
    top_ks = {int(row["top_k"]) for row in rows}
    objective_counts = {int(row.get("num_objectives", 2)) for row in rows}
    if len(schemas) != 1 or len(top_ks) != 1 or len(objective_counts) != 1:
        raise ValueError("All rows must use one feature_schema, top_k and num_objectives")
    schema, top_k, num_objectives = schemas.pop(), top_ks.pop(), objective_counts.pop()

    train_rows, validation_rows, validation_groups = grouped_split(
        rows, args.validation_fraction, args.seed
    )
    model = TokenRouter(
        num_objectives=num_objectives,
        top_k=top_k,
        hidden_dim=args.hidden_dim,
        w_fixed=args.w_fixed,
        mode="binary",
        dropout=args.dropout,
        feature_schema=schema,
    ).to(device)
    train_x, train_y = tensors(train_rows, model.input_dim)
    validation_x, validation_y = tensors(validation_rows, model.input_dim)
    counts = Counter(int(value) for value in train_y.tolist())
    if not counts[0] or not counts[1]:
        raise ValueError(f"Training split needs both classes; got {dict(counts)}")
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.gate_logits_from_features(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.gate_logits_from_features(validation_x.to(device)).cpu()
        epoch_metrics = metrics(val_logits, validation_y)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        if epoch_metrics["loss"] < best_loss - 1e-8:
            best_loss = epoch_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_metrics = metrics(model.gate_logits_from_features(train_x.to(device)).cpu(), train_y)
        validation_metrics = metrics(
            model.gate_logits_from_features(validation_x.to(device)).cpu(), validation_y
        )

    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "oracle_data": str(args.oracle_data.resolve()),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "validation_groups": validation_groups,
        "class_counts_train": dict(counts),
        "epochs_completed": len(history),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    model.cpu().save(str(args.ckpt_out), metadata=metadata)
    report_path = args.ckpt_out.with_suffix(args.ckpt_out.suffix + ".metrics.json")
    report_path.write_text(json.dumps({**metadata, "history": history}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.ckpt_out), **metadata}, indent=2))


if __name__ == "__main__":
    main()
