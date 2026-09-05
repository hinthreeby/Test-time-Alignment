#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.MultiSignal.models.controller import MultiSignalController
from Method.MultiSignal.core.fusion import fuse_scores
from Method.MultiSignal.core.cache import SIGNALS

CONFIG_PATH = PROJECT_ROOT / "Method" / "MultiSignal" / "configs" / "default.json"


def collate(batch):
    return {
        "lm_logits": torch.stack([x["lm_logits"] for x in batch]),
        "signal_scores": torch.stack([x["signal_scores"] for x in batch]),
        "gold_index": torch.tensor([x["gold_index"] for x in batch], dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="dataset/multisignal_cache/train.pt")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not data:
        raise RuntimeError(f"Cache is empty: {cache_path}")

    num_workers = config["training"].get("num_workers", 0)
    if num_workers < 0:
        num_workers = min(4, os.cpu_count() or 1)

    loader = DataLoader(
        data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    num_signals = data[0]["signal_scores"].shape[-1]
    signal_costs = config["controller"].get("signal_costs", [1.0] * num_signals)
    controller = MultiSignalController(
        num_signals=num_signals,
        hidden_dim=config["controller"]["hidden_dim"],
        signal_costs=signal_costs[:num_signals],
    ).to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    use_amp = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Cache: {cache_path}")
    print(f"Rows: {len(data)}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}; workers: {num_workers}; hidden_dim: {config['controller']['hidden_dim']}")

    for epoch in range(1, config["training"]["epochs"] + 1):
        controller.train()
        losses = []
        weight_sum = torch.zeros(num_signals)
        trust_sum = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            lm_logits = batch["lm_logits"].to(device)
            signal_scores = batch["signal_scores"].to(device)
            gold_index = batch["gold_index"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                output = controller(lm_logits, signal_scores)
                final_scores, _ = fuse_scores(lm_logits, signal_scores, output)
                loss = F.cross_entropy(final_scores, gold_index)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().item()))
            weight_sum += output["weights"].detach().cpu().sum(dim=0)
            trust_sum += float(output["trust"].detach().cpu().sum())

        mean_weights = weight_sum / len(data)
        mean_trust = trust_sum / len(data)
        labels = [name for name, _ in SIGNALS[:num_signals]]
        weights_text = ", ".join(f"{name}={weight:.3f}" for name, weight in zip(labels, mean_weights.tolist()))
        print(f"Epoch {epoch}: loss={sum(losses) / len(losses):.6f}")
        print(f"Mean weights: {weights_text}; trust={mean_trust:.3f}; base={1.0 - mean_trust:.3f}")

    output_path = PROJECT_ROOT / "Method" / "MultiSignal" / "checkpoints" / "controller_4signal.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "controller_state_dict": controller.state_dict(),
        "config": config,
        "signals": [name for name, _ in SIGNALS[:num_signals]],
        "signal_costs": signal_costs[:num_signals],
    }, output_path)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
