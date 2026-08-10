#!/usr/bin/env python3

import argparse
import json
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

CONFIG_PATH = PROJECT_ROOT / "Method" / "MultiSignal" / "configs" / "default.json"


def collate(batch):
    return {
        "lm_logits": torch.stack([x["lm_logits"] for x in batch]),
        "signal_scores": torch.stack([x["signal_scores"] for x in batch]),
        "gold_index": torch.tensor([x["gold_index"] for x in batch], dtype=torch.long)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="dataset/multisignal_cache/train.pt")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(PROJECT_ROOT / args.cache, map_location="cpu", weights_only=False)

    loader = DataLoader(data, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate)

    controller = MultiSignalController(num_signals=4, hidden_dim=config["controller"]["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    for epoch in range(1, config["training"]["epochs"] + 1):
        controller.train()
        losses = []
        weight_sum = torch.zeros(4)

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            lm_logits = batch["lm_logits"].to(device)
            signal_scores = batch["signal_scores"].to(device)
            gold_index = batch["gold_index"].to(device)

            output = controller(lm_logits, signal_scores)
            final_scores, _ = fuse_scores(lm_logits, signal_scores, output)
            loss = F.cross_entropy(final_scores, gold_index)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), config["training"]["grad_clip"])
            optimizer.step()

            losses.append(float(loss.detach().item()))
            weight_sum += output["weights"].detach().cpu().sum(dim=0)

        mean_weights = weight_sum / len(data)
        print(f"Epoch {epoch}: loss={sum(losses) / len(losses):.6f}")
        print(f"Mean weights: GenARM={mean_weights[0]:.3f}, RAD={mean_weights[1]:.3f}, CD-Q={mean_weights[2]:.3f}, ARGS={mean_weights[3]:.3f}")

    output_path = PROJECT_ROOT / "Method" / "MultiSignal" / "checkpoints" / "controller_4signal.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "controller_state_dict": controller.state_dict(),
        "config": config,
        "signals": ["genarm", "rad", "cdq", "args"]
    }, output_path)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
