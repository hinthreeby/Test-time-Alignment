#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.Router.adapters.registry import get_adapter
from Method.Router.core.utils import load_config
from Method.Router.models.router import ScalarRouter


def evaluate(router, adapter, rows, device):
    router.eval()
    losses = []

    with torch.inference_mode():
        for row in rows:
            logits = row["base_logits"].unsqueeze(0).to(device)
            signal = row["signal_scores"].unsqueeze(0).to(device)
            features = adapter.build_features(logits, signal)
            value, _ = router(features)
            guided = adapter.guided_scores(logits, signal, value)
            target = torch.tensor([row["gold_index"]], dtype=torch.long, device=device)
            losses.append(float(F.cross_entropy(guided, target).item()))

    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT, args.method)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_adapter(args.method)(config, PROJECT_ROOT, device)

    cache_dir = PROJECT_ROOT / "dataset" / "router_cache" / args.method
    train_data = torch.load(cache_dir / "train.pt", map_location="cpu", weights_only=False)
    val_data = torch.load(cache_dir / "validation.pt", map_location="cpu", weights_only=False)

    router = ScalarRouter(input_dim=config["top_k"] * 2, hidden_dim=config["router"]["hidden_dim"], output_min=config["router"]["output_min"], output_max=config["router"]["output_max"]).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    best_val = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        router.train()
        losses = []

        for row in tqdm(train_data, desc=f"{args.method} epoch {epoch}/{config['training']['epochs']}"):
            logits = row["base_logits"].unsqueeze(0).to(device)
            signal = row["signal_scores"].unsqueeze(0).to(device)
            features = adapter.build_features(logits, signal)
            value, _ = router(features)
            guided = adapter.guided_scores(logits, signal, value)
            target = torch.tensor([row["gold_index"]], dtype=torch.long, device=device)
            loss = F.cross_entropy(guided, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        train_loss = sum(losses) / len(losses)
        val_loss = evaluate(router, adapter, val_data, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            output = PROJECT_ROOT / "Method" / "Router" / "checkpoints" / f"{args.method}_router.pt"
            torch.save({"router_state_dict": router.state_dict(), "config": config, "best_val_loss": best_val}, output)
            print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
