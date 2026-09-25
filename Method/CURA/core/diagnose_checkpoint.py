"""Offline CURA checkpoint diagnostics on an existing token cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.cache_io import ShardedCuraDataset, verify_cache
from Method.CURA.core.config import PROJECT_ROOT
from Method.CURA.core.features import controller_features
from Method.CURA.core.fusion import fuse_policies
from Method.CURA.core.train import collate
from Method.CURA.models.calibrator import build_calibrator
from Method.CURA.models.controller import CuraController


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_components(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["config"]
    num_signals = len(state["signals"])
    calibrator = build_calibrator(config, num_signals).to(device)
    calibrator.load_state_dict(state["calibrator_state_dict"])
    settings = config["controller"]
    controller = CuraController(
        num_signals, state["feature_dim"], settings["hidden_dim"], settings["num_layers"],
        settings["dropout"], settings["lambda_max"], settings["signal_costs"],
    ).to(device)
    controller.load_state_dict(state["controller_state_dict"])
    calibrator.eval(); controller.eval()
    return state, config, calibrator, controller


@torch.inference_mode()
def diagnose(checkpoint_path, cache_dir, batch_size=256, device=torch.device("cpu")):
    verify_cache(cache_dir)
    dataset = ShardedCuraDataset(cache_dir)
    state, config, calibrator, controller = load_components(checkpoint_path, device)
    if dataset.manifest.get("signals") != state["signals"]:
        raise RuntimeError("Checkpoint and cache signal orders differ")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total = 0
    natural_gold_total = 0
    known_gold_total = 0
    known_natural_gold_total = 0
    sums = {name: 0.0 for name in (
        "base_nll", "cura_nll", "base_top1", "cura_top1", "argmax_changed",
        "gate", "strength", "kl", "disagreement",
    )}
    target_sums = {name: 0.0 for name in ("base_utility", "cura_utility", "oracle_utility")}
    target_count = 0
    weight_sums = torch.zeros(len(state["signals"]), device=device)
    for batch in loader:
        base = batch["base_logits"].to(device)
        raw = batch["raw_scores"].to(device)
        mask = batch["signal_mask"].to(device)
        gold = batch["gold_index"].to(device)
        mu, log_var = calibrator(raw, mask)
        features = controller_features(
            base, mu, log_var, mask, batch["position"].to(device),
            batch["prefix_length"].to(device), batch["base_routing_features"].to(device),
        )
        output = controller(features, mask)
        settings = config["controller"]
        fused = fuse_policies(
            base, mu, log_var, output, settings["kappa"], settings["disagreement_penalty"],
            settings["epsilon_kl"], disagreement_clip=settings.get("disagreement_clip"),
        )
        count = gold.numel(); total += count
        base_prob, cura_prob = fused["base_probabilities"], fused["probabilities"]
        natural_gold = batch["gold_in_top_k"].to(device)
        natural_gold_total += int(natural_gold.sum())
        known_gold = batch["gold_in_top_k_known"].to(device)
        known_gold_total += int(known_gold.sum())
        known_natural_gold_total += int((known_gold & natural_gold).sum())
        natural_gold_index = gold[natural_gold].unsqueeze(1)
        sums["base_nll"] += float(-base_prob[natural_gold].gather(1, natural_gold_index).clamp_min(1e-8).log().sum())
        sums["cura_nll"] += float(-cura_prob[natural_gold].gather(1, natural_gold_index).clamp_min(1e-8).log().sum())
        sums["base_top1"] += float((base_prob[natural_gold].argmax(-1) == gold[natural_gold]).sum())
        sums["cura_top1"] += float((cura_prob[natural_gold].argmax(-1) == gold[natural_gold]).sum())
        sums["argmax_changed"] += float((base_prob.argmax(-1) != cura_prob.argmax(-1)).sum())
        sums["gate"] += float(output["gate"].sum())
        sums["strength"] += float(fused["projected_strength"].sum())
        sums["kl"] += float(fused["kl"].sum())
        sums["disagreement"] += float(fused["step_disagreement"].sum())
        weight_sums += output["weights"].sum(0)
        has_targets = batch["has_targets"].to(device)
        if has_targets.any():
            targets = batch["target_utilities"].to(device)[has_targets]
            target_count += int(has_targets.sum())
            target_sums["base_utility"] += float((base_prob[has_targets] * targets).sum())
            target_sums["cura_utility"] += float((cura_prob[has_targets] * targets).sum())
            target_sums["oracle_utility"] += float(targets.max(-1).values.sum())
    metrics = {name: value / max(1, total) for name, value in sums.items()}
    for name in ("base_nll", "cura_nll", "base_top1", "cura_top1"):
        metrics[name] = sums[name] / max(1, natural_gold_total)
    metrics["gold_in_top_k"] = (
        known_natural_gold_total / known_gold_total if known_gold_total else None
    )
    metrics["gold_in_top_k_coverage"] = known_gold_total / max(1, total)
    report = {
        "checkpoint": portable(checkpoint_path), "cache": portable(cache_dir), "steps": total,
        "signals": state["signals"],
        "metrics": metrics,
        "mean_weights": {
            name: float(weight_sums[index] / max(1, total)) for index, name in enumerate(state["signals"])
        },
        "target_rows": target_count,
    }
    if target_count:
        report["target_metrics"] = {name: value / target_count for name, value in target_sums.items()}
        report["target_metrics"]["cura_gain_over_base"] = (
            report["target_metrics"]["cura_utility"] - report["target_metrics"]["base_utility"]
        )
        report["target_metrics"]["remaining_oracle_gap"] = (
            report["target_metrics"]["oracle_utility"] - report["target_metrics"]["cura_utility"]
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Method/CURA/checkpoints/cura_controller.pt")
    parser.add_argument("--cache-dir", default="dataset/cura_cache/validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    report = diagnose(resolve(args.checkpoint), resolve(args.cache_dir), args.batch_size, device)
    payload = json.dumps(report, indent=2)
    if args.output:
        output = resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
