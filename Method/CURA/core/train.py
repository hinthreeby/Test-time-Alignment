from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.cache_io import ShardedCuraDataset, atomic_json, atomic_torch_save, verify_cache
from Method.CURA.core.config import PROJECT_ROOT, load_config
from Method.CURA.core.features import base_routing_features, controller_features
from Method.CURA.core.fusion import fuse_policies
from Method.CURA.models.calibrator import build_calibrator
from Method.CURA.models.controller import CuraController


def collate(rows):
    targets = [row.get("target_utilities") for row in rows]
    has_targets = torch.tensor([target is not None for target in targets], dtype=torch.bool)
    target_tensor = torch.stack([
        target.float() if target is not None else torch.zeros_like(row["base_logits"].float())
        for row, target in zip(rows, targets)
    ])
    base_logits = torch.stack([row["base_logits"].float() for row in rows])
    positions = torch.tensor([row.get("position", row["step"]) for row in rows])
    prefix_lengths = torch.tensor([row.get("prefix_length", 1) for row in rows])
    routing_features = torch.stack([
        row["base_routing_features"].float() if row.get("base_routing_features") is not None
        else base_routing_features(base_logits[index:index + 1], positions[index:index + 1], prefix_lengths[index:index + 1])[0]
        for index, row in enumerate(rows)
    ])
    return {
        "base_logits": base_logits,
        "raw_scores": torch.stack([row["raw_scores"].float().transpose(0, 1) for row in rows]),
        "signal_mask": torch.stack([row["signal_mask"].bool() for row in rows]),
        "gold_index": torch.tensor([row["gold_index"] for row in rows], dtype=torch.long),
        "position": positions,
        "prefix_length": prefix_lengths,
        "base_routing_features": routing_features,
        "target_utilities": target_tensor,
        "has_targets": has_targets,
    }


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_cache(cache_dir, config, expected_split, allow_incomplete=False):
    report = verify_cache(cache_dir)
    dataset = ShardedCuraDataset(cache_dir)
    manifest = dataset.manifest
    errors = []
    if manifest.get("schema_version") != 2:
        errors.append("cache schema must be v2")
    if manifest.get("signals") != config["signals"]:
        errors.append("cache signal order differs from config")
    if manifest.get("objective") != config["objective"]:
        errors.append("cache objective differs from config")
    if manifest.get("split") != expected_split:
        errors.append(f"expected {expected_split} split, got {manifest.get('split')}")
    if config.get("paper_mode") and not manifest.get("paper_mode"):
        errors.append("paper training requires a cache built in paper mode")
    if manifest.get("status") != "complete" and not allow_incomplete:
        errors.append("cache is not complete")
    if manifest.get("failed_prompt_ids") and not allow_incomplete:
        errors.append("cache contains failed prompts")
    if not len(dataset):
        errors.append("cache contains no token rows")
    if errors:
        raise RuntimeError(f"Invalid training cache {cache_dir}: {'; '.join(errors)}")
    return dataset, report


def build_components(config, sample, device):
    num_signals = len(config["signals"])
    cal_cfg = config["calibration"]
    calibrator = build_calibrator(config, num_signals).to(device)
    example = collate([sample])
    with torch.no_grad():
        mu, log_var = calibrator(example["raw_scores"].to(device), example["signal_mask"].to(device))
        feature_dim = controller_features(
            example["base_logits"].to(device), mu, log_var, example["signal_mask"].to(device),
            base_features=example["base_routing_features"].to(device)
        ).size(-1)
    cfg = config["controller"]
    controller = CuraController(
        num_signals, feature_dim, cfg["hidden_dim"], cfg["num_layers"],
        cfg["dropout"], cfg["lambda_max"], cfg["signal_costs"]
    ).to(device)
    return calibrator, controller, feature_dim


def compute_loss(batch, calibrator, controller, config, device):
    base = batch["base_logits"].to(device)
    raw = batch["raw_scores"].to(device)
    mask = batch["signal_mask"].to(device)
    gold = batch["gold_index"].to(device)
    targets = batch["target_utilities"].to(device)
    has_targets = batch["has_targets"].to(device)
    mu, log_var = calibrator(raw, mask)
    features = controller_features(
        base, mu, log_var, mask, batch["position"].to(device), batch["prefix_length"].to(device),
        batch["base_routing_features"].to(device)
    )
    output = controller(features, mask)
    cfg = config["controller"]
    fused = fuse_policies(
        base, mu, log_var, output, cfg["kappa"], cfg["disagreement_penalty"], cfg["epsilon_kl"]
    )
    token_nll = -fused["probabilities"].gather(1, gold.unsqueeze(1)).clamp_min(1e-8).log().mean()
    cost = (output["weights"] * controller.signal_costs).sum(-1).mean()
    settings = config["training"]
    preference = torch.zeros((), device=device)
    calibration = torch.zeros((), device=device)
    selector = torch.zeros((), device=device)
    if has_targets.any():
        selected_targets = targets[has_targets]
        selected_mu, selected_log_var = mu[has_targets], log_var[has_targets]
        selected_mask = mask[has_targets]
        expanded_targets = selected_targets.unsqueeze(-1).expand_as(selected_mu)
        calibration_terms = 0.5 * (
            (expanded_targets - selected_mu).square() * torch.exp(-selected_log_var) + selected_log_var
        )
        calibration = calibration_terms.masked_select(selected_mask.unsqueeze(1).expand_as(calibration_terms)).mean()
        probabilities = fused["probabilities"][has_targets]
        chosen, rejected = selected_targets.argmax(-1), selected_targets.argmin(-1)
        logp = probabilities.clamp_min(1e-8).log()
        margin = logp.gather(1, chosen.unsqueeze(1)) - logp.gather(1, rejected.unsqueeze(1))
        preference = torch.nn.functional.softplus(-settings.get("preference_beta", 1.0) * margin).mean()
        signal_errors = (selected_mu - expanded_targets).square().mean(dim=1)
        selector_targets = torch.softmax(
            -signal_errors.detach() / settings.get("selector_temperature", 0.25)
            - settings.get("selector_cost_weight", 0.01) * controller.signal_costs,
            dim=-1,
        )
        routing_features = batch["base_routing_features"].to(device)[has_targets]
        selector = -(selector_targets * torch.log_softmax(controller.selection_logits(routing_features), -1)).sum(-1).mean()
        base_utility = (fused["base_probabilities"][has_targets] * selected_targets).sum(-1)
        guided_utility = (fused["guided_probabilities"][has_targets] * selected_targets).sum(-1)
        uncertainty = selected_log_var.exp().mean((1, 2))
        gate_logit = (
            guided_utility - base_utility
            - settings.get("gate_uncertainty_penalty", 0.05) * uncertainty
            - settings.get("gate_disagreement_penalty", 0.05) * fused["step_disagreement"][has_targets]
        ) / settings.get("gate_temperature", 0.1)
        gate_target = torch.sigmoid(gate_logit).detach()
        gate_loss = torch.nn.functional.binary_cross_entropy(output["gate"][has_targets], gate_target)
    else:
        uncertainty = log_var.exp().mean((1, 2)).detach()
        disagreement = fused["step_disagreement"].detach()
        uncertain = ((uncertainty > uncertainty.median()) | (disagreement > disagreement.median())).float()
        gate_loss = torch.nn.functional.binary_cross_entropy(output["gate"], 1.0 - uncertain)
    loss = (
        settings.get("gamma_nll", 1.0) * token_nll
        + settings.get("gamma_pref", 0.0) * preference
        + settings.get("gamma_cal", 0.0) * calibration
        + settings["gamma_kl"] * fused["kl"].mean()
        + settings["gamma_cost"] * cost + settings["gamma_gate"] * gate_loss
        + settings.get("gamma_selector", 0.0) * selector
    )
    return loss, {
        "nll": token_nll, "preference": preference, "calibration": calibration,
        "kl": fused["kl"].mean(), "cost": cost, "gate": gate_loss, "selector": selector,
    }


@torch.no_grad()
def evaluate(loader, calibrator, controller, config, device, use_amp):
    controller.eval()
    totals = {name: 0.0 for name in ("loss", "nll", "preference", "calibration", "kl", "cost", "gate", "selector")}
    count = 0
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            loss, pieces = compute_loss(batch, calibrator, controller, config, device)
        size = batch["gold_index"].numel()
        count += size
        totals["loss"] += float(loss) * size
        for name, value in pieces.items():
            totals[name] += float(value) * size
    return {name: value / max(1, count) for name, value in totals.items()}


def checkpoint_payload(config, config_path, feature_dim, controller, calibrator, optimizer, scaler,
                       epoch, best_validation_loss, history):
    return {
        "schema_version": 1, "method": "cura", "config": config, "config_path": str(config_path),
        "signals": config["signals"], "feature_dim": feature_dim, "epoch": epoch,
        "best_validation_loss": best_validation_loss, "history": history,
        "controller_state_dict": controller.state_dict(), "calibrator_state_dict": calibrator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Method/CURA/configs/sentiment.json")
    parser.add_argument("--cache-dir", default="dataset/cura_cache/train")
    parser.add_argument("--validation-cache-dir", default="dataset/cura_cache/validation")
    parser.add_argument("--output", default="Method/CURA/checkpoints/cura_controller.pt")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-incomplete-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config, config_path = load_config(args.config)
    train_data, train_verification = validate_cache(
        resolve(args.cache_dir), config, "train", args.allow_incomplete_cache
    )
    validation_data, validation_verification = validate_cache(
        resolve(args.validation_cache_dir), config, "validation", args.allow_incomplete_cache
    )
    if config["calibration"]["mode"] == "learned_heteroscedastic":
        for label, dataset in (("train", train_data), ("validation", validation_data)):
            if any(dataset[index].get("target_utilities") is None for index in range(len(dataset))):
                raise RuntimeError(f"{label} cache lacks held-out target_utilities required by learned calibration")
        train_target = train_data.manifest.get("target_utility", {})
        validation_target = validation_data.manifest.get("target_utility", {})
        if train_target.get("status") != "complete" or validation_target.get("status") != "complete":
            raise RuntimeError("Learned calibration requires completed target annotation on both caches")
        comparable_fields = ("evaluator", "rollouts", "rollout_tokens")
        if any(train_target.get(field) != validation_target.get(field) for field in comparable_fields):
            raise RuntimeError("Train and validation target protocols differ")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    settings = config["training"]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=settings["batch_size"], shuffle=True, collate_fn=collate,
        num_workers=settings.get("num_workers", 0), pin_memory=device.type == "cuda", generator=generator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=settings["batch_size"], shuffle=False, collate_fn=collate,
        num_workers=settings.get("num_workers", 0), pin_memory=device.type == "cuda"
    )
    calibrator, controller, feature_dim = build_components(config, train_data[0], device)
    optimizer = torch.optim.AdamW(
        list(controller.parameters()) + list(calibrator.parameters()),
        lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    use_amp = device.type == "cuda" and settings.get("mixed_precision", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    output_path = resolve(args.output)
    latest_path = output_path.with_name(output_path.stem + ".latest" + output_path.suffix)
    start_epoch, best_validation_loss, history = 1, float("inf"), []
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state["signals"] != config["signals"] or state["feature_dim"] != feature_dim:
            raise RuntimeError("Resume checkpoint is incompatible with current config/cache")
        controller.load_state_dict(state["controller_state_dict"])
        calibrator.load_state_dict(state["calibrator_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_validation_loss = float(state["best_validation_loss"])
        history = state["history"]
    print(json.dumps({
        "device": str(device), "signals": config["signals"], "train_rows": len(train_data),
        "validation_rows": len(validation_data), "start_epoch": start_epoch,
        "epochs": settings["epochs"], "output": str(output_path)
    }, indent=2))
    for epoch in range(start_epoch, settings["epochs"] + 1):
        controller.train()
        totals = {name: 0.0 for name in ("loss", "nll", "preference", "calibration", "kl", "cost", "gate", "selector")}
        count = 0
        for batch in tqdm(train_loader, desc=f"CURA epoch {epoch}/{settings['epochs']}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                loss, pieces = compute_loss(batch, calibrator, controller, config, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(controller.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            size = batch["gold_index"].numel()
            count += size
            totals["loss"] += float(loss.detach()) * size
            for name, value in pieces.items():
                totals[name] += float(value.detach()) * size
        train_metrics = {name: value / max(1, count) for name, value in totals.items()}
        validation_metrics = evaluate(validation_loader, calibrator, controller, config, device, use_amp)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
        payload = checkpoint_payload(
            config, config_path, feature_dim, controller, calibrator, optimizer, scaler,
            epoch, best_validation_loss, history
        )
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, output_path)
        atomic_json({
            "status": "running", "epoch": epoch, "best_validation_loss": best_validation_loss,
            "latest_checkpoint": str(latest_path), "best_checkpoint": str(output_path), "history": history,
            "cache_verification": {"train": train_verification, "validation": validation_verification}
        }, output_path.with_suffix(".training.json"))
        print(json.dumps(history[-1]))
    atomic_json({
        "status": "complete", "epochs": settings["epochs"], "best_validation_loss": best_validation_loss,
        "latest_checkpoint": str(latest_path), "best_checkpoint": str(output_path), "history": history
    }, output_path.with_suffix(".training.json"))
    print(f"Best checkpoint -> {output_path}")


if __name__ == "__main__":
    main()
