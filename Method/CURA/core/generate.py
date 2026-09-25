from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.config import PROJECT_ROOT, load_config, objective_mismatches
from Method.CURA.core.cache_io import fingerprint, sha256_file
from Method.CURA.core.features import base_routing_features, controller_features
from Method.CURA.core.fusion import fuse_policies
from Method.CURA.models.calibrator import RobustSignalCalibrator, build_calibrator
from Method.CURA.models.controller import CuraController
from Method.CURA.router.adapters.registry import load_adapters


def text(value):
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def load_samples(path, limit):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip(): continue
            item = json.loads(line)
            rows.append({"id": item.get("id", index), "md5_hash": item.get("md5_hash"),
                         "prompt": text(item.get("prompt")),
                         "reference": text(item.get("continuation", item.get("response"))),
                         "num_positive": item.get("num_positive")})
            if limit and len(rows) >= limit: break
    return rows


def key(row):
    return ("md5", str(row["md5_hash"])) if row.get("md5_hash") else ("prompt", row.get("prompt", "").strip())


def completed(row):
    return row and row.get("status") == "success" and isinstance(row.get("response"), str) and row["response"].strip()


def load_existing(path):
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"CURA output must contain a JSON list or JSONL objects: {path}")
    return payload


def save_results(rows, path):
    """Atomically publish the final result using the repo-wide JSON-list format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def apply_inference_overrides(output, args):
    """Apply explicit inference controls after the learned controller heads."""
    if args.ablation == "no-gate":
        output["gate"] = torch.ones_like(output["gate"])
    elif args.fixed_gate is not None:
        output["gate"] = torch.full_like(output["gate"], args.fixed_gate)
    if args.fixed_lambda is not None:
        output["strength"] = torch.full_like(output["strength"], args.fixed_lambda)
    return output


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]; num_signals = len(checkpoint["signals"])
    cal_cfg = config["calibration"]
    calibrator = build_calibrator(config, num_signals).to(device)
    calibrator.load_state_dict(checkpoint["calibrator_state_dict"])
    cfg = config["controller"]
    controller = CuraController(num_signals, checkpoint["feature_dim"], cfg["hidden_dim"], cfg["num_layers"],
                                cfg["dropout"], cfg["lambda_max"], cfg["signal_costs"]).to(device)
    controller.load_state_dict(checkpoint["controller_state_dict"]); controller.eval(); calibrator.eval()
    return checkpoint, config, calibrator, controller


@torch.inference_mode()
def generate_one(model, tokenizer, adapters, calibrator, controller, config, prompt, args, device):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    prompt_length = ids.size(1); telemetry = []
    running_weights = []
    prompt_weights = None
    past_key_values = None
    model_input = ids
    cfg = config["controller"]
    for position in range(args.max_new_tokens):
        base_output = model(
            model_input, past_key_values=past_key_values, use_cache=True
        )
        logits = base_output.logits[0, -1].float()
        past_key_values = base_output.past_key_values
        base, candidates = torch.topk(logits, args.top_k)
        routing = base_routing_features(
            base.unsqueeze(0), torch.tensor([position], device=device), torch.tensor([ids.size(1)], device=device),
            logits.unsqueeze(0)
        )
        mask = controller.select_mask(routing, args.signal_budget)
        if args.leave_out and args.leave_out in config["signals"]:
            mask[:, config["signals"].index(args.leave_out)] = False
            if not mask.any(): raise RuntimeError("leave-out removed every available signal")
        outputs, raw_columns, sources, costs = [], [], [], []
        mismatch_rates = []
        candidate_texts = tokenizer.batch_decode(candidates)
        prefix_text = tokenizer.decode(ids[0], skip_special_tokens=True)
        for signal_index, adapter in enumerate(adapters):
            if mask[0, signal_index]:
                item = adapter.score(
                    ids.to(adapter.device), candidates.to(adapter.device), candidate_texts, prefix_text
                )
                column = item.scores[0].to(device); outputs.append(item)
                sources.append(item.source); costs.append(item.cost_ms)
                mismatch_rates.append(item.metadata.get("tokenization_mismatch_rate", 0.0))
            else:
                column = torch.zeros(args.top_k, device=device)
                sources.append("skipped_by_selector"); costs.append(0.0)
                mismatch_rates.append(0.0)
            if args.corrupt_signal == config["signals"][signal_index]:
                column = column + args.corrupt_std * torch.randn_like(column)
            raw_columns.append(column)
        raw = torch.stack(raw_columns, -1).unsqueeze(0)
        if args.ablation == "no-calibration":
            mu, log_var = RobustSignalCalibrator(len(adapters)).to(device)(raw, mask)
        else:
            mu, log_var = calibrator(raw, mask)
        if args.ablation == "no-uncertainty": log_var = torch.full_like(log_var, -12.0)
        if args.ablation == "shuffled-uncertainty":
            log_var = log_var[..., torch.randperm(log_var.size(-1), device=device)]
        features = controller_features(base.unsqueeze(0), mu, log_var, mask,
                                       torch.tensor([position], device=device), torch.tensor([ids.size(1)], device=device), routing)
        output = controller(features, mask)
        if args.ablation == "uniform": output["weights"] = mask.float() / mask.float().sum(-1, keepdim=True)
        if args.ablation == "inverse-uncertainty":
            confidence = log_var.exp().mean(1).reciprocal() * mask.float()
            output["weights"] = confidence / confidence.sum(-1, keepdim=True).clamp_min(1e-8)
        if args.ablation == "mean-weights":
            running_weights.append(output["weights"])
            output["weights"] = torch.stack(running_weights).mean(0)
        if args.ablation == "prompt-router":
            if prompt_weights is None: prompt_weights = output["weights"]
            output["weights"] = prompt_weights
        output = apply_inference_overrides(output, args)
        disagreement_penalty = 0.0 if args.ablation == "no-disagreement" else cfg["disagreement_penalty"]
        epsilon_kl = None if args.ablation == "no-kl" else cfg["epsilon_kl"]
        fused = fuse_policies(
            base.unsqueeze(0), mu, log_var, output, cfg["kappa"], disagreement_penalty, epsilon_kl,
            disagreement_clip=cfg.get("disagreement_clip"),
        )
        probabilities = fused["probabilities"][0]
        selected = torch.multinomial(probabilities, 1).item() if args.do_sample else probabilities.argmax().item()
        token = candidates[selected].reshape(1, 1); ids = torch.cat([ids, token], 1)
        model_input = token
        telemetry.append({"weights": output["weights"][0].cpu(), "uncertainty": log_var.exp().mean(1)[0].cpu(),
                          "disagreement": float(fused["step_disagreement"][0]),
                          "strength_pre_projection": float(fused["requested_strength"][0]),
                          "strength_post_projection": float(fused["projected_strength"][0]),
                          "pre_projection_kl": float(fused["pre_projection_kl"][0]),
                          "kl_limit_hit": float(fused["kl_limit_hit"][0]),
                          "gate": float(output["gate"][0]), "kl": float(fused["kl"][0]),
                          "cost_ms": costs, "sources": sources,
                          "mismatch_rate": mismatch_rates,
                          "selected": mask[0].cpu()})
        if tokenizer.eos_token_id is not None and int(token) == tokenizer.eos_token_id: break
    count = max(1, len(telemetry)); names = config["signals"]
    average = lambda field, j=None: sum(float(step[field][j] if j is not None else step[field]) for step in telemetry) / count
    return {
        "response": tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True).strip(), "tokens": ids.size(1) - prompt_length,
        "signal_sources": telemetry[-1]["sources"] if telemetry else [],
        "mean_weights": {name: average("weights", j) for j, name in enumerate(names)},
        "mean_uncertainty": {name: average("uncertainty", j) for j, name in enumerate(names)},
        "mean_cost_ms": {name: average("cost_ms", j) for j, name in enumerate(names)},
        "selection_rate": {name: average("selected", j) for j, name in enumerate(names)},
        "tokenization_mismatch_rate": {name: average("mismatch_rate", j) for j, name in enumerate(names)},
        "mean_disagreement": average("disagreement"),
        "mean_strength_pre_projection": average("strength_pre_projection"),
        "mean_strength_post_projection": average("strength_post_projection"),
        "mean_strength": average("strength_post_projection"),
        "mean_pre_projection_kl": average("pre_projection_kl"),
        "kl_limit_hit_rate": average("kl_limit_hit"),
        "signal_calls_per_token": sum(average("selected", j) for j in range(len(names))),
        "mean_gate": average("gate"), "mean_base_weight": 1.0 - average("gate"), "mean_kl": average("kl"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Method/CURA/checkpoints/cura_controller.pt")
    parser.add_argument("--input", default="dataset/rad_benchmark/all.jsonl")
    parser.add_argument("--output", default="results/cura.json")
    parser.add_argument("--num-prompts", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--base-model", default=None, help="Optional unseen-base transfer model under models/ or absolute path")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--allow-objective-mismatch", action="store_true")
    parser.add_argument("--signal-budget", type=int, default=None)
    parser.add_argument("--ablation", choices=[
        "none", "uniform", "no-calibration", "no-uncertainty", "shuffled-uncertainty",
        "inverse-uncertainty", "no-disagreement", "no-gate", "no-kl", "mean-weights", "prompt-router"
    ], default="none")
    parser.add_argument("--fixed-lambda", type=float)
    parser.add_argument(
        "--fixed-gate",
        type=float,
        help="Override the learned gate with a value in [0, 1]; keeps KL projection enabled.",
    )
    parser.add_argument("--leave-out", choices=["genarm", "rad", "cdq", "args"])
    parser.add_argument("--corrupt-signal", choices=["genarm", "rad", "cdq", "args"])
    parser.add_argument("--corrupt-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.fixed_gate is not None and not 0.0 <= args.fixed_gate <= 1.0:
        parser.error("--fixed-gate must be in [0, 1]")
    if args.fixed_gate is not None and args.ablation == "no-gate":
        parser.error("--fixed-gate cannot be combined with --ablation no-gate")
    if args.fixed_lambda is not None and args.fixed_lambda < 0.0:
        parser.error("--fixed-lambda must be non-negative")
    device = torch.device("cuda" if args.base_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    signal_device = torch.device("cuda" if args.signal_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.base_device == "cuda" and device.type != "cuda" or args.signal_device == "cuda" and signal_device.type != "cuda": raise RuntimeError("CUDA requested but unavailable")
    checkpoint_path, input_path, output_path = map(Path, [args.checkpoint, args.input, args.output])
    if not checkpoint_path.is_absolute(): checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not input_path.is_absolute(): input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute(): output_path = PROJECT_ROOT / output_path
    checkpoint, config, calibrator, controller = load_models(checkpoint_path, device)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_fingerprint = fingerprint(config)
    args.signal_budget = args.signal_budget or len(config["signals"])
    if not 1 <= args.signal_budget <= len(config["signals"]): raise ValueError("signal-budget is outside available signals")
    if objective_mismatches(config) and not args.allow_objective_mismatch: raise RuntimeError("Objective mismatch blocks generation; use compatible checkpoints")
    partial_path = output_path.with_name(output_path.name + ".partial.jsonl")
    existing = {}
    for resume_path in (output_path, partial_path):
        for row in load_existing(resume_path):
            existing[key(row)] = row
    samples = load_samples(input_path, args.num_prompts)
    pending = [(i, sample) for i, sample in enumerate(samples) if not completed(existing.get(key(sample)))]
    print(f"Target: {len(samples)} | complete: {len(samples)-len(pending)} | pending: {len(pending)}")
    if not pending:
        ordered = [existing[key(sample)] for sample in samples]
        save_results(ordered, output_path)
        partial_path.unlink(missing_ok=True)
        print(f"Saved -> {output_path}")
        return
    model_path = Path(args.base_model) if args.base_model else PROJECT_ROOT / "models" / config["base_model"]
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path if model_path.parts[0] == "models" else PROJECT_ROOT / "models" / model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    adapters = load_adapters(config, PROJECT_ROOT, signal_device); output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_warmup_prompts = 0
    if device.type == "cuda" and pending:
        warmup_index, warmup_sample = pending[0]
        torch.manual_seed(args.seed + warmup_index)
        torch.cuda.manual_seed_all(args.seed + warmup_index)
        generate_one(model, tokenizer, adapters, calibrator, controller, config, warmup_sample["prompt"], args, device)
        synchronize_cuda()
        timing_warmup_prompts = 1
    if not partial_path.exists() and existing:
        with partial_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                row = existing.get(key(sample))
                if row is not None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        with partial_path.open("a", encoding="utf-8") as handle:
            for index, sample in tqdm(pending, desc="CURA generation"):
                synchronize_cuda()
                started = time.perf_counter()
                try:
                    torch.manual_seed(args.seed + index)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed + index)
                    generated = generate_one(model, tokenizer, adapters, calibrator, controller, config, sample["prompt"], args, device)
                    synchronize_cuda()
                    if not generated["response"]: raise RuntimeError("empty response")
                    status, error = "success", None
                except Exception as exception:
                    synchronize_cuda()
                    generated, status, error = {"response": None}, "failed", f"{type(exception).__name__}: {exception}"
                elapsed = time.perf_counter() - started
                generated_tokens = int(generated.get("tokens", 0) or 0)
                row = {**sample, **generated, "method": "cura", "objective": config["objective"],
                       "signal_objectives": config["signal_objectives"], "calibration_mode": config["calibration"]["mode"],
                       "paper_mode": bool(config.get("paper_mode")),
                       "training_data_provenance": checkpoint.get("data_provenance", {}),
                       "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                       "decoding_method": "sample" if args.do_sample else "greedy", "latency_ms": elapsed * 1000,
                       "generated_tokens_per_second": generated_tokens / elapsed if elapsed > 0 else None,
                       "timing_warmup_prompts": timing_warmup_prompts,
                       "signal_budget": args.signal_budget, "ablation": args.ablation,
                       "seed": args.seed,
                       "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
                       "config_fingerprint": config_fingerprint,
                       "base_model": str(model_path), "transfer_base": args.base_model is not None,
                       "fixed_lambda": args.fixed_lambda, "fixed_gate": args.fixed_gate,
                       "leave_out": args.leave_out,
                       "corrupt_signal": args.corrupt_signal, "corrupt_std": args.corrupt_std if args.corrupt_signal else None,
                       "status": status, "error": error}
                existing[key(sample)] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
    finally:
        del adapters, model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    ordered = [existing[key(sample)] for sample in samples if key(sample) in existing]
    save_results(ordered, output_path)
    partial_path.unlink(missing_ok=True)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
