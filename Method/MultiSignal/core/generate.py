#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.MultiSignal.core.cache import SIGNALS, load_config, load_adapter, release
from Method.MultiSignal.core.fusion import fuse_scores
from Method.MultiSignal.models.controller import MultiSignalController

DEFAULT_CHECKPOINT = PROJECT_ROOT / "Method" / "MultiSignal" / "checkpoints" / "controller_4signal.pt"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "multi_signal.jsonl"


def get_text(value):
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value or "")


def load_prompts(path, max_prompts):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            sample = json.loads(line)
            continuation = sample.get("continuation", sample.get("response"))
            rows.append({
                "id": sample.get("id", index),
                "md5_hash": sample.get("md5_hash"),
                "prompt": get_text(sample.get("prompt")),
                "reference": get_text(continuation),
                "num_positive": sample.get("num_positive"),
            })
            if max_prompts and len(rows) >= max_prompts:
                break
    return rows


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_controller(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    signals = checkpoint.get("signals", [name for name, _ in SIGNALS])
    signal_costs = checkpoint.get("signal_costs", config.get("controller", {}).get("signal_costs", [1.0] * len(signals)))

    controller = MultiSignalController(
        num_signals=len(signals),
        hidden_dim=config["controller"]["hidden_dim"],
        signal_costs=signal_costs[:len(signals)],
    ).to(device)
    controller.load_state_dict(checkpoint["controller_state_dict"], strict=True)
    controller.eval()

    return controller, signals


def fallback_signal_scores(candidate_logits, signal_names):
    scores = []
    for name in signal_names:
        cfg = load_config(name)
        if name == "rad":
            score = candidate_logits * float(cfg.get("beta", 30.0)) / 100.0
        elif name == "cdq":
            score = (candidate_logits - candidate_logits.mean()) * float(cfg.get("alpha", 0.5))
        else:
            score = candidate_logits * float(cfg.get("alpha", 1.0))
        scores.append(score)
    return torch.stack(scores, dim=-1)


def load_signal_adapters(signal_names, device):
    adapters = []
    for name in signal_names:
        adapter = load_adapter(name, device)
        adapters.append(adapter)
        print(f"Signal {name}: {adapter.metadata()['source']}")
    return adapters


def adapter_signal_scores(prefix_ids, candidate_ids, adapters, target_device):
    scores = [
        adapter.get_signal_scores(
            prefix_ids.to(adapter.device),
            candidate_ids.to(adapter.device),
        ).to(target_device).float()
        for adapter in adapters
    ]
    return torch.stack(scores, dim=-1)


@torch.inference_mode()
def generate_one(model, tokenizer, controller, signal_names, signal_adapters, prompt, args, device):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    weight_history = []
    strength_history = []
    trust_history = []

    for _ in range(args.max_new_tokens):
        logits = model(input_ids=input_ids).logits[0, -1].float()
        top_logits, candidate_ids = torch.topk(logits, k=args.top_k)
        if signal_adapters:
            signal_scores = adapter_signal_scores(input_ids, candidate_ids, signal_adapters, device)
        else:
            signal_scores = fallback_signal_scores(top_logits, signal_names)

        controller_output = controller(top_logits.unsqueeze(0), signal_scores.unsqueeze(0))
        final_scores, _ = fuse_scores(top_logits.unsqueeze(0), signal_scores.unsqueeze(0), controller_output)
        final_scores = final_scores.squeeze(0)

        if args.do_sample:
            probs = torch.softmax(final_scores / args.temperature, dim=-1)
            selected_index = torch.multinomial(probs, num_samples=1).item()
        else:
            selected_index = torch.argmax(final_scores).item()

        next_token = candidate_ids[selected_index].view(1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        weight_history.append(controller_output["weights"][0].detach().cpu().tolist())
        strength_history.append(float(controller_output["strength"][0].detach().cpu()))
        trust_history.append(float(controller_output["trust"][0].detach().cpu()))

        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

    new_tokens = input_ids[0, prompt_len:]
    return {
        "response": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "tokens": int(new_tokens.numel()),
        "mean_weights": [
            sum(step[i] for step in weight_history) / len(weight_history)
            for i in range(len(signal_names))
        ] if weight_history else [],
        "mean_strength": sum(strength_history) / len(strength_history) if strength_history else 0.0,
        "mean_trust": sum(trust_history) / len(trust_history) if trust_history else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--fallback-signals", action="store_true", help="Use lightweight LM-logit pseudo-signals for smoke tests.")
    args = parser.parse_args()

    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    device = resolve_device(args.base_device)
    signal_device = resolve_device(args.signal_device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    try:
        controller, signal_names = load_controller(checkpoint_path, device)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint is not compatible with the current controller: {checkpoint_path}. "
            "Re-train with `python Method/run_method.py --method multisignal --action train "
            "--cache dataset/multisignal_cache/train.pt`."
        ) from error

    base_config = load_config("rad")
    model_path = PROJECT_ROOT / "models" / base_config.get("model_name", "gpt2-large")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    signal_adapters = [] if args.fallback_signals else load_signal_adapters(signal_names, signal_device)

    if args.prompt:
        samples = [{"id": 0, "md5_hash": None, "prompt": args.prompt, "reference": "", "num_positive": None}]
    else:
        samples = load_prompts(Path(args.input), args.max_prompts)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Signal device: {signal_device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Signals: {signal_names}")
    print(f"Signal mode: {'fallback' if args.fallback_signals else 'adapter'}")
    print(f"Prompts: {len(samples)}")

    try:
        with output_path.open("w", encoding="utf-8") as file:
            for index, sample in enumerate(tqdm(samples, desc="Generating MultiSignal")):
                start = time.perf_counter()
                try:
                    generated = generate_one(model, tokenizer, controller, signal_names, signal_adapters, sample["prompt"], args, device)
                    response = generated["response"]
                    tokens = generated["tokens"]
                    mean_weights = generated["mean_weights"]
                    mean_strength = generated["mean_strength"]
                    mean_trust = generated["mean_trust"]
                    status = "success"
                    error = None
                except Exception as exception:
                    response = None
                    tokens = 0
                    mean_weights = []
                    mean_strength = 0.0
                    mean_trust = 0.0
                    status = "failed"
                    error = f"{type(exception).__name__}: {exception}"

                row = {
                    "id": sample.get("id", index),
                    "md5_hash": sample.get("md5_hash"),
                    "prompt": sample["prompt"],
                    "reference": sample["reference"],
                    "response": response,
                    "num_positive": sample.get("num_positive"),
                    "method": "MultiSignal",
                    "signals": signal_names,
                    "signal_sources": [a.metadata()["source"] for a in signal_adapters] if signal_adapters else ["fallback_lm_logits"] * len(signal_names),
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "decoding_method": "sample" if args.do_sample else "greedy",
                    "temperature": args.temperature if args.do_sample else None,
                    "mean_weights": mean_weights,
                    "mean_strength": mean_strength,
                    "mean_trust": mean_trust,
                    "mean_base_weight": 1.0 - mean_trust,
                    "tokens": tokens,
                    "latency": round(time.perf_counter() - start, 4),
                    "status": status,
                    "error": error,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
    finally:
        for adapter in signal_adapters:
            release(adapter)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
