#!/usr/bin/env python3
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.MultiSignal.router.adapters.registry import get_adapter
from Method.MultiSignal.router.adapters.fallback_adapter import LocalFallbackAdapter

SIGNALS = [
    ("genarm", "genarm"),
    ("rad", "rad"),
    ("cdq", "cdq"),
    ("args", "args"),
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(name):
    path = PROJECT_ROOT / "Method" / "MultiSignal" / "router" / "configs" / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_adapter(name, device, config_override=None):
    config = load_config(name)
    if config_override:
        config = {**config, **config_override}
    adapter = get_adapter(name)(config, PROJECT_ROOT, device)
    adapter.load_models()
    return adapter


def load_base_adapter(device):
    config = load_config("rad")
    config = {**config, "name": "base-candidate-lm"}
    adapter = LocalFallbackAdapter(config, PROJECT_ROOT, device)
    adapter.load_models()
    return adapter


def release(adapter):
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def build_base_rows(data, top_k, max_response_tokens, max_samples, device):
    adapter = load_base_adapter(device)
    rows = []

    if max_samples:
        data = data[:max_samples]

    for sample_id, sample in enumerate(tqdm(data, desc="Building candidate sets")):
        prompt = sample["prompt"]
        response = sample["response"]

        prefix_ids = adapter.encode_prompt(prompt)
        response_ids = adapter.encode_response(response).flatten()[:max_response_tokens]

        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)

        if prefix_ids.numel() == 0 or response_ids.numel() == 0:
            continue

        for position, gold_token in enumerate(response_ids):
            gold_id = int(gold_token.item())
            full_logits = adapter.next_logits(prefix_ids)[0].float()
            top_logits, candidate_ids = torch.topk(full_logits, k=top_k)

            matches = (candidate_ids == gold_id).nonzero(as_tuple=False)

            if len(matches):
                gold_index = int(matches[0].item())
            else:
                candidate_ids[-1] = gold_id
                top_logits[-1] = full_logits[gold_id]
                gold_index = top_k - 1

            rows.append({
                "sample_id": sample_id,
                "position": position,
                "prefix_ids": prefix_ids[0].detach().cpu(),
                "candidate_ids": candidate_ids.detach().cpu(),
                "lm_logits": top_logits.detach().cpu(),
                "gold_index": gold_index,
                "signal_scores": torch.zeros(top_k, len(SIGNALS), dtype=torch.float32),
            })

            prefix_ids = torch.cat([prefix_ids, torch.tensor([[gold_id]], device=prefix_ids.device)], dim=1)

    release(adapter)
    return rows


@torch.inference_mode()
def add_signal(rows, signal_index, method_name, device):
    signal_device = device
    print(f">>> {method_name} device: {signal_device}", flush=True)
    adapter = load_adapter(method_name, signal_device)
    print(f">>> {method_name} source: {adapter.metadata()['source']}", flush=True)

    for row in tqdm(rows, desc=f"Scoring {method_name}"):
        prefix_ids = row["prefix_ids"].unsqueeze(0).to(signal_device)
        candidate_ids = row["candidate_ids"].to(signal_device)

        scores = adapter.get_signal_scores(prefix_ids, candidate_ids).float().detach().cpu()

        if scores.numel() != candidate_ids.numel():
            raise RuntimeError(f"{method_name}: expected {candidate_ids.numel()} scores, got {scores.numel()}")

        row["signal_scores"][:, signal_index] = scores.reshape(-1)

    release(adapter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-response-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu")
    args = parser.parse_args()

    base_device = resolve_device(args.base_device)
    signal_device = resolve_device(args.signal_device)

    input_path = PROJECT_ROOT / "dataset" / "multisignal_train" / f"{args.split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset split: {input_path}")

    output_dir = PROJECT_ROOT / "dataset" / "multisignal_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(input_path)

    print(f"Split: {args.split}")
    print(f"Samples: {len(data) if args.max_samples is None else min(len(data), args.max_samples)}")
    print(f"Base device: {base_device}")
    print(f"Signal device: {signal_device}")
    print(f"Top-k: {args.top_k}")

    rows = build_base_rows(data, args.top_k, args.max_response_tokens, args.max_samples, base_device)
    if not rows:
        raise RuntimeError("No cache rows were built. Check that prompt/response fields are non-empty.")

    for signal_index, (_, method_name) in enumerate(SIGNALS):
        add_signal(rows, signal_index, method_name, signal_device)

    output_path = output_dir / f"{args.split}.pt"
    torch.save(rows, output_path)

    print(f"Rows: {len(rows)}")
    print(f"Signal order: {[x[0] for x in SIGNALS]}")
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
