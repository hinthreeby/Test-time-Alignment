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

from Method.Router.adapters.registry import get_adapter

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
    path = PROJECT_ROOT / "Method" / "Router" / "configs" / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)

def load_adapter(name, device):
    config = load_config(name)
    adapter = get_adapter(name)(config, PROJECT_ROOT, device)
    adapter.load_models()
    return adapter

def release(adapter):
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.inference_mode()
def build_base_rows(data, top_k, max_response_tokens, max_samples, device):
    adapter = load_adapter("rad", device)
    rows = []

    if max_samples:
        data = data[:max_samples]

    for sample_id, sample in enumerate(tqdm(data, desc="Building candidate sets")):
        prompt = sample["prompt"]
        response = sample["response"]

        prefix_ids = adapter.encode_prompt(prompt)
        response_ids = adapter.encode_response(response)[:max_response_tokens]

        if prefix_ids.numel() == 0 or response_ids.numel() == 0:
            continue

        for position, gold_id in enumerate(response_ids):
            full_logits = adapter.lm(input_ids=prefix_ids).logits[0, -1].float()
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

            prefix_ids = torch.cat([prefix_ids, gold_id.view(1, 1)], dim=1)

    release(adapter)
    return rows

@torch.inference_mode()
def add_signal(rows, signal_index, method_name, device):
    signal_device = torch.device("cpu") if method_name == "genarm" else device
    print(f">>> {method_name} device: {signal_device}", flush=True)
    adapter = load_adapter(method_name, signal_device)

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = PROJECT_ROOT / "dataset" / "multisignal_train" / f"{args.split}.jsonl"
    output_dir = PROJECT_ROOT / "dataset" / "multisignal_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(input_path)

    print(f"Split: {args.split}")
    print(f"Samples: {len(data) if args.max_samples is None else min(len(data), args.max_samples)}")
    print(f"Device: {device}")
    print(f"Top-k: {args.top_k}")

    rows = build_base_rows(data, args.top_k, args.max_response_tokens, args.max_samples, device)

    for signal_index, (_, method_name) in enumerate(SIGNALS):
        add_signal(rows, signal_index, method_name, device)

    output_path = output_dir / f"{args.split}.pt"
    torch.save(rows, output_path)

    print(f"Rows: {len(rows)}")
    print(f"Signal order: {[x[0] for x in SIGNALS]}")
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
