#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.Router.adapters.registry import get_adapter
from Method.Router.core.utils import get_text, load_config, load_jsonl
from Method.Router.models.router import ScalarRouter


@torch.inference_mode()
def generate_one(prompt, adapter, router, config):
    prefix = adapter.encode_prompt(prompt)
    generated = []
    values = []

    for _ in range(config["max_new_tokens"]):
        _, logits, candidate_ids = adapter.get_candidates(prefix)
        signal = adapter.get_signal_scores(prefix, candidate_ids)
        features = adapter.build_features(logits.unsqueeze(0), signal.unsqueeze(0))
        value, _ = router(features)
        guided = adapter.guided_scores(logits.unsqueeze(0), signal.unsqueeze(0), value)
        index = int(torch.argmax(guided[0]).item())
        token = candidate_ids[index]

        generated.append(int(token.item()))
        values.append(float(value.item()))
        prefix = adapter.append_token(prefix, token)

        if int(token.item()) == adapter.eos_token_id:
            break

    return adapter.decode_tokens(generated), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--num-prompts", type=int, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT, args.method)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_adapter(args.method)(config, PROJECT_ROOT, device)
    adapter.load_models()

    checkpoint = torch.load(PROJECT_ROOT / "Method" / "Router" / "checkpoints" / f"{args.method}_router.pt", map_location="cpu", weights_only=False)
    router = ScalarRouter(input_dim=config["top_k"] * 2, hidden_dim=config["router"]["hidden_dim"], output_min=config["router"]["output_min"], output_max=config["router"]["output_max"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"], strict=True)
    router.eval()

    limit = args.num_prompts or config["benchmark"]["num_prompts"]
    rows = load_jsonl(PROJECT_ROOT / config["benchmark"]["dataset"], limit)
    results = []

    for index, row in enumerate(tqdm(rows, desc=f"{args.method}+Router")):
        prompt = get_text(row.get("prompt"))
        start = time.perf_counter()
        response, router_history = generate_one(prompt, adapter, router, config)

        results.append({
            "id": index,
            "prompt": prompt,
            "reference": get_text(row.get("continuation")),
            "response": response,
            "method": f"{args.method.upper()}-Router",
            "router_history": router_history,
            "latency": time.perf_counter() - start,
        })

    output = PROJECT_ROOT / "results" / f"{args.method}_router.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
