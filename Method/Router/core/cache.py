#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.Router.adapters.registry import get_adapter
from Method.Router.core.utils import load_config, load_jsonl


@torch.inference_mode()
def cache_split(rows, adapter, config, output_path):
    records = []

    for sample_id, row in enumerate(tqdm(rows, desc=output_path.stem)):
        prefix = adapter.encode_prompt(row["prompt"])
        response_ids = adapter.encode_response(row["response"])[:config["max_response_tokens"]]

        for position, gold_token in enumerate(response_ids):
            full_logits, top_logits, candidate_ids = adapter.get_candidates(prefix)
            gold_id = int(gold_token.item())
            matches = (candidate_ids == gold_id).nonzero(as_tuple=False)

            if matches.numel() == 0:
                candidate_ids[-1] = gold_id
                top_logits[-1] = full_logits[gold_id]
                gold_index = config["top_k"] - 1
            else:
                gold_index = int(matches[0].item())

            signal_scores = adapter.get_signal_scores(prefix, candidate_ids)

            records.append({
                "base_logits": top_logits.cpu(),
                "signal_scores": signal_scores.cpu(),
                "candidate_ids": candidate_ids.cpu(),
                "gold_token_id": gold_id,
                "gold_index": gold_index,
                "position": position,
                "sample_id": sample_id,
            })

            prefix = adapter.append_token(prefix, gold_token)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)
    print(f"Saved {len(records)} token examples -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT, args.method)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_adapter(args.method)(config, PROJECT_ROOT, device)
    adapter.load_models()

    data_dir = PROJECT_ROOT / "dataset" / "router_train" / args.method
    cache_dir = PROJECT_ROOT / "dataset" / "router_cache" / args.method

    cache_split(load_jsonl(data_dir / "train.jsonl"), adapter, config, cache_dir / "train.pt")
    cache_split(load_jsonl(data_dir / "validation.jsonl"), adapter, config, cache_dir / "validation.pt")


if __name__ == "__main__":
    main()
