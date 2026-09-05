"""Pack independent full-logit tensors into a validated Router V2 cache shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from router_v2.cache.io import require_path_within, save_cache_shard
from router_v2.cache.schema import build_cache_shard_from_full_logits


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pt", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-position", type=int, default=1023)
    return parser.parse_args()


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Input full-logit payload is missing {key!r}")
    return payload[key]


def main() -> None:
    args = parse_args()
    output_path = require_path_within(
        args.output_pt,
        PROJECT_ROOT / "dataset" / "router_v2_cache",
        label="cache shard output",
    )
    if output_path.suffix != ".pt":
        raise ValueError("cache shard output must use the .pt suffix")

    raw = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Input full-logit file must contain a mapping")
    source = dict(raw.get("source", {}))
    source["full_logit_input"] = str(args.input_pt)
    shard = build_cache_shard_from_full_logits(
        split=args.split,
        shard_index=args.shard_index,
        sample_ids=_require(raw, "sample_ids"),
        position=_require(raw, "position"),
        base_full_logits=_require(raw, "base_full_logits"),
        guide_full_logits=_require(raw, "guide_full_logits"),
        gold_token_id=_require(raw, "gold_token_id"),
        top_k=args.top_k,
        max_position=args.max_position,
        source=source,
        preference_vector=raw.get("preference_vector"),
    )
    summary = save_cache_shard(shard, output_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
