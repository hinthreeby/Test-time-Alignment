"""Inventory local RAD/PKU sources and initialize isolated cache namespaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from router_v2.cache.inventory import inventory_cache_sources
from router_v2.cache.io import require_path_within, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "router_v2_train" / "source_inventory.json",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "router_v2_cache" / "cache_status.json",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Allow replacing V2 metadata only; tensor shards are never touched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = PROJECT_ROOT / "dataset" / "router_v2_cache"
    train_root = PROJECT_ROOT / "dataset" / "router_v2_train"
    inventory_output = require_path_within(
        args.inventory_output,
        train_root,
        label="inventory output",
    )
    status_output = require_path_within(
        args.status_output,
        cache_root,
        label="status output",
    )
    inventory = inventory_cache_sources(PROJECT_ROOT)
    for domain in ("rad", "parm"):
        for split in ("train", "validation"):
            (cache_root / domain / split).mkdir(parents=True, exist_ok=True)

    status = {
        "schema_version": 1,
        "cache_format": "router_v2.feature_cache",
        "stage3_status": "HOST_EXECUTION_REQUIRED",
        "tensor_shards_generated": 0,
        "rad": {
            "source": "dataset/RAD_train/router_amazon_polarity",
            "source_ready": inventory["rad"]["source_ready"],
            "guide_pipeline_ready": True,
            "guide_logits_ready": False,
            "smoke_cache_ready": False,
            "train_cache_ready": False,
            "validation_cache_ready": False,
            "status": "host_execution_required_sentiment_guide",
            "reason": (
                "The isolated sentiment-guide pipeline is implemented, but "
                "the trained adapter and real cache shards require host CUDA "
                "execution and validation."
            ),
        },
        "parm": {
            "source": "dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz",
            "schema_inventory_complete": inventory["pku_safe_rlhf"]["schema_inventory_complete"],
            "status": "blocked_missing_parm_model_prerequisites",
            "prerequisites": inventory["pku_safe_rlhf"]["parm_prerequisites"],
        },
        "safety": {
            "v1_cache_modified": False,
            "gold_used_in_router_features": False,
            "normalized_position_uses_fixed_max_position": True,
            "scientific_task_alignment": "sentiment",
        },
    }
    write_json_atomic(
        inventory_output,
        inventory,
        overwrite=args.overwrite_metadata,
    )
    write_json_atomic(
        status_output,
        status,
        overwrite=args.overwrite_metadata,
    )
    print(f"Wrote {inventory_output}")
    print(f"Wrote {status_output}")


if __name__ == "__main__":
    main()
