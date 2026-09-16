"""Validate whether a cache is ready for the configured calibration mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.cache_io import ShardedCuraDataset, verify_cache
from Method.CURA.core.config import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Method/CURA/configs/sentiment.json")
    parser.add_argument("--cache-dir", default="dataset/cura_cache/validation")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    verify_cache(cache_dir)
    dataset = ShardedCuraDataset(cache_dir)
    mode = config["calibration"]["mode"]
    targeted = sum(dataset[index].get("target_utilities") is not None for index in range(len(dataset)))
    if mode == "learned_heteroscedastic" and targeted != len(dataset):
        raise RuntimeError("Learned calibration requires held-out target_utilities on every validation row")
    print(json.dumps({
        "mode": mode, "fit": "joint_with_controller" if mode == "learned_heteroscedastic" else "stateless",
        "validation_rows": len(dataset), "rows_with_targets": targeted, "ready": True,
    }, indent=2))


if __name__ == "__main__": main()
