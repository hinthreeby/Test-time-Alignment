#!/usr/bin/env python3
"""Inspect the local Amazon Polarity saved dataset without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from router.scripts.data.build_router_amazon_polarity import count_labels, inspect_source


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset/RAD_train/amazon_polarity"))
    parser.add_argument("--count-labels", action="store_true")
    args = parser.parse_args(argv)

    inspection = inspect_source(args.source_dir)
    payload = inspection.__dict__.copy()
    if args.count_labels:
        payload["label_counts"] = {
            split: dict(sorted(count_labels(args.source_dir, split).items()))
            for split in inspection.splits
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
