"""Train one ordered Stage 5 Router V2 phase with frozen online logits."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from router_v2.training.config import RouterTrainingConfig
from router_v2.training.engine import train_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--skip-stage-order", action="store_true")
    parser.add_argument("--no-same-average", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RouterTrainingConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_validation_samples is not None:
        overrides["max_validation_samples"] = args.max_validation_samples
    if args.skip_stage_order:
        overrides["enforce_stage_order"] = False
    if args.no_same_average:
        overrides["same_average_baseline"] = False
    if overrides:
        config = replace(config, **overrides)
    result = train_router(config, resume=args.resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "best_validation_nll": result["best_validation_nll"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
