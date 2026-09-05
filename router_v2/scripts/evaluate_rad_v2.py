"""Evaluate Stage 5 TARO/Smart checkpoints against read-only RAD V1 baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from router_v2.evaluation.rad.config import RADEvaluationConfig
from router_v2.evaluation.rad.engine import run_rad_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--validation-limit-per-class", type=int)
    parser.add_argument("--test-limit-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RADEvaluationConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.validation_limit_per_class is not None:
        overrides["validation_prompt_limit_per_class"] = (
            args.validation_limit_per_class
        )
    if args.test_limit_per_class is not None:
        overrides["test_prompt_limit_per_class"] = args.test_limit_per_class
    if overrides:
        config = replace(config, **overrides)
    report = run_rad_evaluation(config, resume=args.resume)
    print(
        json.dumps(
            {
                "status": report["status"],
                "record_counts": report["record_counts"],
                "checks": report["checks"],
                "parm_readiness": report["parm_readiness"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

