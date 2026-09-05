"""Train one Stage 9 PARM-TARO router stage."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.engine import train_stage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-dtype", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument(
        "--model-placement", choices=("accelerate_auto", "single_device")
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    config = ParmRouterTrainingConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    if args.model_dtype is not None:
        overrides["model_dtype"] = args.model_dtype
    if args.model_placement is not None:
        overrides["model_placement"] = args.model_placement
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_validation_samples is not None:
        overrides["max_validation_samples"] = args.max_validation_samples
    if overrides:
        config = replace(config, **overrides)
    report = train_stage(config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
