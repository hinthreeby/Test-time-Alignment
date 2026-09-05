"""Train the full Stage 9 V3 alpha-preference Router with atomic resume."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from PARM_TARO.training.alpha_production import (
    train_alpha_preference_production,
)
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import PROJECT_ROOT


DEFAULT_CONFIG = (
    PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha_preference.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = AlphaPreferenceProductionConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if overrides:
        config = replace(config, **overrides)
    report = train_alpha_preference_production(config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
