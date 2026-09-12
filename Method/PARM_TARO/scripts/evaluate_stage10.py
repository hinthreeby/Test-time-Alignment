"""CLI for Stage 10 preflight, protocol freeze, smoke, and full evaluation."""

from __future__ import annotations

import argparse
import json

import torch

from PARM_TARO.evaluation.config import ParmTaroEvaluationConfig
from PARM_TARO.evaluation.engine import freeze_protocol, run_evaluation, write_preflight
from router_v2.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="PARM_TARO/configs/evaluate_stage10_parm_taro.json",
    )
    parser.add_argument(
        "--phase", choices=("preflight", "freeze", "smoke", "full"), required=True
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ParmTaroEvaluationConfig.load_json(args.config)
    device_name = args.device or config.device
    allow_cpu_fallback = config.allow_cpu_fallback and not args.no_cpu_fallback
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if args.phase == "preflight":
        report = write_preflight(config)
    elif args.phase == "freeze":
        device = resolve_device(device_name, allow_cpu_fallback=allow_cpu_fallback)
        report = freeze_protocol(config, device=device)
    else:
        report = run_evaluation(
            config,
            args.phase,
            device=device_name,
            allow_cpu_fallback=allow_cpu_fallback,
            max_prompts=args.max_prompts,
            resume=args.resume,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") in {"NOT_PASS", "PREREQUISITES_REQUIRED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
