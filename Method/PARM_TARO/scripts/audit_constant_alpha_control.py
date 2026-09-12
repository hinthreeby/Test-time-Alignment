"""Re-evaluate Stage 9 constant-alpha sensitivity without retraining."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from PARM_TARO.training.constant_control_audit import (
    audit_constant_alpha_control,
    render_constant_control_audit,
)
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import PROJECT_ROOT
from router_v2.cache.io import write_json_atomic


DEFAULT_CONFIG = (
    PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha_preference.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_constant_alpha_control_audit.json"
)


def _write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--progress-every-tasks", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = AlphaPreferenceProductionConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if overrides:
        config = replace(config, **overrides)
    report = audit_constant_alpha_control(
        config,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        progress_every_tasks=args.progress_every_tasks,
    )
    write_json_atomic(args.output, report, overwrite=args.overwrite)
    _write_text_atomic(
        args.output.with_suffix(".md"),
        render_constant_control_audit(report),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
