"""Evaluate and persist the final Stage 9 production gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.runtime import PROJECT_ROOT
from PARM_TARO.training.stage9_final import (
    build_stage9_final_gate,
    render_stage9_final_markdown,
)
from router_v2.cache.io import write_json_atomic


DEFAULT_CONFIG = (
    PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha_preference.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "PARM_TARO/reports/stage9_final_gate.json"


def _write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final report: {path}")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = AlphaPreferenceProductionConfig.load_json(args.config)
    report = build_stage9_final_gate(config)
    write_json_atomic(args.output, report, overwrite=args.overwrite)
    markdown_path = args.output.with_suffix(".md")
    _write_text_atomic(
        markdown_path,
        render_stage9_final_markdown(report),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "STAGE 9 PASS" and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
