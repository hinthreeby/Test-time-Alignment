"""Audit Stage 9 training prerequisites without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.runtime import audit_training_prerequisites


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "PARM_TARO/reports/stage9_prerequisite_audit.json"


def _write_atomic(path: Path, value: dict[str, object]) -> None:
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
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_training_prerequisites(
        ParmRouterTrainingConfig.load_json(args.config)
    )
    _write_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
