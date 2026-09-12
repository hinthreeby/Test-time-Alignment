"""Run the protected PARM trainer with HF-Trainer checkpoint resume support.

The protected source calls ``trainer.train()`` without a resume argument.  This
recovery-only entrypoint injects the argument at runtime and does not edit PARM/.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    configured = os.environ.get("TTA_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / ".git").exists():
            return root
        raise RuntimeError(f"TTA_PROJECT_ROOT is not a project root: {root}")
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Cannot locate project root containing .git")


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)

    if args.resume_from_checkpoint:
        from transformers import Trainer

        original = Trainer.train

        def train_with_resume(self, *positional, **keywords):  # type: ignore[no-untyped-def]
            keywords.setdefault("resume_from_checkpoint", args.resume_from_checkpoint)
            return original(self, *positional, **keywords)

        Trainer.train = train_with_resume  # type: ignore[method-assign]

    script = ROOT / "PARM/code/training/train_pref_arm.py"
    sys.argv = [str(script), *trainer_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
