"""Materialize deterministic local PKU multi-objective data for PARM-TARO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PARM_TARO.data.multi_objective import (
    DEFAULT_SPLIT_SEED,
    prepare_multi_objective_dataset,
    validate_processed_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "dataset"
    / "GenARM"
    / "PKU-SafeRLHF-10K"
    / "round0"
    / "train.jsonl.xz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "parm_taro"
DEFAULT_PARM_DATA = PROJECT_ROOT / "PARM" / "code" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parm-data", type=Path, default=DEFAULT_PARM_DATA)
    parser.add_argument("--train-records", type=int, default=8000)
    parser.add_argument("--validation-records", type=int, default=500)
    parser.add_argument("--seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_processed_dataset(
            args.output,
            source_path=args.source,
        )
    else:
        report = prepare_multi_objective_dataset(
            source_path=args.source,
            output_dir=args.output,
            parm_data_path=args.parm_data,
            train_records=args.train_records,
            validation_records=args.validation_records,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
