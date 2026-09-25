from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def validate_records(path, require_real_signals=False, require_objective_match=False,
                     require_calibration=False, paper_mode=False, mismatch_threshold=0.05):
    records, errors = [], []
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        rows = []
    else:
        try:
            payload = json.loads(content)
            rows = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            rows = [line for line in content.splitlines() if line.strip()]
    for line_number, item in enumerate(rows, 1):
        try:
            row = json.loads(item) if isinstance(item, str) else item
            if not isinstance(row, dict):
                raise TypeError("record is not a JSON object")
            records.append(row)
            sources = row.get("signal_sources", [])
            objectives = row.get("signal_objectives", {})
            if row.get("status") != "success":
                errors.append(f"record {line_number}: generation not successful")
            if require_real_signals and any("fallback" in source for source in sources):
                errors.append(f"record {line_number}: fallback signal present")
            if require_objective_match and any(value != row.get("objective") for value in objectives.values()):
                errors.append(f"record {line_number}: objective mismatch")
            if require_calibration and not row.get("calibration_mode"):
                errors.append(f"record {line_number}: calibration metadata absent")
            if paper_mode:
                if not row.get("paper_mode"):
                    errors.append(f"record {line_number}: checkpoint was not trained in paper mode")
                if row.get("calibration_mode") != "learned_heteroscedastic":
                    errors.append(f"record {line_number}: learned calibration required")
                if row.get("ablation", "none") != "none":
                    errors.append(f"record {line_number}: ablation output is not a main paper run")
                if not row.get("checkpoint_sha256") or not row.get("config_fingerprint"):
                    errors.append(f"record {line_number}: reproducibility provenance absent")
                provenance = row.get("training_data_provenance", {})
                for split in ("train", "validation"):
                    target = provenance.get(split, {}).get("target_utility") or {}
                    if target.get("status") != "complete":
                        errors.append(f"record {line_number}: {split} rollout targets are not complete")
                rates = row.get("tokenization_mismatch_rate", {})
                if any(float(rate) > mismatch_threshold + 1e-12 for rate in rates.values()):
                    errors.append(f"record {line_number}: tokenizer mismatch exceeds threshold")
        except Exception as error:
            errors.append(f"record {line_number}: {type(error).__name__}: {error}")
    return {"valid": not errors, "records": len(records), "errors": errors[:100]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--require-real-signals", action="store_true")
    parser.add_argument("--require-objective-match", action="store_true")
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--paper-mode", action="store_true")
    parser.add_argument("--tokenization-mismatch-threshold", type=float, default=0.05)
    args = parser.parse_args()
    report = validate_records(
        args.input, args.require_real_signals or args.paper_mode,
        args.require_objective_match or args.paper_mode,
        args.require_calibration or args.paper_mode, args.paper_mode,
        args.tokenization_mismatch_threshold,
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
