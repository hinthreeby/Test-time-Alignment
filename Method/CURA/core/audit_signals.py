from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.config import PROJECT_ROOT, load_config, objective_mismatches


ARTIFACTS = {
    "genarm": "models/genarm-gpt2-small-hh",
    "rad": "models/rad_rm_sentiment/pytorch_model.bin",
    "cdq": "Method/CD/checkpoints/cd_q.pt",
    "args": "models/sentiment-roberta-large-english",
}


def audit(config):
    artifacts = {}
    for signal in config["signals"]:
        path = PROJECT_ROOT / ARTIFACTS[signal]
        artifacts[signal] = {"path": str(path), "exists": path.exists()}
    mismatches = objective_mismatches(config)
    errors = [f"missing checkpoint for {name}" for name, item in artifacts.items() if not item["exists"]]
    errors.extend(f"{name} objective is {value}, expected {config['objective']}" for name, value in mismatches.items())
    errors.extend(
        f"{name} score direction is not declared higher_is_better"
        for name in config["signals"] if config.get("score_directions", {}).get(name) != "higher_is_better"
    )
    return {
        "valid": not errors,
        "objective": config["objective"],
        "signal_objectives": {name: config["signal_objectives"].get(name, "unknown") for name in config["signals"]},
        "score_directions": config["score_directions"],
        "artifacts": artifacts,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Method/CURA/configs/sentiment.json")
    parser.add_argument("--allow-objective-mismatch", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    report = audit(config)
    print(json.dumps(report, indent=2))
    blocking = [error for error in report["errors"] if "objective is" not in error or not args.allow_objective_mismatch]
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
