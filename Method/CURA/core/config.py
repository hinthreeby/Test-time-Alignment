from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "Method/CURA/configs/default.json"


def deep_merge(base, override):
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        elif key != "extends":
            output[key] = value
    return output


def _load_recursive(path, seen):
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic CURA config inheritance: {path}")
    seen.add(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("extends"):
        parent = path.parent / payload["extends"]
        payload = deep_merge(_load_recursive(parent, seen), payload)
    return payload


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = _load_recursive(path, set())
    payload["signal_objectives"] = {
        name: payload.get("signal_objectives", {}).get(name, "unknown")
        for name in payload["signals"]
    }
    payload["score_directions"] = {
        name: payload.get("score_directions", {}).get(name, "unknown")
        for name in payload["signals"]
    }
    return payload, path


def objective_mismatches(config):
    objective = config["objective"]
    objectives = config.get("signal_objectives", {})
    return {
        name: objectives.get(name, "unknown")
        for name in config["signals"]
        if objectives.get(name, "unknown") != objective
    }
