"""Prepare the deterministic Gate-A manifest and enforce File-13 prerequisites."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "parm_taro" / "recovery" / "val200"
SEED_LABEL = "parm_taro_recovery_val200_seed_2026"
ALPHA_GRID = [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_hash(path) if path.is_file() else None,
    }


def selection_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED_LABEL}\0{sample_id}".encode()).hexdigest()


def main() -> None:
    validation_path = ROOT / "dataset/parm_taro/validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(validation, list) or len(validation) != 500:
        raise ValueError("File-13 Gate B requires exactly 500 validation rows")
    if len({row["sample_id"] for row in validation}) != len(validation):
        raise ValueError("Validation sample IDs must be unique")
    selected = sorted(validation, key=lambda row: selection_key(row["sample_id"]))[:200]
    manifest = {
        "schema_version": 1,
        "split": "validation",
        "selection": "lowest_sha256(seed_label + NUL + sample_id)",
        "seed_label": SEED_LABEL,
        "source": {
            "path": "dataset/parm_taro/validation.json",
            "records": len(validation),
            "sha256": file_hash(validation_path),
        },
        "records": len(selected),
        "alpha_order": ["helpfulness", "harmlessness"],
        "alpha_grid": ALPHA_GRID,
        "sample_ids": [row["sample_id"] for row in selected],
        "source_indices": [row["source_index"] for row in selected],
        "test_records_read": 0,
    }

    required = [
        descriptor("models/tulu-2-7b/config.json"),
        descriptor("results/parm_taro/checkpoints/parm_pku_pblora/adapter_config.json"),
        descriptor("results/parm_taro/training/taro/best.pt"),
        descriptor("results/parm_taro/training/v2_no_alpha/best.pt"),
        descriptor("results/parm_taro/training/v2_alpha_preference/final.pt"),
        descriptor("results/parm_taro/recovery/checkpoints/recovered_full_alpha.pt"),
    ]
    protocol = descriptor("results/parm_taro/evaluation/protocol/protocol_lock.json")
    missing = [item["path"] for item in required if not item["present"]]
    preflight = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Gate A / val200 preflight",
        "status": "READY" if not missing else "BLOCKED",
        "generation_started": False,
        "scoring_started": False,
        "test_split_accessed": False,
        "required_assets": required,
        "missing_assets": missing,
        "protocol_lock": protocol,
        "protocol": {
            "methods": [
                "parm_static",
                "old_v2_full_alpha_stage10",
                "recovered_v2_full_alpha",
                "recovered_same_average_lambda",
                "recovered_shuffled_alpha",
                "recovered_no_alpha",
            ],
            "alpha_grid": ALPHA_GRID,
            "normalization": "reuse frozen Stage-10 validation normalization",
            "hv_reference": [0.0, 0.0],
            "regret_oracle": "shared all-method candidate pool per prompt/alpha/seed",
        },
        "manifest_sha256": None,
        "stop_reason": (
            None
            if not missing
            else "File 13 forbids Gate A generation before a recovered checkpoint and complete frozen runtime are present."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preflight["manifest_sha256"] = file_hash(manifest_path)
    (OUT / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
