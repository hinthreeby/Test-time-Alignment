"""Update cache status only after strict smoke or full-cache evidence passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from router_v2.cache.io import (
    require_path_within,
    sha256_file,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "dataset" / "router_v2_cache"
REPORT_ROOT = PROJECT_ROOT / "router_v2" / "reports"
DEFAULT_STATUS = CACHE_ROOT / "cache_status.json"
DEFAULT_VALIDATION = REPORT_ROOT / "sentiment_guide_validation.json"
DEFAULT_SMOKE_AUDIT = REPORT_ROOT / "stage3_smoke_fp32_audit.json"
DEFAULT_FULL_AUDIT = REPORT_ROOT / "stage3_recomputation_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--smoke-audit", type=Path, default=DEFAULT_SMOKE_AUDIT)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def _load_pass(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS":
        raise ValueError(f"{label} is not PASS: {path}")
    return report


def _load_manifest(root: Path, split: str) -> dict[str, Any]:
    path = root / split / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not manifest.get("complete")
        or manifest.get("task") != "sentiment"
        or manifest.get("nan_inf_count") != 0
        or manifest.get("top_k_uniqueness_failures") != 0
        or int(manifest.get("number_shards", 0)) <= 0
        or manifest.get("precision") != "fp32"
        or manifest.get("extraction_parameters", {}).get("precision")
        != "fp32"
        or manifest.get("extraction_parameters", {}).get(
            "explicit_position_ids"
        )
        is not True
        or manifest.get("extraction_parameters", {}).get(
            "batch_atomic_shards"
        )
        is not True
        or manifest.get("extraction_parameters", {})
        .get("deterministic_runtime", {})
        .get("deterministic_algorithms")
        is not True
    ):
        raise ValueError(f"Invalid or incomplete manifest: {path}")
    for shard in manifest["shards"]:
        shard_path = root / split / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"Shard hash mismatch: {shard_path}")
    return manifest


def _verify_audit(
    audit: dict[str, Any],
    root: Path,
    manifests: dict[str, dict[str, Any]],
) -> None:
    if Path(audit.get("cache_root", "")).resolve() != root.resolve():
        raise ValueError(f"Audit cache root does not match {root}")
    if (
        int(audit.get("schema_version", 0)) < 2
        or audit.get("precision") != "fp32"
        or float(audit.get("absolute_tolerance", float("inf"))) > 1e-5
        or not audit.get("checks", {}).get("online_values_match_cache")
        or not audit.get("checks", {}).get("model_outputs_fp32")
    ):
        raise ValueError("Audit does not satisfy the scientific FP32 gate")
    for split, manifest in manifests.items():
        item = audit.get("splits", {}).get(split, {})
        manifest_path = root / split / "manifest.json"
        if (
            not item.get("pass")
            or int(item.get("records_audited", 0)) < 20
            or item.get("manifest_sha256") != sha256_file(manifest_path)
            or int(item.get("number_source_examples", -1))
            != int(manifest["number_source_examples"])
            or int(item.get("mismatch_count", -1)) != 0
        ):
            raise ValueError(f"Audit evidence does not match {split} cache")


def main() -> None:
    args = parse_args()
    validation = _load_pass(args.validation, "guide validation")
    smoke_audit = _load_pass(args.smoke_audit, "smoke audit")
    smoke_root = CACHE_ROOT / "rad_smoke_fp32"
    smoke_manifests = {
        split: _load_manifest(smoke_root, split)
        for split in ("train", "validation")
    }
    for split, manifest in smoke_manifests.items():
        if int(manifest["number_source_examples"]) != 20:
            raise ValueError(f"Smoke {split} must contain exactly 20 samples")
    _verify_audit(smoke_audit, smoke_root, smoke_manifests)

    full_manifests = None
    full_audit = None
    if args.phase == "full":
        full_audit = _load_pass(args.full_audit, "full recomputation audit")
        full_root = CACHE_ROOT / "rad"
        full_manifests = {
            split: _load_manifest(full_root, split)
            for split in ("train", "validation")
        }
        expected = {"train": 20000, "validation": 2000}
        for split, count in expected.items():
            if int(full_manifests[split]["number_source_examples"]) != count:
                raise ValueError(
                    f"Full {split} cache must contain {count} source samples"
                )
        _verify_audit(full_audit, full_root, full_manifests)

    tensor_shards = sum(
        int(manifest["number_shards"])
        for manifest in smoke_manifests.values()
    )
    if full_manifests is not None:
        tensor_shards += sum(
            int(manifest["number_shards"])
            for manifest in full_manifests.values()
        )
    status_path = require_path_within(
        args.status,
        CACHE_ROOT,
        label="cache status",
    )
    existing = {}
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    stage3_pass = args.phase == "full"
    status = {
        **existing,
        "schema_version": 1,
        "cache_format": "router_v2.feature_cache",
        "stage3_status": (
            "PASS" if stage3_pass else "HOST_EXECUTION_REQUIRED_FULL_CACHE"
        ),
        "tensor_shards_generated": tensor_shards,
        "rad": {
            "source": "dataset/RAD_train/router_amazon_polarity",
            "source_ready": True,
            "guide_logits_ready": True,
            "smoke_cache_ready": True,
            "train_cache_ready": stage3_pass,
            "validation_cache_ready": stage3_pass,
            "task": "sentiment",
            "precision": "fp32",
            "guide_validation": {
                "path": str(args.validation),
                "sha256": sha256_file(args.validation),
            },
            "smoke_audit": {
                "path": str(args.smoke_audit),
                "sha256": sha256_file(args.smoke_audit),
            },
            "full_audit": (
                {
                    "path": str(args.full_audit),
                    "sha256": sha256_file(args.full_audit),
                }
                if full_audit is not None
                else None
            ),
            "base_model": validation["base_model"],
            "guide_model": validation["guide_model"],
            "legacy_fp16_smoke": existing.get("rad", {}).get(
                "legacy_fp16_smoke"
            ),
        },
        "safety": {
            "v1_cache_modified": False,
            "gold_used_in_router_features": False,
            "normalized_position_uses_fixed_max_position": True,
            "scientific_task_alignment": "sentiment",
            "legacy_fp16_smoke_excluded": True,
        },
    }
    write_json_atomic(status_path, status, overwrite=status_path.exists())
    print(
        json.dumps(
            {
                "stage3_status": status["stage3_status"],
                "tensor_shards_generated": tensor_shards,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
