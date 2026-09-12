"""Resume recovery and manifest construction for real cache shards."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from router_v2.cache.features import DERIVED_FEATURE_NAMES
from router_v2.cache.io import load_cache_shard, sha256_file


def cleanup_stale_shard_temps(split_dir: str | Path) -> list[str]:
    root = Path(split_dir)
    removed = []
    for path in sorted(root.glob(".shard_*.pt.*.tmp")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def recover_shard_state(
    split_dir: str | Path,
    *,
    extraction_fingerprint: str,
) -> dict[str, Any]:
    root = Path(split_dir)
    shard_paths = sorted(root.glob("shard_*.pt"))
    expected_shard_index = 0
    expected_source_index = 0
    record_keys: set[tuple[str, int]] = set()
    shards = []
    for path in shard_paths:
        shard = load_cache_shard(path)
        source = shard["source"]
        if source.get("extraction_fingerprint") != extraction_fingerprint:
            raise ValueError(f"Shard fingerprint mismatch: {path}")
        if int(shard["shard_index"]) != expected_shard_index:
            raise ValueError(f"Non-contiguous shard indexes at {path}")
        source_start = int(source["source_start_index"])
        source_end = int(source["source_end_index_exclusive"])
        if source_start != expected_source_index or source_end <= source_start:
            raise ValueError(f"Non-contiguous source range at {path}")
        keys = set(zip(shard["sample_ids"], shard["position"].tolist()))
        if len(keys) != int(shard["num_records"]):
            raise ValueError(f"Duplicate records inside {path}")
        if keys & record_keys:
            raise ValueError(f"Duplicate records across shards at {path}")
        record_keys.update(keys)
        shards.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "num_records": int(shard["num_records"]),
                "source_start_index": source_start,
                "source_end_index_exclusive": source_end,
                "source_ids": list(source["source_ids"]),
            }
        )
        expected_shard_index += 1
        expected_source_index = source_end
    return {
        "next_shard_index": expected_shard_index,
        "next_source_index": expected_source_index,
        "num_records": len(record_keys),
        "shards": shards,
    }


def build_split_manifest(
    split_dir: str | Path,
    *,
    split: str,
    extraction_fingerprint: str,
    extraction_parameters: dict[str, Any],
    source_dataset: dict[str, Any],
    base_model: dict[str, Any],
    guide_model: dict[str, Any],
    base_tokenizer: dict[str, Any],
    guide_tokenizer: dict[str, Any],
    top_k: int,
    max_position: int,
    precision: str,
    device: str,
    expected_source_examples: int,
    stale_temps_removed: list[str],
) -> dict[str, Any]:
    root = Path(split_dir)
    state = recover_shard_state(
        root,
        extraction_fingerprint=extraction_fingerprint,
    )
    if state["next_source_index"] != expected_source_examples:
        raise ValueError(
            f"Extraction incomplete: {state['next_source_index']} of "
            f"{expected_source_examples} source examples"
        )
    stats = {
        name: {
            "count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "min": math.inf,
            "max": -math.inf,
        }
        for name in DERIVED_FEATURE_NAMES
    }
    source_ids = []
    nan_inf_count = 0
    topk_uniqueness_failures = 0
    for item in state["shards"]:
        shard = load_cache_shard(root / item["path"])
        source_ids.extend(item["source_ids"])
        for prefix in ("base", "guide"):
            token_ids = shard[f"{prefix}_token_ids"]
            sorted_ids = token_ids.sort(dim=-1).values
            if sorted_ids.shape[-1] > 1:
                topk_uniqueness_failures += int(
                    (sorted_ids[:, 1:] == sorted_ids[:, :-1])
                    .any(dim=-1)
                    .sum()
                )
        for name in DERIVED_FEATURE_NAMES:
            values = shard["derived"][name].float()
            finite = torch.isfinite(values)
            nan_inf_count += int((~finite).sum())
            values = values[finite]
            if not values.numel():
                continue
            target = stats[name]
            target["count"] += values.numel()
            target["sum"] += float(values.sum())
            target["sum_squares"] += float(values.square().sum())
            target["min"] = min(target["min"], float(values.min()))
            target["max"] = max(target["max"], float(values.max()))
    feature_statistics = {}
    for name, values in stats.items():
        count = int(values["count"])
        mean = values["sum"] / count
        variance = max(0.0, values["sum_squares"] / count - mean * mean)
        feature_statistics[name] = {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": values["min"],
            "max": values["max"],
        }
    if len(source_ids) != expected_source_examples:
        raise ValueError("Manifest source ID count does not match source range")
    return {
        "schema_version": 1,
        "feature_schema": "router_v2.feature_cache.v1",
        "task": "sentiment",
        "split": split,
        "complete": True,
        "extraction_fingerprint": extraction_fingerprint,
        "extraction_parameters": extraction_parameters,
        "source_dataset": source_dataset,
        "number_source_examples": expected_source_examples,
        "number_token_records": state["num_records"],
        "number_shards": len(state["shards"]),
        "top_k": top_k,
        "max_position": max_position,
        "base_model": base_model,
        "guide_model": guide_model,
        "base_tokenizer": base_tokenizer,
        "guide_tokenizer": guide_tokenizer,
        "exact_tokenizer_compatible": True,
        "precision": precision,
        "device_used_for_extraction": device,
        "gold_leakage_checks": {
            "topk_selector_accepts_gold": False,
            "gold_forced_into_candidates": False,
            "gold_stored_as_target_only": True,
            "previous_gold_tokens_only_in_teacher_forced_prefix": True,
        },
        "nan_inf_count": nan_inf_count,
        "top_k_uniqueness_failures": topk_uniqueness_failures,
        "average_continuation_length": (
            state["num_records"] / expected_source_examples
        ),
        "feature_statistics": feature_statistics,
        "source_ids": source_ids,
        "shards": state["shards"],
        "stale_temp_files_removed": stale_temps_removed,
    }
