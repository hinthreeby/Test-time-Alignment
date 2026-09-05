"""Stream real teacher-forced logits directly into Router V2 Top-K shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from router_v2.cache.io import (
    require_path_within,
    save_cache_shard,
    sha256_file,
    write_json_atomic,
)
from router_v2.cache.schema import build_cache_shard
from router_v2.device import resolve_device
from router_v2.guide_model.cache_manifest import (
    build_split_manifest,
    cleanup_stale_shard_temps,
    recover_shard_state,
)
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.data import load_rad_samples
from router_v2.guide_model.extraction import (
    ExtractedSampleRecords,
    combine_extracted_records,
    model_topk,
    prepare_extraction_batch,
    should_flush_before_extraction_batch,
)
from router_v2.guide_model.provenance import (
    checkpoint_descriptor,
    tokenizer_descriptor,
)
from router_v2.guide_model.runtime import (
    configure_deterministic_inference,
    load_compatible_tokenizer_pair,
    load_frozen_causal_model_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "router_v2" / "configs" / "sentiment_guide.json"
DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "router_v2"
    / "reports"
    / "sentiment_guide_validation.json"
)
DEFAULT_SMOKE_AUDIT = (
    PROJECT_ROOT
    / "router_v2"
    / "reports"
    / "stage3_smoke_fp32_audit.json"
)
CACHE_ROOT = PROJECT_ROOT / "dataset" / "router_v2_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--guide-adapter", type=Path)
    parser.add_argument("--guide-validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--smoke-audit", type=Path, default=DEFAULT_SMOKE_AUDIT)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "fp32"),
        default="fp32",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-continuation-tokens", type=int, default=80)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_pass_report(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS":
        raise ValueError(f"{label} is not PASS: {path}")
    return report


def _require_smoke_gate(report: dict[str, Any]) -> None:
    expected_root = (CACHE_ROOT / "rad_smoke_fp32").resolve()
    if Path(report.get("cache_root", "")).resolve() != expected_root:
        raise ValueError("Smoke audit does not describe rad_smoke_fp32")
    if report.get("precision") != "fp32":
        raise ValueError("Scientific smoke audit must use FP32")
    if float(report.get("absolute_tolerance", float("inf"))) > 1e-5:
        raise ValueError("Scientific smoke audit tolerance exceeds 1e-5")
    checks = report.get("checks", {})
    if (
        not checks.get("online_values_match_cache")
        or not checks.get("model_outputs_fp32")
    ):
        raise ValueError("Scientific smoke recomputation checks did not pass")
    split_reports = report.get("splits", {})
    for split in ("train", "validation"):
        item = split_reports.get(split, {})
        if (
            not item.get("pass")
            or int(item.get("number_source_examples", -1)) != 20
            or int(item.get("records_audited", 0)) < 20
        ):
            raise ValueError(
                f"Smoke audit lacks a passing 20-sample {split} split"
            )
        manifest_path = expected_root / split / "manifest.json"
        if item.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"Smoke audit manifest hash is stale for {split}")


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _concatenate_records(
    records: list[ExtractedSampleRecords],
) -> dict[str, Any]:
    return {
        "sample_ids": [
            record.sample.sample_id
            for record in records
            for _ in range(record.num_records)
        ],
        "position": torch.cat([record.position for record in records]),
        "base_token_ids": torch.cat(
            [record.base_token_ids for record in records]
        ),
        "base_logits": torch.cat(
            [record.base_logits for record in records]
        ),
        "guide_token_ids": torch.cat(
            [record.guide_token_ids for record in records]
        ),
        "guide_logits": torch.cat(
            [record.guide_logits for record in records]
        ),
        "gold_token_id": torch.cat(
            [record.gold_token_id for record in records]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    config = SentimentGuideConfig.load_json(args.config)
    top_k = args.top_k or config.cache_top_k
    shard_size = args.shard_size or config.cache_shard_size
    if args.max_continuation_tokens > config.cache_max_position + 1:
        raise ValueError(
            "max-continuation-tokens exceeds configured position range"
        )
    if shard_size <= args.max_continuation_tokens:
        raise ValueError(
            "shard-size must exceed max-continuation-tokens"
        )
    output_root = require_path_within(
        args.output_root,
        CACHE_ROOT,
        label="feature cache output root",
    )
    scientific_root = output_root.name in {"rad_smoke_fp32", "rad"}
    if output_root.name == "rad_smoke_fp32" and args.max_samples != 20:
        raise ValueError("rad_smoke_fp32 requires exactly --max-samples 20")
    if output_root.name == "rad" and args.max_samples is not None:
        raise ValueError("Persistent rad cache must use the complete split")
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        if args.resume:
            print(f"Split already complete: {manifest_path}")
            return
        raise FileExistsError(f"Cache manifest already exists: {manifest_path}")

    guide_validation = _load_pass_report(
        args.guide_validation,
        "sentiment guide validation",
    )
    full_persistent_run = output_root.name == "rad" and args.max_samples is None
    if full_persistent_run:
        smoke_audit = _load_pass_report(
            args.smoke_audit,
            "smoke cache audit",
        )
        _require_smoke_gate(smoke_audit)

    device = resolve_device(
        args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    precision = args.precision
    if precision == "auto":
        precision = "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 cache extraction requires CUDA")
    if scientific_root and precision != "fp32":
        raise ValueError(
            "Scientific rad_smoke_fp32/rad caches require FP32 inference"
        )
    use_fp16 = precision == "fp16"
    deterministic_runtime = configure_deterministic_inference(
        seed=config.seed,
        device=device,
    )

    base_path = _project_path(config.base_model_path)
    guide_path = (
        args.guide_adapter.resolve()
        if args.guide_adapter
        else _project_path(config.output_dir) / "final_adapter"
    )
    source_path = _project_path(
        config.train_data_path
        if args.split == "train"
        else config.validation_data_path
    )
    base_checkpoint = checkpoint_descriptor(base_path)
    guide_checkpoint = checkpoint_descriptor(guide_path)
    if (
        guide_validation["base_model"]["checkpoint_sha256"]
        != base_checkpoint["checkpoint_sha256"]
        or guide_validation["guide_model"]["checkpoint_sha256"]
        != guide_checkpoint["checkpoint_sha256"]
    ):
        raise ValueError("Guide validation checkpoint hashes are stale")

    tokenizer_pair = load_compatible_tokenizer_pair(base_path, guide_path)
    tokenizer = tokenizer_pair.base
    guide_tokenizer = tokenizer_pair.guide
    base_tokenizer_info = tokenizer_descriptor(tokenizer, base_path)
    guide_tokenizer_info = tokenizer_descriptor(guide_tokenizer, guide_path)
    samples = load_rad_samples(source_path, max_samples=args.max_samples)

    fingerprint_payload = {
        "schema_version": 1,
        "task": "sentiment",
        "split": args.split,
        "source_sha256": sha256_file(source_path),
        "base_checkpoint_sha256": base_checkpoint["checkpoint_sha256"],
        "guide_checkpoint_sha256": guide_checkpoint["checkpoint_sha256"],
        "base_tokenizer_sha256": base_tokenizer_info["semantic_sha256"],
        "guide_tokenizer_sha256": guide_tokenizer_info["semantic_sha256"],
        "top_k": top_k,
        "max_position": config.cache_max_position,
        "max_continuation_tokens": args.max_continuation_tokens,
        "batch_size": args.batch_size,
        "shard_size": shard_size,
        "precision": precision,
        "device_type": device.type,
        "deterministic_runtime": deterministic_runtime,
        "explicit_position_ids": True,
        "batch_atomic_shards": True,
        "num_source_examples": len(samples),
    }
    extraction_fingerprint = _fingerprint(fingerprint_payload)
    stale_temps = cleanup_stale_shard_temps(split_dir)
    state = recover_shard_state(
        split_dir,
        extraction_fingerprint=extraction_fingerprint,
    )
    if state["shards"] and not args.resume:
        raise FileExistsError(
            "Existing cache shards require --resume or a new output root"
        )
    if state["next_source_index"] > len(samples):
        raise ValueError("Existing shards exceed requested source sample count")

    model_pair = load_frozen_causal_model_pair(
        base_path,
        guide_path,
        device=device,
        precision=precision,
    )
    base_model = model_pair.base
    guide_model = model_pair.guide

    source_common = {
        "task": "sentiment",
        "source_dataset_path": str(source_path),
        "source_dataset_sha256": sha256_file(source_path),
        "base_model": base_checkpoint,
        "guide_model": guide_checkpoint,
        "base_tokenizer": base_tokenizer_info,
        "guide_tokenizer": guide_tokenizer_info,
        "exact_tokenizer_compatible": True,
        "teacher_forcing": True,
        "gold_forced_into_topk": False,
        "precision": precision,
        "device": str(device),
        "deterministic_runtime": deterministic_runtime,
        "explicit_position_ids": True,
        "batch_atomic_shards": True,
        "extraction_fingerprint": extraction_fingerprint,
        "extraction_parameters": fingerprint_payload,
    }
    shard_index = int(state["next_shard_index"])
    buffer: list[ExtractedSampleRecords] = []
    buffer_records = 0

    def flush_buffer() -> None:
        nonlocal shard_index, buffer, buffer_records
        if not buffer:
            return
        combined = _concatenate_records(buffer)
        source_start = samples.index(buffer[0].sample)
        source_end = samples.index(buffer[-1].sample) + 1
        source_ids = [record.sample.sample_id for record in buffer]
        shard = build_cache_shard(
            split=args.split,
            shard_index=shard_index,
            sample_ids=combined["sample_ids"],
            position=combined["position"],
            base_token_ids=combined["base_token_ids"],
            base_logits=combined["base_logits"],
            guide_token_ids=combined["guide_token_ids"],
            guide_logits=combined["guide_logits"],
            gold_token_id=combined["gold_token_id"],
            max_position=config.cache_max_position,
            vocab_size=len(tokenizer),
            source={
                **source_common,
                "source_start_index": source_start,
                "source_end_index_exclusive": source_end,
                "source_ids": source_ids,
            },
        )
        path = split_dir / f"shard_{shard_index:05d}.pt"
        summary = save_cache_shard(shard, path)
        shard_index += 1
        progress = {
            "schema_version": 1,
            "extraction_fingerprint": extraction_fingerprint,
            "split": args.split,
            "next_source_index": source_end,
            "next_shard_index": shard_index,
            "latest_shard": summary,
        }
        progress_path = split_dir / "extraction_state.json"
        write_json_atomic(
            progress_path,
            progress,
            overwrite=progress_path.exists(),
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
        buffer = []
        buffer_records = 0

    start_index = int(state["next_source_index"])
    for batch_start in range(start_index, len(samples), args.batch_size):
        batch_samples = samples[
            batch_start : batch_start + args.batch_size
        ]
        prepared = prepare_extraction_batch(
            tokenizer,
            batch_samples,
            max_length=config.max_length,
            max_continuation_tokens=args.max_continuation_tokens,
        )
        base_candidates = model_topk(
            base_model,
            prepared,
            top_k=top_k,
            device=device,
            use_fp16=use_fp16,
        )
        guide_candidates = model_topk(
            guide_model,
            prepared,
            top_k=top_k,
            device=device,
            use_fp16=use_fp16,
        )
        extracted = combine_extracted_records(
            prepared,
            base_candidates,
            guide_candidates,
        )
        incoming_records = sum(record.num_records for record in extracted)
        if should_flush_before_extraction_batch(
            buffered_records=buffer_records,
            incoming_batch_records=incoming_records,
            shard_size=shard_size,
        ):
            flush_buffer()
        buffer.extend(extracted)
        buffer_records += incoming_records
        if buffer_records >= shard_size:
            flush_buffer()
    flush_buffer()

    manifest = build_split_manifest(
        split_dir,
        split=args.split,
        extraction_fingerprint=extraction_fingerprint,
        extraction_parameters=fingerprint_payload,
        source_dataset={
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        base_model=base_checkpoint,
        guide_model=guide_checkpoint,
        base_tokenizer=base_tokenizer_info,
        guide_tokenizer=guide_tokenizer_info,
        top_k=top_k,
        max_position=config.cache_max_position,
        precision=precision,
        device=str(device),
        expected_source_examples=len(samples),
        stale_temps_removed=stale_temps,
    )
    if manifest["nan_inf_count"] or manifest["top_k_uniqueness_failures"]:
        raise ValueError("Final cache manifest validation found invalid records")
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(manifest_path),
                "number_shards": manifest["number_shards"],
                "number_token_records": manifest["number_token_records"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
