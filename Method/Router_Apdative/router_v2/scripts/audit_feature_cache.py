"""Recompute cached Router V2 records in their exact extraction batch context."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from router_v2.cache.features import compute_confidence_disagreement
from router_v2.cache.io import (
    load_cache_shard,
    require_path_within,
    sha256_file,
    write_json_atomic,
)
from router_v2.device import resolve_device
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.data import load_rad_samples
from router_v2.guide_model.extraction import (
    CandidateBatch,
    ExtractionBatch,
    model_topk,
    prepare_extraction_batch,
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
REPORT_ROOT = PROJECT_ROOT / "router_v2" / "reports"
CACHE_ROOT = PROJECT_ROOT / "dataset" / "router_v2_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--guide-adapter", type=Path)
    parser.add_argument("--guide-validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--records-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--fp16-atol", type=float, default=0.02)
    parser.add_argument("--fp32-atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _sample_cached_records(
    split_dir: Path,
    manifest: dict[str, Any],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    total = int(manifest["number_token_records"])
    if total < count:
        raise ValueError(
            f"Cache has only {total} token records; audit requires {count}"
        )
    selected_global = sorted(random.Random(seed).sample(range(total), count))
    selected_set = set(selected_global)
    records = []
    offset = 0
    for shard_info in manifest["shards"]:
        shard_path = split_dir / shard_info["path"]
        shard = load_cache_shard(shard_path)
        if sha256_file(shard_path) != shard_info["sha256"]:
            raise ValueError(f"Shard hash mismatch: {shard_info['path']}")
        for local_index in range(int(shard["num_records"])):
            global_index = offset + local_index
            if global_index not in selected_set:
                continue
            records.append(
                {
                    "global_index": global_index,
                    "sample_id": shard["sample_ids"][local_index],
                    "position": int(shard["position"][local_index]),
                    "base_token_ids": shard["base_token_ids"][local_index],
                    "base_logits": shard["base_logits"][local_index],
                    "guide_token_ids": shard["guide_token_ids"][local_index],
                    "guide_logits": shard["guide_logits"][local_index],
                    "gold_token_id": int(shard["gold_token_id"][local_index]),
                    "derived": {
                        name: shard["derived"][name][local_index]
                        for name in shard["derived"]
                    },
                }
            )
        offset += int(shard["num_records"])
    if len(records) != count:
        raise RuntimeError("Failed to locate every selected audit record")
    return records


def group_records_by_extraction_batch(
    records: list[dict[str, Any]],
    source_index_by_id: dict[str, int],
    *,
    batch_size: int,
) -> dict[int, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("Extraction batch size must be positive")
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in source_index_by_id:
            raise ValueError(f"Cached sample is absent from source: {sample_id}")
        source_index = source_index_by_id[sample_id]
        batch_start = (source_index // batch_size) * batch_size
        groups.setdefault(batch_start, []).append(record)
    return groups


def _prefix_descriptor(
    batch: ExtractionBatch,
    *,
    row_index: int,
    continuation_position: int,
) -> dict[str, Any]:
    logit_position = batch.logit_positions[row_index][continuation_position]
    prefix_ids = (
        batch.input_ids[row_index, : logit_position + 1]
        .detach()
        .cpu()
        .tolist()
    )
    encoded = json.dumps(
        prefix_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "prefix_length": len(prefix_ids),
        "logit_position": logit_position,
        "padded_batch_length": int(batch.input_ids.shape[1]),
        "attention_length": int(batch.attention_mask[row_index].sum()),
        "position_id": int(batch.position_ids[row_index, logit_position]),
    }


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def _mismatch_diagnostic(
    *,
    cached: dict[str, Any],
    online_base: CandidateBatch,
    online_guide: CandidateBatch,
    position: int,
    prefix: dict[str, Any],
    precision: str,
    base_checkpoint_sha256: str,
    guide_checkpoint_sha256: str,
) -> dict[str, Any]:
    base_margin = float(online_base.boundary_margin[position])
    guide_margin = float(online_guide.boundary_margin[position])
    return {
        "sample_id": cached["sample_id"],
        "position": position,
        "cached_base_topk_token_ids": cached["base_token_ids"].tolist(),
        "cached_base_topk_logits": cached["base_logits"].float().tolist(),
        "online_base_topk_token_ids": (
            online_base.token_ids[position].tolist()
        ),
        "online_base_topk_logits": (
            online_base.logits[position].float().tolist()
        ),
        "cached_guide_topk_token_ids": cached["guide_token_ids"].tolist(),
        "cached_guide_topk_logits": cached["guide_logits"].float().tolist(),
        "online_guide_topk_token_ids": (
            online_guide.token_ids[position].tolist()
        ),
        "online_guide_topk_logits": (
            online_guide.logits[position].float().tolist()
        ),
        "base_topk_boundary_margin": base_margin,
        "guide_topk_boundary_margin": guide_margin,
        "minimum_topk_boundary_margin": min(base_margin, guide_margin),
        **prefix,
        "precision": precision,
        "base_model_output_dtype": online_base.source_dtype,
        "guide_model_output_dtype": online_guide.source_dtype,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "guide_checkpoint_sha256": guide_checkpoint_sha256,
    }


def main() -> None:
    args = parse_args()
    if args.records_per_split < 20:
        raise ValueError("records-per-split must be at least 20")
    config = SentimentGuideConfig.load_json(args.config)
    validation = _load_json(args.guide_validation)
    if validation.get("status") != "PASS":
        raise ValueError("Sentiment guide validation is not PASS")
    cache_root = require_path_within(
        args.cache_root,
        CACHE_ROOT,
        label="audited cache root",
    )
    output_path = require_path_within(
        args.output,
        REPORT_ROOT,
        label="recomputation audit report",
    )
    manifests = {
        split: _load_json(cache_root / split / "manifest.json")
        for split in args.splits
    }
    manifest_precisions = {
        str(manifest.get("precision")) for manifest in manifests.values()
    }
    if len(manifest_precisions) != 1:
        raise ValueError("All audited splits must use the same precision")
    manifest_precision = manifest_precisions.pop()
    for split, manifest in manifests.items():
        if not manifest.get("complete") or manifest.get("task") != "sentiment":
            raise ValueError(f"Incomplete/non-sentiment manifest for {split}")
        if (
            manifest["nan_inf_count"] != 0
            or manifest["top_k_uniqueness_failures"] != 0
        ):
            raise ValueError(f"Invalid cached values reported for {split}")

    device = resolve_device(
        args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    precision = manifest_precision if args.precision == "auto" else args.precision
    if precision != manifest_precision:
        raise ValueError(
            "Audit precision must exactly match cache manifest precision"
        )
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 cache audit requires CUDA")
    scientific_root = cache_root.name in {"rad_smoke_fp32", "rad"}
    if scientific_root and precision != "fp32":
        raise ValueError("Scientific smoke/persistent cache audit requires FP32")
    if precision == "fp32":
        if not 0 < args.fp32_atol <= 1e-5:
            raise ValueError("FP32 audit tolerance must be in (0, 1e-5]")
        tolerance = args.fp32_atol
    else:
        tolerance = args.fp16_atol
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
    base_checkpoint = checkpoint_descriptor(base_path)
    guide_checkpoint = checkpoint_descriptor(guide_path)
    for split, manifest in manifests.items():
        parameters = manifest.get("extraction_parameters", {})
        if (
            manifest["base_model"]["checkpoint_sha256"]
            != base_checkpoint["checkpoint_sha256"]
            or manifest["guide_model"]["checkpoint_sha256"]
            != guide_checkpoint["checkpoint_sha256"]
        ):
            raise ValueError(f"Model provenance mismatch for {split}")
        if parameters.get("precision") != precision:
            raise ValueError(f"Extraction precision mismatch for {split}")
        if parameters.get("device_type") != device.type:
            raise ValueError(f"Extraction/audit device mismatch for {split}")
        if scientific_root:
            if parameters.get("deterministic_runtime") != deterministic_runtime:
                raise ValueError(
                    f"Deterministic runtime mismatch for {split}"
                )
            if parameters.get("explicit_position_ids") is not True:
                raise ValueError(f"Position IDs were not pinned for {split}")
            if parameters.get("batch_atomic_shards") is not True:
                raise ValueError(
                    f"Shard resume boundary is not batch-atomic for {split}"
                )

    tokenizer_pair = load_compatible_tokenizer_pair(base_path, guide_path)
    tokenizer = tokenizer_pair.base
    base_tokenizer = tokenizer_descriptor(tokenizer, base_path)
    guide_tokenizer_info = tokenizer_descriptor(
        tokenizer_pair.guide,
        guide_path,
    )
    for split, manifest in manifests.items():
        if (
            manifest["base_tokenizer"]["semantic_sha256"]
            != base_tokenizer["semantic_sha256"]
            or manifest["guide_tokenizer"]["semantic_sha256"]
            != guide_tokenizer_info["semantic_sha256"]
        ):
            raise ValueError(f"Tokenizer provenance mismatch for {split}")

    model_pair = load_frozen_causal_model_pair(
        base_path,
        guide_path,
        device=device,
        precision=precision,
    )
    base_model = model_pair.base
    guide_model = model_pair.guide

    split_reports = {}
    all_pass = True
    for split_index, split in enumerate(args.splits):
        manifest = manifests[split]
        parameters = manifest["extraction_parameters"]
        source_path = Path(manifest["source_dataset"]["path"])
        samples = load_rad_samples(
            source_path,
            max_samples=int(manifest["number_source_examples"]),
        )
        source_index_by_id = {
            sample.sample_id: index for index, sample in enumerate(samples)
        }
        if len(source_index_by_id) != len(samples):
            raise ValueError(f"Duplicate source sample IDs in {split}")
        cached_records = _sample_cached_records(
            cache_root / split,
            manifest,
            count=args.records_per_split,
            seed=args.seed + split_index,
        )
        batch_size = int(parameters["batch_size"])
        grouped = group_records_by_extraction_batch(
            cached_records,
            source_index_by_id,
            batch_size=batch_size,
        )
        rows = []
        mismatch_diagnostics = []
        for batch_start in sorted(grouped):
            batch_samples = samples[batch_start : batch_start + batch_size]
            prepared = prepare_extraction_batch(
                tokenizer,
                batch_samples,
                max_length=config.max_length,
                max_continuation_tokens=int(
                    parameters["max_continuation_tokens"]
                ),
            )
            base_candidates = model_topk(
                base_model,
                prepared,
                top_k=int(manifest["top_k"]),
                device=device,
                use_fp16=use_fp16,
            )
            guide_candidates = model_topk(
                guide_model,
                prepared,
                top_k=int(manifest["top_k"]),
                device=device,
                use_fp16=use_fp16,
            )
            for cached in grouped[batch_start]:
                source_index = source_index_by_id[str(cached["sample_id"])]
                row_index = source_index - batch_start
                position = int(cached["position"])
                online_base = base_candidates[row_index]
                online_guide = guide_candidates[row_index]
                online_base_ids = online_base.token_ids[position]
                online_base_logits = online_base.logits[position]
                online_guide_ids = online_guide.token_ids[position]
                online_guide_logits = online_guide.logits[position]
                online_gold = int(
                    prepared.continuation_ids[row_index][position]
                )
                derived = compute_confidence_disagreement(
                    online_base_ids.unsqueeze(0),
                    online_base_logits.unsqueeze(0),
                    online_guide_ids.unsqueeze(0),
                    online_guide_logits.unsqueeze(0),
                    torch.tensor([position]),
                    max_position=int(manifest["max_position"]),
                ).as_dict()
                base_ids_match = torch.equal(
                    online_base_ids,
                    cached["base_token_ids"],
                )
                guide_ids_match = torch.equal(
                    online_guide_ids,
                    cached["guide_token_ids"],
                )
                gold_match = online_gold == cached["gold_token_id"]
                base_logit_error = _maximum_error(
                    online_base_logits,
                    cached["base_logits"],
                )
                guide_logit_error = _maximum_error(
                    online_guide_logits,
                    cached["guide_logits"],
                )
                derived_error = max(
                    _maximum_error(
                        derived[name][0],
                        cached["derived"][name],
                    )
                    for name in derived
                )
                row_pass = (
                    base_ids_match
                    and guide_ids_match
                    and gold_match
                    and base_logit_error <= tolerance
                    and guide_logit_error <= tolerance
                    and derived_error <= tolerance
                )
                all_pass = all_pass and row_pass
                prefix = _prefix_descriptor(
                    prepared,
                    row_index=row_index,
                    continuation_position=position,
                )
                rows.append(
                    {
                        "sample_id": cached["sample_id"],
                        "source_index": source_index,
                        "extraction_batch_start": batch_start,
                        "position": position,
                        "gold_token_id": online_gold,
                        "gold_id_match": gold_match,
                        "base_ids_match": base_ids_match,
                        "guide_ids_match": guide_ids_match,
                        "base_max_abs_logit_error": base_logit_error,
                        "guide_max_abs_logit_error": guide_logit_error,
                        "derived_max_abs_error": derived_error,
                        "minimum_topk_boundary_margin": min(
                            float(online_base.boundary_margin[position]),
                            float(online_guide.boundary_margin[position]),
                        ),
                        "prefix_token_ids_sha256": (
                            prefix["prefix_token_ids_sha256"]
                        ),
                        "pass": row_pass,
                    }
                )
                if not row_pass:
                    mismatch_diagnostics.append(
                        _mismatch_diagnostic(
                            cached=cached,
                            online_base=online_base,
                            online_guide=online_guide,
                            position=position,
                            prefix=prefix,
                            precision=precision,
                            base_checkpoint_sha256=(
                                base_checkpoint["checkpoint_sha256"]
                            ),
                            guide_checkpoint_sha256=(
                                guide_checkpoint["checkpoint_sha256"]
                            ),
                        )
                    )
        split_reports[split] = {
            "manifest_path": str(cache_root / split / "manifest.json"),
            "manifest_sha256": sha256_file(
                cache_root / split / "manifest.json"
            ),
            "number_source_examples": manifest["number_source_examples"],
            "number_token_records": manifest["number_token_records"],
            "number_shards": manifest["number_shards"],
            "records_audited": len(rows),
            "source_batches_recomputed": len(grouped),
            "base_guide_topk_different_records": sum(
                not torch.equal(
                    cached["base_token_ids"],
                    cached["guide_token_ids"],
                )
                for cached in cached_records
            ),
            "mismatch_count": len(mismatch_diagnostics),
            "mismatch_diagnostics": mismatch_diagnostics,
            "rows": sorted(
                rows,
                key=lambda row: (
                    int(row["source_index"]),
                    int(row["position"]),
                ),
            ),
            "pass": all(row["pass"] for row in rows),
        }

    report = {
        "schema_version": 2,
        "status": "PASS" if all_pass else "FAIL",
        "task": "sentiment",
        "cache_root": str(cache_root),
        "record_sampling_seed": args.seed,
        "inference_seed": config.seed,
        "device": str(device),
        "precision": precision,
        "absolute_tolerance": tolerance,
        "deterministic_runtime": deterministic_runtime,
        "base_model": base_checkpoint,
        "guide_model": guide_checkpoint,
        "base_tokenizer": base_tokenizer,
        "guide_tokenizer": guide_tokenizer_info,
        "tokenizer_compatibility_checks": (
            tokenizer_pair.compatibility_checks
        ),
        "checks": {
            "independent_base_topk_recomputed": True,
            "independent_guide_topk_recomputed": True,
            "topk_selector_received_no_gold": True,
            "gold_used_as_target_only": True,
            "tokenizers_exactly_compatible": True,
            "shard_hashes_verified": True,
            "same_model_loader": True,
            "same_tokenizer_loader": True,
            "same_batch_size_and_padding_context": True,
            "batch_atomic_shard_resume": True,
            "same_attention_mask_construction": True,
            "explicit_position_ids_match": True,
            "models_eval_and_frozen": True,
            "full_logits_transient": True,
            "model_outputs_fp32": precision == "fp32",
            "online_values_match_cache": all_pass,
        },
        "splits": split_reports,
    }
    write_json_atomic(
        output_path,
        report,
        overwrite=args.overwrite_report,
    )
    print(json.dumps({"status": report["status"], "report": str(output_path)}))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
