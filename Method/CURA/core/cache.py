from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path
import sys

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.audit_signals import ARTIFACTS, audit
from Method.CURA.core.cache_io import (
    SCHEMA_VERSION, atomic_json, atomic_shard, fingerprint, read_manifest, sha256_file, verify_cache,
)
from Method.CURA.core.config import PROJECT_ROOT, load_config, objective_mismatches
from Method.CURA.core.features import base_routing_features
from Method.CURA.router.adapters.registry import load_adapters
from Method.MultiSignal.core.cache import load_base_adapter


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text(value):
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def config_fingerprint(config, args):
    artifacts = {}
    for signal in config["signals"]:
        path = PROJECT_ROOT / ARTIFACTS[signal]
        if path.is_dir():
            files = sorted(
                (str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
                for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts
            )
            artifacts[signal] = {"path": str(path.relative_to(PROJECT_ROOT)), "files": files}
        else:
            artifacts[signal] = {"path": str(path.relative_to(PROJECT_ROOT)),
                                 "size": path.stat().st_size if path.exists() else None,
                                 "mtime_ns": path.stat().st_mtime_ns if path.exists() else None}
    return fingerprint({
        "base_model": config["base_model"], "signals": config["signals"],
        "signal_objectives": config["signal_objectives"], "score_directions": config["score_directions"],
        "objective": config["objective"],
        "top_k": args.top_k, "max_response_tokens": args.max_response_tokens,
        "max_samples": args.max_samples, "shard_size": args.shard_size,
        "dtype": config["cache"]["dtype"], "paper_mode": config.get("paper_mode", False),
        "tokenization_mismatch_threshold": config.get("tokenization_mismatch_threshold"),
        "artifacts": artifacts,
    })


@torch.inference_mode()
def build_prompt_rows(sample, index, base, adapters, args, config):
    prompt = text(sample.get("prompt"))
    response = text(sample.get("response", sample.get("continuation")))
    prefix_ids = base.encode_prompt(prompt)
    response_ids = base.encode_response(response).flatten()[:args.max_response_tokens]
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    if not prompt.strip() or not response_ids.numel():
        raise ValueError("empty prompt or response")
    rows = []
    dtype = torch.float16 if config["cache"]["dtype"] == "float16" else torch.float32
    prompt_id = str(sample.get("md5_hash") or sample.get("id", index))
    for step, gold_token in enumerate(response_ids):
        full_logits = base.next_logits(prefix_ids)[0].float()
        top_logits, candidate_ids = torch.topk(full_logits, args.top_k)
        routing_features = base_routing_features(
            top_logits.unsqueeze(0), torch.tensor([step], device=top_logits.device),
            torch.tensor([prefix_ids.size(1)], device=top_logits.device), full_logits.unsqueeze(0)
        )[0]
        gold_id = int(gold_token)
        matches = (candidate_ids == gold_id).nonzero(as_tuple=False)
        gold_index = int(matches[0]) if len(matches) else args.top_k - 1
        if not len(matches):
            candidate_ids[-1], top_logits[-1] = gold_id, full_logits[gold_id]
        candidate_text = base.tokenizer.batch_decode(candidate_ids)
        prefix_text = base.tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        outputs = [
            adapter.score(
                prefix_ids.to(adapter.device), candidate_ids.to(adapter.device), candidate_text, prefix_text
            )
            for adapter in adapters
        ]
        if config.get("paper_mode"):
            fallback = [output.name for output in outputs if "fallback" in output.source]
            if fallback:
                raise RuntimeError(f"Paper cache forbids fallback signals: {fallback}")
            excessive_mismatch = [
                output.name for output in outputs
                if output.metadata.get("tokenization_mismatch_rate", 0.0)
                > config.get("tokenization_mismatch_threshold", 0.05)
            ]
            if excessive_mismatch:
                raise RuntimeError(f"Tokenizer mismatch exceeds threshold: {excessive_mismatch}")
        raw_scores = torch.stack([output.scores[0].cpu() for output in outputs], dim=0)
        row = {
            "prompt_id": prompt_id, "step": step,
            "prefix_token_ids": prefix_ids[0].detach().cpu(),
            "candidate_token_ids": candidate_ids.cpu(),
            "candidate_text": candidate_text,
            "base_logits": top_logits.cpu().to(dtype),
            "base_probs": torch.softmax(top_logits.cpu().float(), -1).to(dtype),
            "base_routing_features": routing_features.detach().cpu().to(dtype),
            "raw_scores": raw_scores.to(dtype),
            "signal_mask": torch.tensor([output.available for output in outputs]),
            "signal_sources": [output.source for output in outputs],
            "signal_objectives": [output.objective for output in outputs],
            "cost_ms": torch.tensor([output.cost_ms for output in outputs]),
            "tokenization_mismatch": [output.metadata.get("tokenization_mismatch", False) for output in outputs],
            "tokenization_mismatch_rate": [output.metadata.get("tokenization_mismatch_rate", 0.0) for output in outputs],
            "candidate_mapping": [output.metadata.get("candidate_mapping") for output in outputs],
            "gold_index": gold_index, "target_utilities": None, "preference_alpha": None,
            "position": step, "prefix_length": prefix_ids.size(1),
        }
        if config["cache"].get("store_text"):
            row["prefix"] = base.tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        rows.append(row)
        prefix_ids = torch.cat([prefix_ids, torch.tensor([[gold_id]], device=prefix_ids.device)], dim=1)
    return rows


def initial_manifest(args, config, dataset_path, dataset_sha, cfg_fingerprint, total):
    return {
        "schema_version": SCHEMA_VERSION, "method": "cura", "split": args.split,
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_fingerprint": "sha256:" + dataset_sha,
        "config_fingerprint": cfg_fingerprint, "signals": config["signals"],
        "objective": config["objective"], "base_model": config["base_model"],
        "paper_mode": config.get("paper_mode", False), "score_directions": config["score_directions"],
        "tokenization_mismatch_threshold": config.get("tokenization_mismatch_threshold", 0.05),
        "top_k": args.top_k,
        "max_response_tokens": args.max_response_tokens, "shard_size": args.shard_size,
        "total_prompts": total, "completed_prompts": 0, "completed_shards": [],
        "failed_prompt_ids": [], "status": "running",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Method/CURA/configs/sentiment.json")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--input")
    parser.add_argument("--cache-dir")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-response-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-objective-mismatch", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    args.shard_size = args.shard_size or int(config["cache"]["shard_size"])
    dataset_path = Path(args.input) if args.input else PROJECT_ROOT / f"dataset/multisignal_train/{args.split}.jsonl"
    if not dataset_path.is_absolute(): dataset_path = PROJECT_ROOT / dataset_path
    cache_dir = Path(args.cache_dir) if args.cache_dir else PROJECT_ROOT / f"dataset/cura_cache/{args.split}"
    if not cache_dir.is_absolute(): cache_dir = PROJECT_ROOT / cache_dir
    if args.shard_size <= 0 or args.top_k < 2: raise ValueError("shard-size must be positive and top-k >= 2")
    report = audit(config)
    missing = [error for error in report["errors"] if "missing checkpoint" in error]
    if missing: raise RuntimeError("; ".join(missing))
    if objective_mismatches(config) and not args.allow_objective_mismatch:
        raise RuntimeError("Signal objectives do not match. Use a compatible config; --allow-objective-mismatch is MVP-only.")
    data = load_jsonl(dataset_path)
    if args.max_samples is not None: data = data[:args.max_samples]
    dataset_sha, cfg_fp = sha256_file(dataset_path), config_fingerprint(config, args)
    manifest = read_manifest(cache_dir)
    if manifest:
        if not args.resume: raise FileExistsError(f"Cache exists; pass --resume: {cache_dir}")
        if manifest["dataset_fingerprint"] != "sha256:" + dataset_sha or manifest["config_fingerprint"] != cfg_fp:
            raise RuntimeError("Cache fingerprint mismatch; choose a new cache directory")
        verify_cache(cache_dir)
    else:
        manifest = initial_manifest(args, config, dataset_path, dataset_sha, cfg_fp, len(data))
        atomic_json(manifest, cache_dir / "manifest.json")
    if args.retry_errors and (cache_dir / "errors.jsonl").exists():
        failed_indexes = set()
        active_failed_ids = set(manifest["failed_prompt_ids"])
        with (cache_dir / "errors.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    error_row = json.loads(line)
                    if str(error_row["prompt_id"]) in active_failed_ids:
                        failed_indexes.add(int(error_row["index"]))
        manifest["completed_shards"] = [
            shard for shard in manifest["completed_shards"]
            if not any(shard["start_index"] <= index <= shard["end_index"] for index in failed_indexes)
        ]
        manifest["completed_prompts"] = sum(x["num_prompts"] for x in manifest["completed_shards"])
        manifest["status"] = "running"
        atomic_json(manifest, cache_dir / "manifest.json")
    completed = {i for shard in manifest["completed_shards"] for i in range(shard["start_index"], shard["end_index"] + 1)}
    device = torch.device("cuda" if args.base_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    signal_device = torch.device("cuda" if args.signal_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.base_device == "cuda" and device.type != "cuda" or args.signal_device == "cuda" and signal_device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    base, adapters = load_base_adapter(device), load_adapters(config, PROJECT_ROOT, signal_device)
    try:
        for start in range(0, len(data), args.shard_size):
            end = min(start + args.shard_size, len(data))
            if all(index in completed for index in range(start, end)): continue
            rows, failed = [], []
            for index in tqdm(range(start, end), desc=f"CURA shard {start // args.shard_size}"):
                last_error = None
                last_trace = None
                for _ in range(int(config["cache"].get("max_retries", 2)) + 1):
                    try:
                        rows.extend(build_prompt_rows(data[index], index, base, adapters, args, config)); last_error = None; break
                    except Exception as error:
                        last_error, last_trace = error, traceback.format_exc()
                if last_error:
                    prompt_id = str(data[index].get("md5_hash") or index)
                    failed.append(prompt_id)
                    with (cache_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"prompt_id": prompt_id, "index": index, "error": f"{type(last_error).__name__}: {last_error}", "traceback": last_trace}) + "\n")
                else:
                    prompt_id = str(data[index].get("md5_hash") or index)
                    manifest["failed_prompt_ids"] = [item for item in manifest["failed_prompt_ids"] if item != prompt_id]
            shard_id = start // args.shard_size
            relative = f"shards/shard_{shard_id:06d}.pt"
            payload = {"schema_version": SCHEMA_VERSION, "start_index": start, "end_index": end - 1, "rows": rows}
            checksum = atomic_shard(payload, cache_dir / relative)
            descriptor = {"id": shard_id, "file": relative, "start_index": start, "end_index": end - 1,
                          "num_prompts": end - start, "num_steps": len(rows), "sha256": checksum}
            manifest["completed_shards"] = [x for x in manifest["completed_shards"] if x["id"] != shard_id] + [descriptor]
            manifest["completed_shards"].sort(key=lambda x: x["id"])
            manifest["failed_prompt_ids"] = sorted(set(manifest["failed_prompt_ids"] + failed))
            manifest["completed_prompts"] = sum(x["num_prompts"] for x in manifest["completed_shards"])
            manifest["status"] = "complete" if manifest["completed_prompts"] == len(data) else "running"
            atomic_json(manifest, cache_dir / "manifest.json")
    finally:
        del adapters, base
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps({"cache_dir": str(cache_dir), "completed_prompts": manifest["completed_prompts"],
                      "total_prompts": len(data), "failed": len(manifest["failed_prompt_ids"]), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
