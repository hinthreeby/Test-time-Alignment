from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.audit_signals import ARTIFACTS
from Method.CURA.core.cache_io import atomic_json, atomic_shard, fingerprint, read_manifest, verify_cache
from Method.CURA.core.config import PROJECT_ROOT


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluator_scores(texts, tokenizer, model, device, positive_label):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = model(**encoded).logits.float()
    if logits.size(-1) == 1:
        return logits[:, 0]
    return torch.softmax(logits, -1)[:, positive_label]


@torch.inference_mode()
def annotate_row(row, base_model, base_tokenizer, evaluator, evaluator_tokenizer, args, device):
    prefix = row["prefix_token_ids"].to(device)
    candidates = row["candidate_token_ids"].to(device)
    utilities = torch.zeros(candidates.numel(), device=device)
    # Cache prefixes contain prompt + ``step`` teacher-forced response tokens.
    # The benchmark sentiment metric scores responses only, so rollout targets
    # must exclude prompt tokens while retaining response-so-far.
    response_start = max(0, prefix.numel() - int(row["step"]))
    prompt_key = str(row.get("prompt_id", ""))
    prompt_seed = int.from_bytes(hashlib.sha256(prompt_key.encode("utf-8")).digest()[:4], "big")
    for start in range(0, candidates.numel(), args.candidate_batch_size):
        stop = min(start + args.candidate_batch_size, candidates.numel())
        candidate_batch = candidates[start:stop]
        prefixes = prefix.unsqueeze(0).expand(candidate_batch.numel(), -1)
        initial = torch.cat([prefixes, candidate_batch.unsqueeze(1)], dim=1)
        initial = initial.repeat_interleave(args.rollouts, dim=0)
        rollout_seed = args.seed + prompt_seed + 1000003 * int(row["step"]) + 97 * start
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        generated = base_model.generate(
            initial, max_new_tokens=args.rollout_tokens, do_sample=args.rollouts > 1,
            top_p=args.top_p, temperature=args.temperature,
            pad_token_id=base_tokenizer.eos_token_id,
        )
        rollout_texts = base_tokenizer.batch_decode(
            generated[:, response_start:], skip_special_tokens=True
        )
        scores = evaluator_scores(
            rollout_texts, evaluator_tokenizer, evaluator, device, args.positive_label
        ).reshape(candidate_batch.numel(), args.rollouts)
        utilities[start:stop] = scores.mean(dim=1)
    row["target_utilities"] = utilities.cpu().to(torch.float16)
    return row


def main():
    parser = argparse.ArgumentParser(description="Add held-out rollout utilities to CURA cache shards.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--rollout-tokens", type=int, default=16)
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--candidate-batch-size", type=int, default=10,
        help="Candidates generated together; effective rollout batch is this value times --rollouts.",
    )
    parser.add_argument(
        "--save-every-rows", type=int, default=100,
        help="Atomically checkpoint each target shard after this many newly annotated token rows.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-signal-evaluator", action="store_true")
    args = parser.parse_args()
    if min(args.rollouts, args.rollout_tokens, args.candidate_batch_size, args.save_every_rows) < 1:
        raise ValueError("rollouts, rollout-tokens, candidate-batch-size and save-every-rows must be positive")
    cache_dir, evaluator_path = resolve(args.cache_dir), resolve(args.evaluator)
    manifest = read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"Missing cache manifest: {cache_dir}")
    verify_cache(cache_dir)
    manifest_artifacts = manifest.get("signal_artifacts", {})
    signal_paths = [
        resolve(manifest_artifacts.get(name, ARTIFACTS[name]))
        for name in manifest["signals"]
    ]
    if not args.allow_signal_evaluator and any(
        evaluator_path == path or evaluator_path in path.parents or path in evaluator_path.parents for path in signal_paths
    ):
        raise RuntimeError("Held-out evaluator overlaps a CURA signal; choose an independent evaluator")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    base_path = PROJECT_ROOT / "models" / manifest.get("base_model", "gpt2-large")
    base_tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    # Decoder-only models require left-padding for correct batched generation.
    base_tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_path, local_files_only=True)
    evaluator = AutoModelForSequenceClassification.from_pretrained(
        evaluator_path, local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    target_fingerprint = fingerprint({
        "annotation_version": 3,
        "evaluator": str(evaluator_path), "rollouts": args.rollouts, "rollout_tokens": args.rollout_tokens,
        "positive_label": args.positive_label, "top_p": args.top_p, "temperature": args.temperature,
        "seed": args.seed, "candidate_batch_size": args.candidate_batch_size,
    })
    previous_target = manifest.get("target_utility")
    if previous_target and previous_target.get("fingerprint") != target_fingerprint:
        raise RuntimeError("Target annotation fingerprint differs; use a fresh cache or identical evaluator settings")
    manifest["target_utility"] = {
        "status": "running", "fingerprint": target_fingerprint, "evaluator": str(evaluator_path),
        "rollouts": args.rollouts, "rollout_tokens": args.rollout_tokens, "seed": args.seed,
        "annotation_version": 3, "candidate_batch_size": args.candidate_batch_size,
    }
    atomic_json(manifest, cache_dir / "manifest.json")

    # ── Progress bookkeeping ───────────────────────────────────────────────
    total_shards = len(manifest["completed_shards"])
    total_rows = sum(
        len(torch.load(cache_dir / d.get("target_file", d["file"]),
                       map_location="cpu", weights_only=False)["rows"])
        for d in manifest["completed_shards"]
    )
    done_rows = 0
    skipped_shards = 0
    t_start = time.time()
    print(f"[annotate] {total_shards} shards | {total_rows} token-rows | "
          f"rollouts={args.rollouts} rollout_tokens={args.rollout_tokens} "
          f"candidate_batch={args.candidate_batch_size}",
          flush=True)
    # ──────────────────────────────────────────────────────────────────────

    for shard_idx, descriptor in enumerate(manifest["completed_shards"], 1):
        shard_path = cache_dir / descriptor.get("target_file", descriptor["file"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_rows = len(payload["rows"])
        if all(row.get("target_utilities") is not None for row in payload["rows"]):
            done_rows += shard_rows
            skipped_shards += 1
            print(f"[annotate] shard {shard_idx:>3}/{total_shards}  SKIP (already annotated) "
                  f"| rows done {done_rows}/{total_rows}", flush=True)
            continue
        target_relative = f"target_shards/shard_{descriptor['id']:06d}.pt"
        newly_annotated = 0
        print(f"[annotate] shard {shard_idx:>3}/{total_shards}  START ({shard_rows} rows)", flush=True)
        for row_idx, row in enumerate(payload["rows"], 1):
            if row.get("target_utilities") is None:
                annotate_row(row, base_model, base_tokenizer, evaluator, evaluator_tokenizer, args, device)
                newly_annotated += 1
            done_rows += 1
            elapsed = time.time() - t_start
            rate = done_rows / elapsed if elapsed > 0 else 0.0
            remaining = (total_rows - done_rows) / rate if rate > 0 else float("inf")
            eta_str = f"{remaining/60:.1f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"
            print(
                f"\r[annotate] shard {shard_idx:>3}/{total_shards}  "
                f"row {row_idx:>4}/{shard_rows}  "
                f"total {done_rows}/{total_rows}  "
                f"{rate:.2f} rows/s  ETA {eta_str}   ",
                end="", flush=True,
            )
            if newly_annotated > 0 and newly_annotated % args.save_every_rows == 0:
                descriptor["target_file"] = target_relative
                descriptor["target_sha256"] = atomic_shard(payload, cache_dir / target_relative)
                atomic_json(manifest, cache_dir / "manifest.json")
        print(flush=True)  # newline after \r progress
        descriptor["target_file"] = target_relative
        descriptor["target_sha256"] = atomic_shard(payload, cache_dir / target_relative)
        atomic_json(manifest, cache_dir / "manifest.json")
        print(f"[annotate] shard {shard_idx:>3}/{total_shards}  DONE  "
              f"newly_annotated={newly_annotated}  "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

    elapsed_total = time.time() - t_start
    print(f"[annotate] COMPLETE  total_rows={done_rows}  skipped_shards={skipped_shards}  "
          f"elapsed={elapsed_total:.0f}s ({elapsed_total/60:.1f}m)", flush=True)
    manifest["target_utility"]["status"] = "complete"
    atomic_json(manifest, cache_dir / "manifest.json")
    print(json.dumps(manifest["target_utility"], indent=2))


if __name__ == "__main__":
    main()
