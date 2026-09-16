from __future__ import annotations

import argparse
import json
import sys
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
    for candidate_index, candidate in enumerate(candidates):
        rollout_texts = []
        initial = torch.cat([prefix, candidate.reshape(1)]).reshape(1, -1)
        for rollout in range(args.rollouts):
            rollout_seed = args.seed + 1000003 * row["step"] + 97 * candidate_index + rollout
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rollout_seed)
            generated = base_model.generate(
                initial, max_new_tokens=args.rollout_tokens, do_sample=args.rollouts > 1,
                top_p=args.top_p, temperature=args.temperature,
                pad_token_id=base_tokenizer.eos_token_id,
            )
            rollout_texts.append(base_tokenizer.decode(generated[0], skip_special_tokens=True))
        utilities[candidate_index] = evaluator_scores(
            rollout_texts, evaluator_tokenizer, evaluator, device, args.positive_label
        ).mean()
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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-signal-evaluator", action="store_true")
    args = parser.parse_args()
    if args.rollouts < 1 or args.rollout_tokens < 1:
        raise ValueError("rollouts and rollout-tokens must be positive")
    cache_dir, evaluator_path = resolve(args.cache_dir), resolve(args.evaluator)
    manifest = read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"Missing cache manifest: {cache_dir}")
    verify_cache(cache_dir)
    signal_paths = [resolve(ARTIFACTS[name]) for name in manifest["signals"]]
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
        "evaluator": str(evaluator_path), "rollouts": args.rollouts, "rollout_tokens": args.rollout_tokens,
        "positive_label": args.positive_label, "top_p": args.top_p, "temperature": args.temperature,
        "seed": args.seed,
    })
    previous_target = manifest.get("target_utility")
    if previous_target and previous_target.get("fingerprint") != target_fingerprint:
        raise RuntimeError("Target annotation fingerprint differs; use a fresh cache or identical evaluator settings")
    manifest["target_utility"] = {
        "status": "running", "fingerprint": target_fingerprint, "evaluator": str(evaluator_path),
        "rollouts": args.rollouts, "rollout_tokens": args.rollout_tokens, "seed": args.seed,
    }
    atomic_json(manifest, cache_dir / "manifest.json")
    for descriptor in manifest["completed_shards"]:
        shard_path = cache_dir / descriptor.get("target_file", descriptor["file"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        if all(row.get("target_utilities") is not None for row in payload["rows"]):
            continue
        for row in payload["rows"]:
            if row.get("target_utilities") is None:
                annotate_row(row, base_model, base_tokenizer, evaluator, evaluator_tokenizer, args, device)
        target_relative = f"target_shards/shard_{descriptor['id']:06d}.pt"
        descriptor["target_file"] = target_relative
        descriptor["target_sha256"] = atomic_shard(payload, cache_dir / target_relative)
        atomic_json(manifest, cache_dir / "manifest.json")
    manifest["target_utility"]["status"] = "complete"
    atomic_json(manifest, cache_dir / "manifest.json")
    print(json.dumps(manifest["target_utility"], indent=2))


if __name__ == "__main__":
    main()
