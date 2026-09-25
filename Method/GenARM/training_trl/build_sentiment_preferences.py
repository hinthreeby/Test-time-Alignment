#!/usr/bin/env python3
"""Build local pseudo-labeled preferences for sentiment GenARM training.

The pipeline is intentionally split into two resumable stages:

1. Generate multiple continuations from a frozen local causal LM.
2. Rank response-only continuations with a frozen local sentiment teacher.

The final JSONL contains DPO/ARM-compatible ``prompt``, ``chosen`` and
``rejected`` fields. Final benchmark prompts are excluded by default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GENERATOR = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_TEACHER = PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
DEFAULT_FINAL_EVALUATOR = PROJECT_ROOT / "models" / "distilbert-sst2"
DEFAULT_FINAL_TEST = PROJECT_ROOT / "dataset" / "rad_benchmark" / "all.jsonl"
DEFAULT_SOURCE = {
    "train": PROJECT_ROOT / "dataset" / "cura_astar_source" / "train.jsonl",
    "validation": PROJECT_ROOT / "dataset" / "cura_astar_source" / "validation.jsonl",
}


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if isinstance(row, dict):
                rows.append(row)
    return rows


def text(value) -> str:
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def normalized_prompt(value: str) -> str:
    return " ".join(str(value).strip().split())


def prompt_digest(value: str) -> str:
    return hashlib.sha256(normalized_prompt(value).encode("utf-8")).hexdigest()


def row_key(row: dict) -> str:
    if row.get("source_id") is not None:
        return str(row["source_id"])
    if row.get("id") is not None:
        return str(row["id"])
    if row.get("md5_hash"):
        return str(row["md5_hash"])
    return prompt_digest(text(row.get("prompt")))


def load_source(path: Path, max_prompts: int | None, forbidden: set[str]) -> tuple[list[dict], int]:
    selected = []
    seen = set()
    excluded = 0
    for index, row in enumerate(load_jsonl(path)):
        prompt = normalized_prompt(text(row.get("prompt")))
        if not prompt:
            continue
        digest = prompt_digest(prompt)
        if digest in forbidden:
            excluded += 1
            continue
        if digest in seen:
            continue
        seen.add(digest)
        selected.append({
            "source_id": row.get("id", row.get("md5_hash", f"row-{index}")),
            "source_index": index,
            "prompt": prompt,
            "prompt_sha256": digest,
            "source_label": row.get("label"),
            "source": row.get("source"),
        })
        if max_prompts is not None and len(selected) >= max_prompts:
            break
    return selected, excluded


def local_model_provenance(path: Path) -> dict:
    weight_files = sorted(
        file for pattern in ("*.safetensors", "*.bin") for file in path.glob(pattern)
    )
    config_path = path / "config.json"
    return {
        "path": str(path),
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "weights": [
            {"name": file.name, "bytes": file.stat().st_size}
            for file in weight_files
        ],
    }


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def generation_fingerprint(args) -> str:
    return fingerprint({
        "generator": local_model_provenance(args.generator_model),
        "num_candidates": args.num_candidates,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    })


def labeling_fingerprint(args) -> str:
    return fingerprint({
        "teacher": local_model_provenance(args.teacher_model),
        "classifier_max_length": args.classifier_max_length,
        "min_chosen_score": args.min_chosen_score,
        "max_rejected_score": args.max_rejected_score,
        "min_score_margin": args.min_score_margin,
        "max_length_ratio": args.max_length_ratio,
        "teacher_input": "response_only",
    })


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def release(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def per_prompt_seed(base_seed: int, digest: str) -> int:
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


@torch.inference_mode()
def generate_candidates_for_prompt(model, tokenizer, prompt: str, digest: str, args, device) -> list[dict]:
    seed = per_prompt_seed(args.seed, digest)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_limit = int(
        getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024))
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, model_limit - args.max_new_tokens),
    ).to(device)
    output = model.generate(
        **encoded,
        do_sample=True,
        num_return_sequences=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_length = encoded["input_ids"].shape[1]
    candidates = []
    seen = set()
    for sequence in output:
        token_ids = sequence[prompt_length:]
        effective_count = int(token_ids.numel())
        if tokenizer.eos_token_id is not None:
            eos_positions = (token_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel():
                effective_count = int(eos_positions[0].item()) + 1
        effective_ids = token_ids[:effective_count]
        response = tokenizer.decode(effective_ids, skip_special_tokens=True).strip()
        normalized = " ".join(response.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append({
            "response": response,
            "generated_tokens": effective_count,
        })
    return candidates


def generate_stage(source_rows: list[dict], candidate_path: Path, args) -> list[dict]:
    partial_path = candidate_path.with_suffix(candidate_path.suffix + ".partial")
    existing = {
        row_key(row): row
        for path in (candidate_path, partial_path)
        for row in load_jsonl(path)
    }
    expected_fingerprint = generation_fingerprint(args)
    incompatible = [
        row_key(row) for row in existing.values()
        if row.get("generation_fingerprint") != expected_fingerprint
    ]
    if incompatible:
        raise RuntimeError(
            "Existing candidate rows use different generation settings. "
            f"Use a fresh --output-dir. First incompatible key: {incompatible[0]}"
        )
    pending = [row for row in source_rows if row_key(row) not in existing]
    print(
        f"Candidate generation: target={len(source_rows)} "
        f"complete={len(source_rows) - len(pending)} pending={len(pending)}"
    )
    if pending:
        device = resolve_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.generator_model,
            local_files_only=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()
        try:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("a", encoding="utf-8") as handle:
                for row in tqdm(pending, desc="Generating sentiment candidates"):
                    try:
                        candidates = generate_candidates_for_prompt(
                            model, tokenizer, row["prompt"], row["prompt_sha256"], args, device
                        )
                        status = "success" if len(candidates) >= 2 else "insufficient_candidates"
                        error = None
                    except Exception as exception:
                        candidates = []
                        status = "failed"
                        error = f"{type(exception).__name__}: {exception}"
                    output = {
                        **row,
                        "seed": per_prompt_seed(args.seed, row["prompt_sha256"]),
                        "generation_fingerprint": expected_fingerprint,
                        "candidates": candidates,
                        "status": status,
                        "error": error,
                    }
                    existing[row_key(row)] = output
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
        finally:
            del model, tokenizer
            release()

    ordered = [existing[row_key(row)] for row in source_rows if row_key(row) in existing]
    atomic_jsonl(ordered, candidate_path)
    partial_path.unlink(missing_ok=True)
    return ordered


def infer_positive_label(model) -> int:
    labels = {
        int(index): str(label).upper()
        for index, label in model.config.id2label.items()
    }
    for index, label in labels.items():
        if "POS" in label:
            return index
    if len(labels) == 2:
        return 1
    raise ValueError(f"Cannot infer positive class from {labels}")


@torch.inference_mode()
def classifier_scores(texts: list[str], tokenizer, model, positive_label: int, args, device) -> list[float]:
    scores = []
    for start in range(0, len(texts), args.classifier_batch_size):
        encoded = tokenizer(
            texts[start:start + args.classifier_batch_size],
            padding=True,
            truncation=True,
            max_length=args.classifier_max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits.float()
        if logits.shape[-1] == 1:
            values = torch.sigmoid(logits[:, 0])
        else:
            values = torch.softmax(logits, dim=-1)[:, positive_label]
        scores.extend(float(value) for value in values.cpu())
    return scores


def select_preference(candidates: list[dict], args) -> tuple[dict | None, str | None]:
    possible = []
    for chosen_index, chosen in enumerate(candidates):
        for rejected_index, rejected in enumerate(candidates):
            if chosen_index == rejected_index:
                continue
            chosen_score = float(chosen["positive_score"])
            rejected_score = float(rejected["positive_score"])
            margin = chosen_score - rejected_score
            if chosen_score < args.min_chosen_score:
                continue
            if rejected_score > args.max_rejected_score:
                continue
            if margin < args.min_score_margin:
                continue
            shorter = max(1, min(chosen["generated_tokens"], rejected["generated_tokens"]))
            longer = max(chosen["generated_tokens"], rejected["generated_tokens"])
            if longer / shorter > args.max_length_ratio:
                continue
            possible.append((margin, chosen_score, -rejected_score, chosen_index, rejected_index))
    if not possible:
        return None, "no_pair_passed_thresholds"
    _, _, _, chosen_index, rejected_index = max(possible)
    chosen = candidates[chosen_index]
    rejected = candidates[rejected_index]
    return {
        "chosen": chosen["response"],
        "rejected": rejected["response"],
        "chosen_score": chosen["positive_score"],
        "rejected_score": rejected["positive_score"],
        "score_margin": chosen["positive_score"] - rejected["positive_score"],
        "chosen_tokens": chosen["generated_tokens"],
        "rejected_tokens": rejected["generated_tokens"],
    }, None


def label_stage(candidate_rows: list[dict], labeled_path: Path, preference_path: Path, args) -> list[dict]:
    partial_path = labeled_path.with_suffix(labeled_path.suffix + ".partial")
    existing = {
        row_key(row): row
        for path in (labeled_path, partial_path)
        for row in load_jsonl(path)
    }
    expected_fingerprint = labeling_fingerprint(args)
    incompatible = [
        row_key(row) for row in existing.values()
        if row.get("labeling_fingerprint") != expected_fingerprint
    ]
    if incompatible:
        raise RuntimeError(
            "Existing labeled rows use a different teacher or filtering settings. "
            f"Use a fresh --output-dir. First incompatible key: {incompatible[0]}"
        )
    eligible = [row for row in candidate_rows if row.get("status") == "success"]
    pending = [row for row in eligible if row_key(row) not in existing]
    print(
        f"Teacher labeling: eligible={len(eligible)} "
        f"complete={len(eligible) - len(pending)} pending={len(pending)}"
    )
    if pending:
        device = resolve_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_model,
            local_files_only=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()
        positive_label = infer_positive_label(model)
        try:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("a", encoding="utf-8") as handle:
                for row in tqdm(pending, desc="Labeling sentiment candidates"):
                    candidates = [dict(candidate) for candidate in row["candidates"]]
                    try:
                        scores = classifier_scores(
                            [candidate["response"] for candidate in candidates],
                            tokenizer,
                            model,
                            positive_label,
                            args,
                            device,
                        )
                        for candidate, score in zip(candidates, scores):
                            candidate["positive_score"] = score
                        preference, reason = select_preference(candidates, args)
                        status = "success" if preference is not None else "skipped"
                        error = reason
                    except Exception as exception:
                        preference = None
                        status = "failed"
                        error = f"{type(exception).__name__}: {exception}"
                    output = {
                        **{key: value for key, value in row.items() if key != "candidates"},
                        "labeling_fingerprint": expected_fingerprint,
                        "candidates": candidates,
                        "preference": preference,
                        "status": status,
                        "error": error,
                    }
                    existing[row_key(row)] = output
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
        finally:
            del model, tokenizer
            release()

    ordered = [existing[row_key(row)] for row in eligible if row_key(row) in existing]
    atomic_jsonl(ordered, labeled_path)
    partial_path.unlink(missing_ok=True)

    preferences = []
    for row in ordered:
        if row.get("status") != "success" or not row.get("preference"):
            continue
        preference = row["preference"]
        preferences.append({
            "prompt": row["prompt"],
            "chosen": preference["chosen"],
            "rejected": preference["rejected"],
            "chosen_score": preference["chosen_score"],
            "rejected_score": preference["rejected_score"],
            "score_margin": preference["score_margin"],
            "chosen_tokens": preference["chosen_tokens"],
            "rejected_tokens": preference["rejected_tokens"],
            "source_id": row["source_id"],
            "prompt_sha256": row["prompt_sha256"],
            "source_label": row.get("source_label"),
            "generator_seed": row.get("seed"),
            "teacher_model": str(args.teacher_model),
        })
    atomic_jsonl(preferences, preference_path)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=["all", "generate", "label"], default="all")
    parser.add_argument("--generator-model", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--final-evaluator-model", type=Path, default=DEFAULT_FINAL_EVALUATOR)
    parser.add_argument("--final-test", type=Path, default=DEFAULT_FINAL_TEST)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--classifier-max-length", type=int, default=512)
    parser.add_argument("--min-chosen-score", type=float, default=0.70)
    parser.add_argument("--max-rejected-score", type=float, default=0.30)
    parser.add_argument("--min-score-margin", type=float, default=0.40)
    parser.add_argument("--max-length-ratio", type=float, default=1.25)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    args.input = resolve(args.input or DEFAULT_SOURCE[args.split])
    args.output_dir = resolve(
        args.output_dir
        or PROJECT_ROOT / "dataset" / "GenARM" / "sentiment_preferences" / args.split
    )
    args.generator_model = resolve(args.generator_model)
    args.teacher_model = resolve(args.teacher_model)
    args.final_evaluator_model = resolve(args.final_evaluator_model)
    args.final_test = resolve(args.final_test)

    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be positive")
    if args.num_candidates < 2:
        parser.error("--num-candidates must be at least 2")
    if args.max_new_tokens < 1 or args.classifier_batch_size < 1:
        parser.error("token and batch sizes must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("invalid sampling settings")
    if not 0 <= args.min_chosen_score <= 1 or not 0 <= args.max_rejected_score <= 1:
        parser.error("classifier score thresholds must be in [0, 1]")
    if not 0 <= args.min_score_margin <= 1 or args.max_length_ratio < 1:
        parser.error("invalid pair filtering settings")
    if args.teacher_model.resolve() == args.final_evaluator_model.resolve():
        parser.error("teacher model must remain separate from the final evaluator")
    for label, path in (
        ("input", args.input),
        ("generator model", args.generator_model),
        ("teacher model", args.teacher_model),
        ("final evaluator model", args.final_evaluator_model),
        ("final test", args.final_test),
    ):
        if not path.exists():
            parser.error(f"missing {label}: {path}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidates.jsonl"
    labeled_path = args.output_dir / "labeled_candidates.jsonl"
    preference_path = args.output_dir / "preferences.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    forbidden = {
        prompt_digest(text(row.get("prompt")))
        for row in load_jsonl(args.final_test)
        if normalized_prompt(text(row.get("prompt")))
    }
    source_rows, excluded_overlap = load_source(args.input, args.max_prompts, forbidden)
    if not source_rows:
        raise RuntimeError("No eligible source prompts remain after filtering")

    candidate_rows = load_jsonl(candidate_path)
    if candidate_rows:
        candidate_by_key = {row_key(row): row for row in candidate_rows}
        candidate_rows = [
            candidate_by_key[row_key(row)]
            for row in source_rows
            if row_key(row) in candidate_by_key
        ]
    if args.stage in {"all", "generate"}:
        candidate_rows = generate_stage(source_rows, candidate_path, args)
    if args.stage in {"all", "label"}:
        if not candidate_rows:
            raise RuntimeError(f"Candidate file is missing or empty: {candidate_path}")
        label_stage(candidate_rows, labeled_path, preference_path, args)

    candidates = load_jsonl(candidate_path)
    labeled = load_jsonl(labeled_path)
    preferences = load_jsonl(preference_path)
    manifest = {
        "schema_version": 1,
        "split": args.split,
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "final_test": {"path": str(args.final_test), "sha256": sha256_file(args.final_test)},
        "excluded_final_test_overlap": excluded_overlap,
        "eligible_prompts": len(source_rows),
        "candidate_rows": len(candidates),
        "candidate_successes": sum(row.get("status") == "success" for row in candidates),
        "labeled_rows": len(labeled),
        "preference_pairs": len(preferences),
        "generator": local_model_provenance(args.generator_model),
        "teacher": local_model_provenance(args.teacher_model),
        "final_evaluator": local_model_provenance(args.final_evaluator_model),
        "generation": {
            "fingerprint": generation_fingerprint(args),
            "num_candidates": args.num_candidates,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "pair_filter": {
            "fingerprint": labeling_fingerprint(args),
            "min_chosen_score": args.min_chosen_score,
            "max_rejected_score": args.max_rejected_score,
            "min_score_margin": args.min_score_margin,
            "max_length_ratio": args.max_length_ratio,
        },
        "outputs": {
            "candidates": str(candidate_path),
            "labeled_candidates": str(labeled_path),
            "preferences": str(preference_path),
        },
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps({
        "split": args.split,
        "eligible_prompts": len(source_rows),
        "excluded_final_test_overlap": excluded_overlap,
        "candidate_rows": len(candidates),
        "preference_pairs": len(preferences),
        "output": str(preference_path),
    }, indent=2))


if __name__ == "__main__":
    main()
