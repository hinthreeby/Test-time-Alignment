#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.rad import RewardAugmentedDecoder
from Method.RAD.reward_modeling.reward_model import GPT2RewardModel
from Method.RAD.utils.metrics import compute_perplexity, distinctness


DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"
DEFAULT_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_RM_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_RM_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_RM_CHECKPOINT_PATH = DEFAULT_RM_TOKENIZER_PATH / "pytorch_model.bin"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "rad.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "rad_report.json"


def load_dataset(path: Path, num_prompts: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {error}"
                ) from error

            prompt_value = sample.get("prompt", "")
            prompt_text = (
                str(prompt_value.get("text", "")).strip()
                if isinstance(prompt_value, dict)
                else str(prompt_value).strip()
            )

            if not prompt_text:
                continue

            sample["_prompt_text"] = prompt_text
            sample["_dataset_index"] = len(samples)
            samples.append(sample)

            if num_prompts is not None and len(samples) >= num_prompts:
                break

    return samples


def chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def load_base_lm(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Base LM path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval().to(device)

    max_length = int(
        getattr(
            model.config,
            "n_positions",
            getattr(model.config, "max_position_embeddings", 1024),
        )
    )

    return model, tokenizer, max_length


def load_reward_model(
    rm_base_path: Path,
    rm_tokenizer_path: Path,
    rm_checkpoint_path: Path,
    device: torch.device,
):
    if not rm_base_path.exists():
        raise FileNotFoundError(f"RM base path not found: {rm_base_path}")
    if not rm_tokenizer_path.exists():
        raise FileNotFoundError(f"RM tokenizer path not found: {rm_tokenizer_path}")
    if not rm_checkpoint_path.exists():
        raise FileNotFoundError(f"RM checkpoint not found: {rm_checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(rm_tokenizer_path),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 256

    reward_model = GPT2RewardModel(
        reward_model_name=str(rm_base_path),
        out_features=1,
        loss_fn="cumulative_mse",
    )

    state_dict = torch.load(str(rm_checkpoint_path), map_location="cpu")
    # RAD's original Transformers version persisted GPT-2 causal-mask buffers
    # (attn.bias and attn.masked_bias). Newer Transformers versions recreate
    # those buffers instead of registering them in the state dict. Allow only
    # these known legacy extras while keeping all learned weights strict.
    load_result = reward_model.load_state_dict(state_dict, strict=False)
    allowed_legacy_suffixes = (".attn.bias", ".attn.masked_bias")
    unexpected_learned_keys = [
        key
        for key in load_result.unexpected_keys
        if not key.endswith(allowed_legacy_suffixes)
    ]
    if load_result.missing_keys or unexpected_learned_keys:
        raise RuntimeError(
            "Incompatible RAD reward checkpoint: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={unexpected_learned_keys}"
        )

    print("RM missing keys:", load_result.missing_keys)
    print("RM unexpected keys:", load_result.unexpected_keys)

    reward_model.eval().to(device)
    return reward_model, tokenizer


def load_rad(args: argparse.Namespace) -> RewardAugmentedDecoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading base LM from: {args.lm_path}")
    lm, lm_tokenizer, max_length = load_base_lm(args.lm_path, device)

    print(f"Loading RM checkpoint from: {args.rm_checkpoint_path}")
    rm, rm_tokenizer = load_reward_model(
        args.rm_base_path,
        args.rm_tokenizer_path,
        args.rm_checkpoint_path,
        device,
    )

    return RewardAugmentedDecoder(
        lm,
        lm_tokenizer,
        rm,
        rm_tokenizer,
        max_length,
        num_gpus=max(torch.cuda.device_count(), 1),
        inverse=args.inverse,
    )


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    md5_hash = record.get("md5_hash")
    if md5_hash:
        return "md5", str(md5_hash)

    prompt_value = record.get("_prompt_text", record.get("prompt", ""))
    if isinstance(prompt_value, dict):
        prompt_value = prompt_value.get("text", "")
    return "prompt", str(prompt_value).strip()


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Resume output must be a JSON list: {path}")
    return [record for record in payload if isinstance(record, dict)]


def is_completed(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def generate_on_prompts(
    args: argparse.Namespace,
    rad: RewardAugmentedDecoder,
    samples: list[dict[str, Any]],
    existing_results: list[dict[str, Any]] | None = None,
):
    sample_chunks = list(chunks(samples, args.batch_size))
    flat_results = list(existing_results or [])
    result_indices = {
        record_key(record): index for index, record in enumerate(flat_results)
    }
    legacy_generation = []
    dist_n_values = [
        distinctness(record["all_responses"])
        for record in flat_results
        if is_completed(record) and record.get("all_responses")
    ]

    def upsert_result(source_sample: dict[str, Any], result: dict[str, Any]) -> None:
        key = record_key(source_sample)
        existing_index = result_indices.get(key)
        if existing_index is None:
            result_indices[key] = len(flat_results)
            flat_results.append(result)
        else:
            flat_results[existing_index] = result

    progress_bar = tqdm(sample_chunks, desc="RAD decoding", unit="batch")

    for sample_chunk in progress_bar:
        prompts = [sample["_prompt_text"] for sample in sample_chunk]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            with torch.inference_mode():
                generated_texts = rad.sample(
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                    topk=args.topk,
                    beta=args.beta,
                    num_return_sequences=args.num_return_sequences,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_latency = time.perf_counter() - start_time
            latency_per_prompt = batch_latency / max(len(prompts), 1)

            for local_index, generated_samples in enumerate(generated_texts):
                source_sample = sample_chunk[local_index]

                if isinstance(generated_samples, str):
                    generated_samples = [generated_samples]

                generated_samples = [str(text).strip() for text in generated_samples]
                response = generated_samples[0] if generated_samples else None

                if generated_samples:
                    dist_n_values.append(distinctness(generated_samples))

                continuation = source_sample.get("continuation", {})
                reference = (
                    continuation.get("text")
                    if isinstance(continuation, dict)
                    else continuation
                )

                upsert_result(
                    source_sample,
                    {
                        "id": source_sample["_dataset_index"],
                        "md5_hash": source_sample.get("md5_hash"),
                        "prompt": prompts[local_index],
                        "reference": reference,
                        "response": response,
                        "all_responses": generated_samples,
                        "num_positive": source_sample.get("num_positive"),
                        "method": "RAD",
                        "base_model": str(args.lm_path),
                        "reward_model": str(args.rm_checkpoint_path),
                        "beta": args.beta,
                        "top_k": args.topk,
                        "max_new_tokens": args.max_new_tokens,
                        "num_return_sequences": args.num_return_sequences,
                        "inverse": args.inverse,
                        "latency": round(latency_per_prompt, 4),
                        "status": "success" if response else "failed",
                        "error": None if response else "RAD returned no generations.",
                    }
                )

                legacy_generation.append(
                    {
                        "prompt": {"text": prompts[local_index]},
                        "generations": [{"text": text} for text in generated_samples],
                    }
                )

        except Exception as error:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_latency = time.perf_counter() - start_time
            latency_per_prompt = batch_latency / max(len(prompts), 1)

            for local_index, source_sample in enumerate(sample_chunk):
                continuation = source_sample.get("continuation", {})
                reference = (
                    continuation.get("text")
                    if isinstance(continuation, dict)
                    else continuation
                )

                upsert_result(
                    source_sample,
                    {
                        "id": source_sample["_dataset_index"],
                        "md5_hash": source_sample.get("md5_hash"),
                        "prompt": prompts[local_index],
                        "reference": reference,
                        "response": None,
                        "all_responses": [],
                        "num_positive": source_sample.get("num_positive"),
                        "method": "RAD",
                        "base_model": str(args.lm_path),
                        "reward_model": str(args.rm_checkpoint_path),
                        "beta": args.beta,
                        "top_k": args.topk,
                        "max_new_tokens": args.max_new_tokens,
                        "num_return_sequences": args.num_return_sequences,
                        "inverse": args.inverse,
                        "latency": round(latency_per_prompt, 4),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        save_json(args.output_path, flat_results)

        if dist_n_values:
            mean_dist_n = np.nanmean(np.asarray(dist_n_values), axis=0)
            progress_bar.set_postfix(
                samples=len(flat_results),
                dist_n="/".join(
                    f"{value:.3f}" for value in np.atleast_1d(mean_dist_n)
                ),
            )
        else:
            progress_bar.set_postfix(samples=len(flat_results))

    report = {
        "method": "RAD",
        "num_samples": len(flat_results),
        "num_success": sum(item["status"] == "success" for item in flat_results),
        "num_failed": sum(item["status"] != "success" for item in flat_results),
        "beta": args.beta,
        "top_k": args.topk,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
        "inverse": args.inverse,
        "average_latency": (
            float(np.mean([item["latency"] for item in flat_results]))
            if flat_results
            else 0.0
        ),
    }

    if dist_n_values:
        report["dist_n"] = np.nanmean(
            np.asarray(dist_n_values),
            axis=0,
        ).tolist()

    if args.compute_perplexity and legacy_generation:
        try:
            perplexities = compute_perplexity(args, legacy_generation, rad)
            report["perplexity"] = float(np.mean(perplexities))
        except Exception as error:
            report["perplexity_error"] = f"{type(error).__name__}: {error}"

    save_json(args.report_path, report)
    return report, flat_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAD sentiment decoding on the shared benchmark."
    )

    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)

    parser.add_argument("--lm-path", type=Path, default=DEFAULT_LM_PATH)
    parser.add_argument("--rm-base-path", type=Path, default=DEFAULT_RM_BASE_PATH)
    parser.add_argument(
        "--rm-tokenizer-path",
        type=Path,
        default=DEFAULT_RM_TOKENIZER_PATH,
    )
    parser.add_argument(
        "--rm-checkpoint-path",
        type=Path,
        default=DEFAULT_RM_CHECKPOINT_PATH,
    )

    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--compute-perplexity", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Beta: {args.beta}")
    print(f"Top-k: {args.topk}")

    samples = load_dataset(args.dataset_path, args.num_prompts)
    print(f"Loaded {len(samples)} prompts.")

    existing_results = load_existing_results(args.output_path)
    existing_by_key = {
        record_key(record): record for record in existing_results
    }
    pending_samples = [
        sample
        for sample in samples
        if not is_completed(existing_by_key.get(record_key(sample)))
    ]
    print(f"Already completed: {len(samples) - len(pending_samples)}")
    print(f"Remaining: {len(pending_samples)}")

    if not pending_samples:
        print("No prompts need decoding; existing output is unchanged.")
        return

    rad = load_rad(args)
    report, _ = generate_on_prompts(
        args,
        rad,
        pending_samples,
        existing_results=existing_results,
    )

    print("\nFinished.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
