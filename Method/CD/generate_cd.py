#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CD_ROOT = PROJECT_ROOT / "Method" / "CD"

BASE_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
SCORER_BACKBONE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DATASET_PATH = PROJECT_ROOT / "dataset" / "rad_benchmark" / "all.jsonl"

sys.path.insert(0, str(CD_ROOT / "models"))
sys.path.insert(0, str(CD_ROOT / "decoding"))

from prefix_scorer import PrefixScorer
import blockwise
import tokenwise


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable CD decoding.")
    parser.add_argument("--mode", choices=["tokenwise", "blockwise"], default="tokenwise")
    parser.add_argument("--scorer", choices=["fudge", "cdq"], default="fudge")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts phải >= 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens phải > 0")
    return args


def load_dataset(path, limit):
    samples = []

    if limit == 0:
        return samples

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            prompt = sample.get("prompt", "")
            prompt = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
            prompt = prompt.strip()

            if prompt:
                samples.append({**sample, "prompt_text": prompt})

            if limit and len(samples) >= limit:
                break

    return samples


def get_checkpoint_path(scorer_name):
    checkpoint_name = "cd_fudge.pt" if scorer_name == "fudge" else "cd_q.pt"
    return CD_ROOT / "checkpoints" / checkpoint_name


def get_output_path(scorer_name, mode):
    filename = f"{scorer_name}_{mode}.json"
    return PROJECT_ROOT / "results" / filename


def record_key(record):
    if record.get("md5_hash"):
        return "md5", str(record["md5_hash"])
    prompt = record.get("prompt", record.get("prompt_text", ""))
    if isinstance(prompt, dict):
        prompt = prompt.get("text", "")
    return "prompt", str(prompt).strip()


def is_completed(record):
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def load_existing_results(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"File resume phải là JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def load_models(checkpoint_path, device):
    lm_tokenizer = AutoTokenizer.from_pretrained(str(BASE_LM_PATH), local_files_only=True)
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(
        str(BASE_LM_PATH), local_files_only=True, torch_dtype=model_dtype,
    ).to(device).eval()

    scorer_tokenizer = AutoTokenizer.from_pretrained(str(SCORER_BACKBONE_PATH), local_files_only=True)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_tokenizer.padding_side = "right"

    scorer = PrefixScorer(str(SCORER_BACKBONE_PATH)).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scorer.load_state_dict(state_dict)
    scorer.eval()

    return lm, lm_tokenizer, scorer, scorer_tokenizer


def generate_response(prompt, mode, lm, lm_tokenizer, scorer, scorer_tokenizer, device):
    if mode == "tokenwise":
        return tokenwise.generate_tokenwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device)

    return blockwise.generate_blockwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_checkpoint_path(args.scorer)
    output_path = args.output_path or get_output_path(args.scorer, args.mode)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {args.dataset_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    samples = load_dataset(args.dataset_path, args.num_prompts)
    existing_results = load_existing_results(output_path)
    results_by_key = {record_key(result): result for result in existing_results}
    pending = [
        (index, sample) for index, sample in enumerate(samples)
        if not is_completed(results_by_key.get(record_key(sample)))
    ]

    print("Device:", device)
    print("Dataset:", args.dataset_path)
    print("Scorer:", args.scorer)
    print("Mode:", args.mode)
    print("Checkpoint:", checkpoint_path)
    print("Output:", output_path)
    print(f"Target: {len(samples)} | Đã có: {len(samples) - len(pending)} | Còn lại: {len(pending)}")

    if not pending:
        print("Không còn prompt nào cần chạy.")
        return

    if args.max_new_tokens is not None:
        tokenwise.MAX_NEW_TOKENS = args.max_new_tokens
        blockwise.MAX_NEW_TOKENS = args.max_new_tokens

    lm, lm_tokenizer, scorer, scorer_tokenizer = load_models(checkpoint_path, device)

    for index, sample in tqdm(pending, desc="CD decoding", unit="prompt"):
        prompt = sample["prompt_text"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            response = generate_response(prompt, args.mode, lm, lm_tokenizer, scorer, scorer_tokenizer, device)
            status = "success"
            error = None
        except Exception as exception:
            response = ""
            status = "failed"
            error = f"{type(exception).__name__}: {exception}"

        if device.type == "cuda":
            torch.cuda.synchronize()

        result = {
            "id": sample.get("id", index),
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": sample.get("continuation"),
            "response": response,
            "method": f"CD-{args.scorer.upper()}",
            "scorer": args.scorer,
            "decoding_mode": args.mode,
            "latency": time.perf_counter() - start_time,
            "status": status,
            "error": error,
        }

        results_by_key[record_key(sample)] = result
        save_results(list(results_by_key.values()), output_path)

    completed = sum(
        is_completed(results_by_key.get(record_key(sample))) for sample in samples
    )
    print(f"Completed target: {completed}/{len(samples)}")
    print(f"Saved {len(results_by_key)} records: {output_path}")


if __name__ == "__main__":
    main()
