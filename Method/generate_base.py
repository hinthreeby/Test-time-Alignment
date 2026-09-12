#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gpt2-large"
INPUT_FILE = ROOT / "dataset" / "rad_benchmark" / "all.jsonl"
OUTPUT_FILE = ROOT / "results" / "base.json"

NUM_PROMPTS = 10000
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
SEED = 42


def get_text(value):
    return value.get("text", "") if isinstance(value, dict) else str(value or "")


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable base-model generation.")
    parser.add_argument("--dataset-path", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts must be >= 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    return args


def load_prompts(input_file, num_prompts):
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")
    if num_prompts == 0:
        return []

    rows = []
    with input_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= num_prompts:
                break
    return rows


def save_results(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    temporary_file.replace(output_file)


def record_key(record):
    md5_hash = record.get("md5_hash")
    if md5_hash:
        return "md5", str(md5_hash)
    return "prompt", get_text(record.get("prompt")).strip()


def load_existing_results(output_file):
    if not output_file.exists():
        return []
    with output_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Resume output must be a JSON list: {output_file}")
    return [record for record in payload if isinstance(record, dict)]


def is_completed(record):
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def main():
    args = parse_args()
    samples = load_prompts(args.dataset_path, args.num_prompts)
    existing_results = load_existing_results(args.output_path)
    results_by_key = {record_key(record): record for record in existing_results}
    pending = [
        (index, sample)
        for index, sample in enumerate(samples)
        if not is_completed(results_by_key.get(record_key(sample)))
    ]

    print(f"Dataset: {args.dataset_path}")
    print(f"Target prompts: {len(samples)}")
    print(f"Already completed: {len(samples) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print(f"Output: {args.output_path}")

    if not pending:
        print("No prompts need generation; existing output is unchanged.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    model.eval()
    torch.manual_seed(SEED)

    for index, sample in tqdm(pending, desc="Generating base"):
        prompt = get_text(sample.get("prompt"))
        start = time.perf_counter()

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generate_args = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": DO_SAMPLE,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if DO_SAMPLE:
                generate_args.update({
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                })

            with torch.inference_mode():
                output = model.generate(**inputs, **generate_args)

            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            status = "success"
            error = None

        except Exception as exc:
            response = None
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        results_by_key[record_key(sample)] = {
            "id": index,
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": get_text(sample.get("continuation")),
            "response": response,
            "method": "Base",
            "latency": round(time.perf_counter() - start, 4),
            "status": status,
            "error": error,
        }

        save_results(list(results_by_key.values()), args.output_path)

    results = list(results_by_key.values())
    save_results(results, args.output_path)
    completed = sum(
        is_completed(results_by_key.get(record_key(sample))) for sample in samples
    )
    print(f"Completed target: {completed}/{len(samples)}")
    print(f"Saved {len(results)} total records to {args.output_path}")


if __name__ == "__main__":
    main()
