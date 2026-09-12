#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from argsearch import ARGS


BASE_DIR = Path(__file__).resolve().parents[2]
LLM_PATH = BASE_DIR / "models" / "gpt2-large"
RM_PATH = BASE_DIR / "models" / "rad_rm_sentiment"
RM_BASE_PATH = BASE_DIR / "models" / "gpt2-small"
INPUT_FILE = BASE_DIR / "dataset" / "rad_benchmark" / "all.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable ARGS decoding.")
    parser.add_argument("--method", choices=["greedy", "topk"], default="topk")
    parser.add_argument("--num-prompts", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--input-path", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts phải >= 0")
    if args.topk <= 0 or args.max_new_tokens <= 0 or args.temperature <= 0:
        parser.error("topk, max-new-tokens và temperature phải > 0")
    if args.output_path is None:
        args.output_path = BASE_DIR / "results" / f"args_{args.method}.json"
    return args


def extract_prompt(sample):
    prompt = sample.get("prompt", "")
    return str(prompt.get("text", "")) if isinstance(prompt, dict) else str(prompt)


def extract_reference(sample):
    continuation = sample.get("continuation")
    return continuation.get("text") if isinstance(continuation, dict) else continuation


def record_key(record):
    if record.get("md5_hash"):
        return "md5", str(record["md5_hash"])
    return "prompt", extract_prompt(record).strip()


def is_completed(record):
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def load_dataset(path, limit):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy benchmark: {path}")
    if limit == 0:
        return []
    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON lỗi tại dòng {line_number}: {error}") from error
            if extract_prompt(sample).strip():
                samples.append(sample)
            if len(samples) >= limit:
                break
    return samples


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


def main():
    args = parse_args()
    dataset = load_dataset(args.input_path, args.num_prompts)
    existing_results = load_existing_results(args.output_path)
    results_by_key = {record_key(record): record for record in existing_results}
    pending = [
        (index, sample) for index, sample in enumerate(dataset)
        if not is_completed(results_by_key.get(record_key(sample)))
    ]

    print(f"Method: ARGS-{args.method}")
    print(f"Dataset: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Target: {len(dataset)} | Đã có: {len(dataset) - len(pending)} | Còn lại: {len(pending)}")

    if not pending:
        print("Không còn prompt nào cần chạy.")
        return
    if not LLM_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy base model: {LLM_PATH}")
    if not RM_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ARGS reward model: {RM_PATH}\n"
            "Hãy train RAD reward model trước."
        )
    if not RM_BASE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy reward backbone: {RM_BASE_PATH}")

    searcher = ARGS(
        llm_path=str(LLM_PATH),
        rm_path=str(RM_PATH),
        llm_dev="cuda:0",
        rm_dev="cuda:0",
        torch_dtype=torch.float16,
        rm_base_path=str(RM_BASE_PATH),
    )

    for index, sample in tqdm(pending, desc="ARGS decoding", unit="prompt"):
        prompt = extract_prompt(sample).strip()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        try:
            output_tokens = searcher.generate(
                prompt=prompt,
                topk=args.topk,
                weight=args.weight,
                max_new_token=args.max_new_tokens,
                method=args.method,
                temperature=args.temperature,
                debug=False,
            )
            if output_tokens is None:
                raise RuntimeError("ARGS không sinh được output")
            prompt_length = searcher.get_input_ids(prompt).shape[1]
            response = searcher.tokens_to_text(output_tokens[:, prompt_length:])[0].strip()
            status, error_message = "success", None
        except Exception as error:
            response = None
            status, error_message = "failed", f"{type(error).__name__}: {error}"
        torch.cuda.synchronize()

        result = {
            "id": sample.get("id", index),
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": extract_reference(sample),
            "response": response,
            "num_positive": sample.get("num_positive"),
            "method": "ARGS",
            "reward_model": str(RM_PATH),
            "decoding_method": args.method,
            "topk": args.topk,
            "weight": args.weight,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "latency": round(time.perf_counter() - start_time, 4),
            "status": status,
            "error": error_message,
        }
        results_by_key[record_key(sample)] = result
        save_results(list(results_by_key.values()), args.output_path)

    completed = sum(
        is_completed(results_by_key.get(record_key(sample))) for sample in dataset
    )
    print(f"Completed target: {completed}/{len(dataset)}")
    print(f"Saved {len(results_by_key)} records: {args.output_path}")


if __name__ == "__main__":
    main()
