#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-large"
ARM_MODEL_PATH = PROJECT_ROOT / "models" / "genarm-gpt2-small-hh"
INPUT_FILE = PROJECT_ROOT / "dataset" / "rad_benchmark" / "all.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "results" / "genarm.json"

NUM_PROMPTS = 10000
MAX_NEW_TOKENS = 128
ALPHA = 1.0
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0
SEED = 42
SAVE_EVERY = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable GenARM decoding.")
    parser.add_argument("--dataset-path", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts must be >= 0")
    return args


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    temporary_path.replace(output_path)


def load_dataset(dataset_path: Path, num_prompts: int) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy benchmark: {dataset_path}")
    if num_prompts == 0:
        return []
    samples = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON lỗi tại dòng {line_number}: {error}") from error
            if len(samples) >= num_prompts:
                break
    return samples


def extract_prompt(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    return str(prompt.get("text", "")) if isinstance(prompt, dict) else str(prompt)


def extract_reference(sample: dict):
    continuation = sample.get("continuation")
    return continuation.get("text") if isinstance(continuation, dict) else continuation


def record_key(record: dict) -> tuple[str, str]:
    md5_hash = record.get("md5_hash")
    if md5_hash:
        return "md5", str(md5_hash)
    return "prompt", extract_prompt(record).strip()


def load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Kết quả resume phải là JSON list: {output_path}")
    return [record for record in payload if isinstance(record, dict)]


def is_completed(record: dict) -> bool:
    return (
        record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and bool(record["response"].strip())
    )


def direct_prompt_template(system_prompt: str, input_string: str) -> str:
    return input_string


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_path, args.num_prompts)
    existing_results = load_existing_results(args.output_path)
    results_by_key = {record_key(record): record for record in existing_results}
    pending = [
        (index, sample)
        for index, sample in enumerate(dataset)
        if not is_completed(results_by_key.get(record_key(sample), {}))
    ]

    completed_in_target = len(dataset) - len(pending)
    print(f"Dataset: {args.dataset_path}")
    print(f"Target prompts: {len(dataset)}")
    print(f"Already completed: {completed_in_target}")
    print(f"Remaining: {len(pending)}")
    print(f"Output: {args.output_path}")

    if not pending:
        print("Không còn prompt nào cần chạy.")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH), local_files_only=True, torch_dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    )
    arm_model = AutoModelForCausalLM.from_pretrained(
        str(ARM_MODEL_PATH), local_files_only=True, torch_dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    )
    base_model.eval()
    arm_model.eval()

    if base_model.config.vocab_size != arm_model.config.vocab_size:
        raise ValueError("Base LM và ARM có vocab_size khác nhau")

    m_base = PromptedLLM(
        system_prompt="",
        prompt_template=direct_prompt_template,
        model=base_model,
        tokenizer=tokenizer,
        run_eager=True,
    )
    m_arm = PromptedLLM(
        system_prompt="",
        prompt_template=direct_prompt_template,
        model=arm_model,
        tokenizer=tokenizer,
        run_eager=True,
    )
    genarm = ModelArithmetic(
        m_base + ALPHA * m_arm,
        needs_input_tokens_lm_eval=False,
        lm_eval_task=None,
        dtype=torch.float16,
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generated_count = 0

    for index, sample in tqdm(pending, desc="GenARM decoding", unit="prompt"):
        prompt = extract_prompt(sample)
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            outputs = genarm.generate_text(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE / (1.0 + ALPHA),
                top_p=TOP_P,
                top_k=TOP_K,
                do_speculation=False,
            )
            response = str(outputs[0]).removesuffix(tokenizer.eos_token or "").strip()
            status, error_message = "success", None
        except Exception as error:
            response = None
            status, error_message = "failed", f"{type(error).__name__}: {error}"
        torch.cuda.synchronize()
        latency = time.perf_counter() - start

        result = {
            "id": index,
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": extract_reference(sample),
            "response": response,
            "num_positive": sample.get("num_positive"),
            "method": "GenARM",
            "base_model": str(BASE_MODEL_PATH),
            "reward_model": str(ARM_MODEL_PATH),
            "alpha": ALPHA,
            "temperature": TEMPERATURE,
            "effective_temperature": TEMPERATURE / (1.0 + ALPHA),
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "latency": round(latency, 4),
            "status": status,
            "error": error_message,
        }
        results_by_key[record_key(sample)] = result
        generated_count += 1

        if generated_count % SAVE_EVERY == 0:
            save_results(list(results_by_key.values()), args.output_path)

    results = list(results_by_key.values())
    save_results(results, args.output_path)
    completed_after_run = sum(
        is_completed(results_by_key.get(record_key(sample), {}))
        for sample in dataset
    )
    print(f"Completed target: {completed_after_run}/{len(dataset)}")
    print(f"Saved {len(results)} total records to: {args.output_path}")


if __name__ == "__main__":
    main()
