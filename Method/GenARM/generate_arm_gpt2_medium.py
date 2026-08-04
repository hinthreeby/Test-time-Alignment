#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-large"
ARM_MODEL_PATH = PROJECT_ROOT / "models" / "genarm-gpt2-medium-hh"
INPUT_FILE = PROJECT_ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "results" / "genarm.json"

NUM_PROMPTS = 300
MAX_NEW_TOKENS = 64
ALPHA = 2.0
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
SEED = 42
SAVE_EVERY = 1


def save_results(results: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def load_dataset() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy benchmark: {INPUT_FILE}")
    samples = []
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON lỗi tại dòng {line_number}: {error}") from error
    return samples[:NUM_PROMPTS] if NUM_PROMPTS is not None else samples


def extract_prompt(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    return str(prompt.get("text", "")) if isinstance(prompt, dict) else str(prompt)


def extract_reference(sample: dict):
    continuation = sample.get("continuation")
    return continuation.get("text") if isinstance(continuation, dict) else continuation


def direct_prompt_template(system_prompt: str, input_string: str) -> str:
    return input_string


def main() -> None:
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

    dataset = load_dataset()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    results = []

    for index, sample in enumerate(tqdm(dataset, desc="GenARM decoding", unit="prompt")):
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

        results.append({
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
        })

        if len(results) % SAVE_EVERY == 0:
            save_results(results)

    save_results(results)
    print(f"Saved {len(results)} samples to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()