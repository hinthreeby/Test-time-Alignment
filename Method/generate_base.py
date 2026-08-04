#!/usr/bin/env python3

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gpt2-large"
INPUT_FILE = ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"
OUTPUT_FILE = ROOT / "results" / "base.json"

NUM_PROMPTS = 300
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
SEED = 42


def get_text(value):
    return value.get("text", "") if isinstance(value, dict) else str(value or "")


def load_prompts():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    return rows[:NUM_PROMPTS] if NUM_PROMPTS else rows


def save_results(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main():
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

    results = []

    for index, sample in enumerate(tqdm(load_prompts(), desc="Generating base")):
        prompt = get_text(sample.get("prompt"))
        start = time.perf_counter()

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generate_args = {
                "max_new_tokens": MAX_NEW_TOKENS,
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

        results.append({
            "id": index,
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": get_text(sample.get("continuation")),
            "response": response,
            "method": "Base",
            "latency": round(time.perf_counter() - start, 4),
            "status": status,
            "error": error,
        })

        save_results(results)

    print(f"Saved {len(results)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()