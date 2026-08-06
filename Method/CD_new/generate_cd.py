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
CD_ROOT = PROJECT_ROOT / "Method" / "CD_new"

BASE_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
SCORER_BACKBONE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DATASET_PATH = PROJECT_ROOT / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"

sys.path.insert(0, str(CD_ROOT / "models"))
sys.path.insert(0, str(CD_ROOT / "decoding"))

from prefix_scorer import PrefixScorer
import blockwise
import tokenwise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tokenwise", "blockwise"], default="tokenwise")
    parser.add_argument("--scorer", choices=["fudge", "cdq"], default="fudge")
    parser.add_argument("--num-prompts", type=int, default=100)
    return parser.parse_args()


def load_dataset(path, limit):
    samples = []

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

            if len(samples) >= limit:
                break

    return samples


def get_checkpoint_path(scorer_name):
    checkpoint_name = "cd_fudge.pt" if scorer_name == "fudge" else "cd_q.pt"
    return CD_ROOT / "checkpoints" / checkpoint_name


def get_output_path(scorer_name, mode):
    filename = f"{scorer_name}_{mode}.json"
    return PROJECT_ROOT / scorer_name / filename


def load_models(checkpoint_path, device):
    lm_tokenizer = AutoTokenizer.from_pretrained(str(BASE_LM_PATH), local_files_only=True)
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

    lm = AutoModelForCausalLM.from_pretrained(str(BASE_LM_PATH), local_files_only=True).to(device).eval()

    scorer_tokenizer = AutoTokenizer.from_pretrained(str(SCORER_BACKBONE_PATH), local_files_only=True)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_tokenizer.padding_side = "right"

    scorer = PrefixScorer(str(SCORER_BACKBONE_PATH)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
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
    output_path = get_output_path(args.scorer, args.mode)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {DATASET_PATH}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    print("Device:", device)
    print("Scorer:", args.scorer)
    print("Mode:", args.mode)
    print("Checkpoint:", checkpoint_path)
    print("Output:", output_path)

    lm, lm_tokenizer, scorer, scorer_tokenizer = load_models(checkpoint_path, device)
    samples = load_dataset(DATASET_PATH, args.num_prompts)
    results = []

    for index, sample in enumerate(tqdm(samples, desc="Generating")):
        prompt = sample["prompt_text"]
        start_time = time.perf_counter()

        try:
            response = generate_response(prompt, args.mode, lm, lm_tokenizer, scorer, scorer_tokenizer, device)
            status = "success"
            error = None
        except Exception as exception:
            response = ""
            status = "error"
            error = str(exception)

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

        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    success_count = sum(result["status"] == "success" for result in results)

    print(f"Completed: {success_count}/{len(results)}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()