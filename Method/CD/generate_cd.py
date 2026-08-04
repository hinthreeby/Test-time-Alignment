#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(CD_ROOT) not in sys.path:
    sys.path.insert(0, str(CD_ROOT))

from decoding.cd_decoder import CDGenerationConfig, ControlledDecoder
from models.prefix_scorer import PrefixScorer, load_prefix_tokenizer

INPUT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "rad_benchmark"
    / "negative_prompts.jsonl"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "cd_results.json"

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-small"
PREFIX_SCORER_PATH = PROJECT_ROOT / "models" / "cd_prefix_scorer"


# ============================================================
# GENERATION CONFIG — chỉnh tại đây
# ============================================================

NUM_PROMPTS = 300
MAX_NEW_TOKENS = 64

MODE = "tokenwise"          # "tokenwise" hoặc "blockwise"
LAMBDA_WEIGHT = 4.0

TOP_K = 20
TEMPERATURE = 1.0
SAMPLING = "greedy"         # "greedy" hoặc "sample"

BLOCK_SIZE = 8
NUM_CANDIDATES = 4
TOP_P = 0.95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--base-model", default=str(BASE_MODEL_PATH))
    parser.add_argument("--prefix-scorer", default=str(PREFIX_SCORER_PATH))
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--mode", choices=["tokenwise", "blockwise"], default=MODE)
    parser.add_argument("--lambda-weight", type=float, default=LAMBDA_WEIGHT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--sampling", choices=["greedy", "sample"], default=SAMPLING)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--num-candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy input file: {path}")

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
                    f"JSON lỗi tại dòng {line_number}: {error}"
                ) from error

            samples.append(sample)

    if not samples:
        raise ValueError(f"Input file không có dữ liệu: {path}")

    return samples


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def extract_prompt(sample: dict) -> str:
    prompt_data = sample.get("prompt", "")

    if isinstance(prompt_data, dict):
        return str(prompt_data.get("text", ""))

    return str(prompt_data)


def extract_reference(sample: dict):
    continuation = sample.get("continuation")

    if isinstance(continuation, dict):
        return continuation.get("text")

    return continuation


def main():
    args = parse_args()

    if args.mode not in {"tokenwise", "blockwise"}:
        raise ValueError("MODE phải là 'tokenwise' hoặc 'blockwise'.")

    if args.sampling not in {"greedy", "sample"}:
        raise ValueError("SAMPLING phải là 'greedy' hoặc 'sample'.")

    if not Path(args.base_model).exists():
        raise FileNotFoundError(f"Không tìm thấy base model: {args.base_model}")

    if not Path(args.prefix_scorer).exists():
        raise FileNotFoundError(
            f"Không tìm thấy prefix scorer: {args.prefix_scorer}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Controlled Decoding")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Base model: {args.base_model}")
    print(f"Prefix scorer: {args.prefix_scorer}")
    print(f"Mode: {args.mode}")
    print(f"Lambda: {args.lambda_weight}")
    print(f"Number of prompts: {args.num_prompts}")
    print("=" * 70)

    print("Loading base tokenizer...")

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=True,
    )

    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)

    base_model.eval()

    print("Loading prefix scorer...")

    prefix_scorer = PrefixScorer.from_pretrained(
        args.prefix_scorer,
        map_location=device,
    ).to(device)

    prefix_scorer.eval()

    scorer_tokenizer = load_prefix_tokenizer(args.prefix_scorer)

    decoder = ControlledDecoder(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        prefix_scorer=prefix_scorer,
        scorer_tokenizer=scorer_tokenizer,
        base_device=device,
        scorer_device=device,
    )

    dataset = load_dataset(args.input)

    if args.num_prompts is not None:
        dataset = dataset[:args.num_prompts]

    print(f"Loaded {len(dataset)} prompts.")

    results = []

    progress_bar = tqdm(dataset, desc="CD decoding", unit="prompt")

    for index, sample in enumerate(progress_bar):
        prompt = extract_prompt(sample)
        start_time = time.perf_counter()

        try:
            if args.mode == "tokenwise":
                response = decoder.generate_tokenwise(
                    prompt,
                    CDGenerationConfig(
                        lambda_weight=args.lambda_weight,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        method=args.sampling,
                    ),
                )
            else:
                response = decoder.generate_blockwise(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    num_candidates=args.num_candidates,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            status = "success"
            error_message = None

        except Exception as error:
            response = None
            status = "failed"
            error_message = f"{type(error).__name__}: {error}"

        latency = time.perf_counter() - start_time

        result = {
            "id": index,
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": extract_reference(sample),
            "response": response,
            "num_positive": sample.get("num_positive"),
            "method": "CD",
            "decoding_mode": args.mode,
            "lambda_weight": args.lambda_weight,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "sampling": args.sampling,
            "block_size": args.block_size,
            "num_candidates": args.num_candidates,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "latency": round(latency, 4),
            "status": status,
            "error": error_message,
        }

        results.append(result)
        save_results(args.output, results)

        progress_bar.set_postfix(
            latency=f"{latency:.2f}s",
            status=status,
        )

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()