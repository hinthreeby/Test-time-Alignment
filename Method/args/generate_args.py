import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from argsearch import ARGS


BASE_DIR = Path(__file__).resolve().parents[2]

LLM_PATH = BASE_DIR / "models" / "gpt2-large"
RM_PATH = BASE_DIR / "models" / "gpt2-large-helpful-rm"

INPUT_FILE = BASE_DIR / "dataset" / "rad_benchmark" / "negative_prompts.jsonl"




NUM_PROMPTS = 300       # Đặt None để chạy toàn bộ dataset
TOPK = 10
WEIGHT = 2.0
MAX_NEW_TOKEN = 128
# METHOD = "greedy"
METHOD = "topk"

if METHOD not in {"topk", "greedy"}:
    raise ValueError("METHOD phải là 'topk' hoặc 'greedy'")

OUTPUT_FILE = BASE_DIR / "results" / f"args_{METHOD}.json"


searcher = ARGS(
    llm_path=str(LLM_PATH),
    rm_path=str(RM_PATH),
    llm_dev="cuda:0",
    rm_dev="cuda:0",
    torch_dtype=torch.float16,
)


dataset = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            dataset.append(json.loads(line))

if NUM_PROMPTS is not None:
    dataset = dataset[:NUM_PROMPTS]

print(f"Generating {len(dataset)} prompts.")

results = []

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

progress_bar = tqdm(dataset, total=len(dataset), desc="ARGS decoding", unit="prompt",)

for idx, sample in enumerate(progress_bar):
    prompt = sample["prompt"]["text"]

    start_time = time.perf_counter()

    output_tokens = searcher.generate(
        prompt=prompt,
        topk=TOPK,
        weight=WEIGHT,
        max_new_token=MAX_NEW_TOKEN,
        method=METHOD,
        debug=False,
    )

    if output_tokens is None:
        response = None
        status = "failed"
    else:
        response = searcher.tokens_to_text(output_tokens)[0]
        status = "success"

    latency = time.perf_counter() - start_time

    results.append(
        {
            "id": idx,
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": sample.get("continuation", {}).get("text"),
            "response": response,
            "num_positive": sample.get("num_positive"),
            "method": "ARGS",
            "decoding_method": METHOD,
            "topk": TOPK,
            "weight": WEIGHT,
            "max_new_tokens": MAX_NEW_TOKEN,
            "latency": round(latency, 4),
            "status": status,
        }
    )

    progress_bar.set_postfix(
        latency=f"{latency:.2f}s",
        saved=len(results),
    )

    # Lưu sau mỗi prompt để tránh mất kết quả nếu chương trình bị dừng
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


print(f"Generation finished. Saved to: {OUTPUT_FILE}")