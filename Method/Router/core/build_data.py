#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.Router.core.utils import load_config


def split_text(text, ratio, min_words):
    words = str(text).strip().split()
    if len(words) < min_words:
        return None
    index = min(max(2, int(len(words) * ratio)), len(words) - 2)
    prompt, response = " ".join(words[:index]).strip(), " ".join(words[index:]).strip()
    return {"prompt": prompt, "response": response} if prompt and response else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT, args.method)
    data = config["router_data"]
    random.seed(config["seed"])

    dataset = load_dataset(data["dataset"], split=data["split"])
    texts = []

    for item in dataset:
        if data.get("label_field") is not None and int(item[data["label_field"]]) != int(data["label_value"]):
            continue
        texts.append(item[data["text_field"]])

    random.shuffle(texts)
    samples = []

    for text in texts:
        sample = split_text(text, data["prompt_ratio"], data["min_words"])
        if sample:
            samples.append(sample)
        if len(samples) >= data["num_train"] + data["num_val"]:
            break

    train = samples[:data["num_train"]]
    val = samples[data["num_train"]:data["num_train"] + data["num_val"]]

    output_dir = PROJECT_ROOT / "dataset" / "router_train" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in [("train", train), ("validation", val)]:
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
