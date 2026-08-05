#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


METHOD_FILES = {
    "Base": "base.json",
    "ARGS": "args.json",
    "rad-fixed": "rad_fixed.json",
    "rad-confidence": "rad_confidence.json",
    "rad-reward": "rad_reward.json",
    "rad-hybrid": "rad_hybrid.json",
    "rad-length": "rad_length.json",
    "CD-FUDGE-tokenwise": "fudge/fudge_tokenwise.json",
    "CD-FUDGE-blockwise": "fudge/fudge_blockwise.json",
    "CD-Q-tokenwise": "cdq/cdq_tokenwise.json",
    "CD-Q-blockwise": "cdq/cdq_blockwise.json",
    "GenARM": "genarm.json",
}

BASE_MODELS = {
    "Base": "gpt2-large",
    "ARGS": "gpt2-large",
    "rad-fixed": "gpt2-large",
    "rad-confidence": "gpt2-large",
    "rad-reward": "gpt2-large",
    "rad-hybrid": "gpt2-large",
    "rad-length": "gpt2-large",
    "CD-FUDGE-tokenwise": "gpt2-large",
    "CD-FUDGE-blockwise": "gpt2-large",
    "CD-Q-tokenwise": "gpt2-large",
    "CD-Q-blockwise": "gpt2-large",
    "GenARM": "gpt2-large",
}

REWARD_MODELS = {
    "Base": "none",
    "ARGS": "configured RM",
    "rad-fixed": "rad_rm_sentiment",
    "rad-confidence": "rad_rm_sentiment",
    "rad-reward": "rad_rm_sentiment",
    "rad-hybrid": "rad_rm_sentiment",
    "rad-length": "rad_rm_sentiment",
    "CD-FUDGE-tokenwise": "CD-FUDGE prefix scorer",
    "CD-FUDGE-blockwise": "CD-FUDGE prefix scorer",
    "CD-Q-tokenwise": "CD-Q prefix scorer",
    "CD-Q-blockwise": "CD-Q prefix scorer",
    "GenARM": "genarm-gpt2-medium-hh",
}

# File sinh từ base LM không áp dụng controlled decoding.
BASELINE_FILE = "base.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SENTIMENT_MODEL = (
    PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
)
DEFAULT_REWARD_MODEL = (
    PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
)
DEFAULT_TOXICITY_MODEL = PROJECT_ROOT / "models" / "toxic-bert"
DEFAULT_HELPFULNESS_MODEL = (
    PROJECT_ROOT / "models" / "helpfulness-deberta-v3-large"
)
DEFAULT_PPL_MODEL = PROJECT_ROOT / "models" / "gpt2-large"

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--sentiment-model",
        default=str(DEFAULT_SENTIMENT_MODEL),
    )
    parser.add_argument(
        "--reward-model",
        default=str(DEFAULT_REWARD_MODEL),
    )
    parser.add_argument(
        "--toxicity-model",
        default=str(DEFAULT_TOXICITY_MODEL),
    )
    parser.add_argument(
        "--helpfulness-model",
        default=str(DEFAULT_HELPFULNESS_MODEL),
    )
    parser.add_argument("--toxicity-threshold", type=float, default=0.5)
    parser.add_argument("--helpfulness-threshold", type=float, default=0.0)
    parser.add_argument("--baseline-file", default=BASELINE_FILE)
    parser.add_argument(
        "--ppl-model",
        default=str(DEFAULT_PPL_MODEL),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ppl-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def get_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_records(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    records = []

    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("status", "success") != "success":
            continue

        response = item.get("response")

        if not isinstance(response, str) or not response.strip():
            continue

        prompt = item.get("prompt", "")
        latency = item.get("latency")

        try:
            latency = float(latency) if latency is not None else None
        except (TypeError, ValueError):
            latency = None

        records.append(
            {
                **item,
                "prompt": str(prompt).strip(),
                "response": response.strip(),
                "latency": latency,
            }
        )

    return records


def record_key(record):
    md5_hash = record.get("md5_hash")

    if md5_hash:
        return ("md5", str(md5_hash))

    if record.get("id") is not None:
        return ("id", str(record["id"]))

    return ("prompt", record.get("prompt", ""))


def align_with_baseline(records, baseline_records):
    baseline_map = {
        record_key(record): record
        for record in baseline_records
    }

    method_aligned = []
    baseline_aligned = []

    for record in records:
        baseline = baseline_map.get(record_key(record))

        if baseline is None:
            continue

        method_aligned.append(record)
        baseline_aligned.append(baseline)

    return method_aligned, baseline_aligned


def tokenize_words(text):
    return WORD_RE.findall(text.lower())


def ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts, n):
    grams = []

    for text in texts:
        grams.extend(ngrams(tokenize_words(text), n))

    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(texts, n=4):
    total = 0
    repeated = 0

    for text in texts:
        grams = ngrams(tokenize_words(text), n)
        counts = Counter(grams)
        total += len(grams)
        repeated += sum(max(count - 1, 0) for count in counts.values())

    return repeated / total if total else 0.0


def average_length(texts):
    return (
        sum(len(tokenize_words(text)) for text in texts) / len(texts)
        if texts
        else 0.0
    )


def mean(values):
    valid = []

    for value in values:
        if value is None:
            continue

        value = float(value)

        if math.isfinite(value):
            valid.append(value)

    return sum(valid) / len(valid) if valid else 0.0


def infer_positive_label(model):
    id2label = {
        int(k): str(v).upper()
        for k, v in model.config.id2label.items()
    }

    for index, label in id2label.items():
        if any(token in label for token in ("POS", "HELPFUL", "SAFE", "GOOD")):
            return index

    if len(id2label) == 2:
        return 1

    return None


@torch.inference_mode()
def evaluate_classifier_scores(
    texts,
    model_name,
    device,
    batch_size,
    max_length,
    description,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    positive_id = infer_positive_label(model)
    scores = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=description,
        leave=False,
    ):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**encoded).logits

        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        else:
            if positive_id is None:
                raise ValueError(
                    f"Cannot infer positive/reward class from "
                    f"{model.config.id2label}"
                )

            batch_scores = torch.softmax(logits, dim=-1)[:, positive_id]

        scores.extend(float(value.item()) for value in batch_scores)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return scores



@torch.inference_mode()
def evaluate_toxicity_scores(
    texts,
    model_name,
    device,
    batch_size,
    max_length,
    description,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc=description, leave=False):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**encoded).logits

        if logits.shape[-1] == 1:
            batch_scores = torch.sigmoid(logits.squeeze(-1))
        else:
            batch_scores = torch.sigmoid(logits).max(dim=-1).values

        scores.extend(float(value.item()) for value in batch_scores)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return scores


@torch.inference_mode()
def evaluate_helpfulness_scores(
    prompts,
    responses,
    model_name,
    device,
    batch_size,
    max_length,
    description,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    scores = []

    for start in tqdm(range(0, len(responses), batch_size), desc=description, leave=False):
        prompt_batch = prompts[start:start + batch_size]
        response_batch = responses[start:start + batch_size]

        encoded = tokenizer(
            prompt_batch,
            response_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**encoded).logits

        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        else:
            positive_id = infer_positive_label(model)
            if positive_id is None:
                raise ValueError(
                    f"Cannot infer helpfulness class from {model.config.id2label}"
                )
            batch_scores = torch.softmax(logits, dim=-1)[:, positive_id]

        scores.extend(float(value.item()) for value in batch_scores)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return scores


def prepare_ppl_batch(prompts, responses, tokenizer, max_length):
    rows = []
    labels_rows = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        if not response_ids:
            response_ids = [tokenizer.eos_token_id]

        if len(prompt_ids) + len(response_ids) > max_length:
            overflow = len(prompt_ids) + len(response_ids) - max_length

            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                response_ids = response_ids[overflow - len(prompt_ids):]
                prompt_ids = []

        rows.append(prompt_ids + response_ids)
        labels_rows.append([-100] * len(prompt_ids) + response_ids)

    max_len = max(len(row) for row in rows)
    pad_id = tokenizer.pad_token_id

    input_ids = []
    labels = []
    attention_mask = []

    for row, label_row in zip(rows, labels_rows):
        pad_len = max_len - len(row)
        input_ids.append(row + [pad_id] * pad_len)
        labels.append(label_row + [-100] * pad_len)
        attention_mask.append([1] * len(row) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


@torch.inference_mode()
def evaluate_ppl(
    prompts,
    responses,
    model_name,
    device,
    batch_size,
    max_length,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    perplexities = []

    for start in tqdm(
        range(0, len(responses), batch_size),
        desc="Perplexity",
        leave=False,
    ):
        batch = prepare_ppl_batch(
            prompts[start:start + batch_size],
            responses[start:start + batch_size],
            tokenizer,
            max_length,
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits[:, :-1, :]

        labels = batch["labels"][:, 1:]

        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)

        mask = labels.ne(-100)
        counts = mask.sum(dim=1).clamp_min(1)
        sample_loss = (token_loss * mask).sum(dim=1) / counts
        sample_ppl = torch.exp(sample_loss).clamp(max=1_000_000)

        perplexities.extend(
            float(value.item())
            for value in sample_ppl
        )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return perplexities


def compare_scores(method_scores, baseline_scores, epsilon=1e-8):
    if not method_scores or not baseline_scores:
        return {
            "win_rate": None,
            "tie_rate": None,
            "loss_rate": None,
            "margin": None,
        }

    wins = ties = losses = 0
    margins = []

    for method_score, baseline_score in zip(method_scores, baseline_scores):
        difference = method_score - baseline_score
        margins.append(difference)

        if difference > epsilon:
            wins += 1
        elif difference < -epsilon:
            losses += 1
        else:
            ties += 1

    total = len(margins)

    return {
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "loss_rate": losses / total,
        "margin": mean(margins),
    }


def evaluate_method(
    method,
    records,
    baseline_records,
    args,
    device,
):
    prompts = [record["prompt"] for record in records]
    responses = [record["response"] for record in records]

    sentiment_scores = evaluate_classifier_scores(
        responses,
        args.sentiment_model,
        device,
        args.batch_size,
        args.max_length,
        "Sentiment",
    )

    perplexities = evaluate_ppl(
        prompts,
        responses,
        args.ppl_model,
        device,
        args.ppl_batch_size,
        args.max_length,
    )

    positive_rate = (
        sum(score >= 0.5 for score in sentiment_scores)
        / len(sentiment_scores)
    )

    reward_score = None
    reward_win_rate = None
    reward_tie_rate = None
    reward_loss_rate = None
    reward_margin = None

    sentiment_win_rate = None
    sentiment_tie_rate = None
    sentiment_loss_rate = None
    sentiment_margin = None

    harmfulness_score = None
    harmful_rate = None
    safety_rate = None
    safety_win_rate = None
    safety_tie_rate = None
    safety_loss_rate = None
    harmfulness_margin = None

    helpfulness_score = None
    helpful_rate = None
    helpfulness_win_rate = None
    helpfulness_tie_rate = None
    helpfulness_loss_rate = None
    helpfulness_margin = None

    reward_pairs = 0

    aligned_method, aligned_baseline = align_with_baseline(
        records,
        baseline_records,
    )

    reward_pairs = len(aligned_method)

    if reward_pairs:
        aligned_sentiment_scores = evaluate_classifier_scores(
            [record["response"] for record in aligned_method],
            args.sentiment_model,
            device,
            args.batch_size,
            args.max_length,
            f"{method} sentiment",
        )

        baseline_sentiment_scores = evaluate_classifier_scores(
            [record["response"] for record in aligned_baseline],
            args.sentiment_model,
            device,
            args.batch_size,
            args.max_length,
            "Base sentiment",
        )

        sentiment_comparison = compare_scores(
            aligned_sentiment_scores,
            baseline_sentiment_scores,
        )

        sentiment_win_rate = sentiment_comparison["win_rate"]
        sentiment_tie_rate = sentiment_comparison["tie_rate"]
        sentiment_loss_rate = sentiment_comparison["loss_rate"]
        sentiment_margin = sentiment_comparison["margin"]

    if args.reward_model:
        reward_scores = evaluate_classifier_scores(
            responses,
            args.reward_model,
            device,
            args.batch_size,
            args.max_length,
            "Reward",
        )

        reward_score = mean(reward_scores)

        if reward_pairs:
            aligned_method_scores = evaluate_classifier_scores(
                [record["response"] for record in aligned_method],
                args.reward_model,
                device,
                args.batch_size,
                args.max_length,
                f"{method} reward",
            )

            baseline_scores = evaluate_classifier_scores(
                [record["response"] for record in aligned_baseline],
                args.reward_model,
                device,
                args.batch_size,
                args.max_length,
                "Base reward",
            )

            reward_comparison = compare_scores(
                aligned_method_scores,
                baseline_scores,
            )

            reward_win_rate = reward_comparison["win_rate"]
            reward_tie_rate = reward_comparison["tie_rate"]
            reward_loss_rate = reward_comparison["loss_rate"]
            reward_margin = reward_comparison["margin"]


    if args.toxicity_model:
        toxicity_scores = evaluate_toxicity_scores(
            responses,
            args.toxicity_model,
            device,
            args.batch_size,
            args.max_length,
            "Toxicity",
        )

        harmfulness_score = mean(toxicity_scores)
        harmful_rate = (
            sum(score >= args.toxicity_threshold for score in toxicity_scores)
            / len(toxicity_scores)
            if toxicity_scores
            else 0.0
        )
        safety_rate = 1.0 - harmful_rate

        if reward_pairs:
            method_toxicity = evaluate_toxicity_scores(
                [record["response"] for record in aligned_method],
                args.toxicity_model,
                device,
                args.batch_size,
                args.max_length,
                f"{method} toxicity",
            )

            base_toxicity = evaluate_toxicity_scores(
                [record["response"] for record in aligned_baseline],
                args.toxicity_model,
                device,
                args.batch_size,
                args.max_length,
                "Base toxicity",
            )

            safety_comparison = compare_scores(
                [-score for score in method_toxicity],
                [-score for score in base_toxicity],
            )

            safety_win_rate = safety_comparison["win_rate"]
            safety_tie_rate = safety_comparison["tie_rate"]
            safety_loss_rate = safety_comparison["loss_rate"]
            harmfulness_margin = mean(
                method_score - base_score
                for method_score, base_score in zip(method_toxicity, base_toxicity)
            )

    if args.helpfulness_model:
        helpfulness_scores = evaluate_helpfulness_scores(
            prompts,
            responses,
            args.helpfulness_model,
            device,
            args.batch_size,
            args.max_length,
            "Helpfulness",
        )

        helpfulness_score = mean(helpfulness_scores)
        helpful_rate = (
            sum(score >= args.helpfulness_threshold for score in helpfulness_scores)
            / len(helpfulness_scores)
            if helpfulness_scores
            else 0.0
        )

        if reward_pairs:
            method_helpfulness = evaluate_helpfulness_scores(
                [record["prompt"] for record in aligned_method],
                [record["response"] for record in aligned_method],
                args.helpfulness_model,
                device,
                args.batch_size,
                args.max_length,
                f"{method} helpfulness",
            )

            base_helpfulness = evaluate_helpfulness_scores(
                [record["prompt"] for record in aligned_baseline],
                [record["response"] for record in aligned_baseline],
                args.helpfulness_model,
                device,
                args.batch_size,
                args.max_length,
                "Base helpfulness",
            )

            helpfulness_comparison = compare_scores(
                method_helpfulness,
                base_helpfulness,
            )

            helpfulness_win_rate = helpfulness_comparison["win_rate"]
            helpfulness_tie_rate = helpfulness_comparison["tie_rate"]
            helpfulness_loss_rate = helpfulness_comparison["loss_rate"]
            helpfulness_margin = helpfulness_comparison["margin"]

    return {
        "Method": method,
        "Base LM": BASE_MODELS.get(method, "unknown"),
        "Reward Model (RM)": REWARD_MODELS.get(method, "unknown"),
        "Samples": len(records),
        "Positive Rate": positive_rate,
        "PPL": mean(perplexities),
        "Avg Sentiment": mean(sentiment_scores),
        "RM Reward": reward_score,
        "RM Win Rate vs Base": reward_win_rate,
        "RM Tie Rate vs Base": reward_tie_rate,
        "RM Loss Rate vs Base": reward_loss_rate,
        "RM Margin vs Base": reward_margin,
        "Sentiment Win Rate vs Base": sentiment_win_rate,
        "Sentiment Tie Rate vs Base": sentiment_tie_rate,
        "Sentiment Loss Rate vs Base": sentiment_loss_rate,
        "Sentiment Margin vs Base": sentiment_margin,
        "Avg Harmfulness": harmfulness_score,
        "Harmful Rate": harmful_rate,
        "Safety Rate": safety_rate,
        "Safety Win Rate vs Base": safety_win_rate,
        "Safety Tie Rate vs Base": safety_tie_rate,
        "Safety Loss Rate vs Base": safety_loss_rate,
        "Harmfulness Margin vs Base": harmfulness_margin,
        "Avg Helpfulness": helpfulness_score,
        "Helpful Rate": helpful_rate,
        "Helpfulness Win Rate vs Base": helpfulness_win_rate,
        "Helpfulness Tie Rate vs Base": helpfulness_tie_rate,
        "Helpfulness Loss Rate vs Base": helpfulness_loss_rate,
        "Helpfulness Margin vs Base": helpfulness_margin,
        "Reward Pairs": reward_pairs,
        "Avg Length": average_length(responses),
        "Dist-1": distinct_n(responses, 1),
        "Dist-2": distinct_n(responses, 2),
        "Dist-3": distinct_n(responses, 3),
        "Repetition": repetition_rate(responses, 4),
        "Latency (s/sample)": mean(
            record.get("latency")
            for record in records
        ),
    }


def format_csv_value(value):
    if value is None:
        return ""

    if isinstance(value, float):
        return round(value, 6)

    return value


def save_csv(rows, path):
    fieldnames = [
        "Method",
        "Base LM",
        "Reward Model (RM)",
        "Samples",
        "Positive Rate",
        "PPL",
        "Avg Sentiment",
        "RM Reward",
        "RM Win Rate vs Base",
        "RM Tie Rate vs Base",
        "RM Loss Rate vs Base",
        "RM Margin vs Base",
        "Sentiment Win Rate vs Base",
        "Sentiment Tie Rate vs Base",
        "Sentiment Loss Rate vs Base",
        "Sentiment Margin vs Base",
        "Avg Harmfulness",
        "Harmful Rate",
        "Safety Rate",
        "Safety Win Rate vs Base",
        "Safety Tie Rate vs Base",
        "Safety Loss Rate vs Base",
        "Harmfulness Margin vs Base",
        "Avg Helpfulness",
        "Helpful Rate",
        "Helpfulness Win Rate vs Base",
        "Helpfulness Tie Rate vs Base",
        "Helpfulness Loss Rate vs Base",
        "Helpfulness Margin vs Base",
        "Reward Pairs",
        "Avg Length",
        "Dist-1",
        "Dist-2",
        "Dist-3",
        "Repetition",
        "Latency (s/sample)",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: format_csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def main():
    args = parse_args()
    device = get_device(args.device)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(f"Results: {args.results_dir}")

    if args.reward_model:
        print(f"Independent reward model: {args.reward_model}")
    else:
        print(
            "Independent reward model: disabled "
            "(use --reward-model to enable RM Reward and Win Rate)"
        )

    print(f"Toxicity model: {args.toxicity_model or 'disabled'}")
    print(f"Helpfulness model: {args.helpfulness_model or 'disabled'}")

    baseline_path = args.results_dir / args.baseline_file
    baseline_records = load_records(baseline_path)

    if baseline_records:
        print(
            f"Baseline: {len(baseline_records)} samples "
            f"from {baseline_path}"
        )
    else:
        print(
            f"Baseline not found: {baseline_path}. "
            "RM Win Rate vs Base will be empty."
        )

    all_metrics = []

    for method, filename in METHOD_FILES.items():
        path = args.results_dir / filename
        records = load_records(path)

        if not records:
            print(
                f"[SKIP] {method}: "
                f"no valid records in {path}"
            )
            continue

        print(
            f"\nEvaluating {method}: "
            f"{len(records)} samples"
        )

        metrics = evaluate_method(
            method,
            records,
            baseline_records,
            args,
            device,
        )

        all_metrics.append(metrics)

    if not all_metrics:
        raise RuntimeError(
            "No valid output JSON files were found."
        )

    csv_path = args.results_dir / "metrics.csv"
    json_path = args.results_dir / "metrics.json"

    save_csv(all_metrics, csv_path)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDone")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    for row in all_metrics:
        rm_reward = row["RM Reward"]
        win_rate = row["RM Win Rate vs Base"]
        sentiment_win_rate = row["Sentiment Win Rate vs Base"]

        rm_text = (
            f"{rm_reward:.4f}"
            if rm_reward is not None
            else "N/A"
        )

        win_text = (
            f"{win_rate:.4f}"
            if win_rate is not None
            else "N/A"
        )

        sentiment_win_text = (
            f"{sentiment_win_rate:.4f}"
            if sentiment_win_rate is not None
            else "N/A"
        )

        safety_rate = row["Safety Rate"]
        harmfulness = row["Avg Harmfulness"]
        helpfulness = row["Avg Helpfulness"]
        helpfulness_win = row["Helpfulness Win Rate vs Base"]

        safety_text = (
            f"{safety_rate:.4f}"
            if safety_rate is not None
            else "N/A"
        )

        harmfulness_text = (
            f"{harmfulness:.4f}"
            if harmfulness is not None
            else "N/A"
        )

        helpfulness_text = (
            f"{helpfulness:.4f}"
            if helpfulness is not None
            else "N/A"
        )

        helpfulness_win_text = (
            f"{helpfulness_win:.4f}"
            if helpfulness_win is not None
            else "N/A"
        )

        print(
            f"{row['Method']:15s} | "
            f"N={row['Samples']} | "
            f"Positive={row['Positive Rate']:.4f} | "
            f"PPL={row['PPL']:.4f} | "
            f"Sentiment={row['Avg Sentiment']:.4f} | "
            f"RM={rm_text} | "
            f"RM-Win={win_text} | "
            f"Sent-Win={sentiment_win_text} | "
            f"Safety={safety_text} | "
            f"Harm={harmfulness_text} | "
            f"Helpful={helpfulness_text} | "
            f"Help-Win={helpfulness_win_text} | "
            f"Dist-3={row['Dist-3']:.4f} | "
            f"Rep={row['Repetition']:.4f} | "
            f"Latency={row['Latency (s/sample)']:.4f}s"
        )


if __name__ == "__main__":
    main()