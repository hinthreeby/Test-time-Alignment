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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


METHOD_FILES = {
    "Base": "base.json",
    "ARGS-greedy": "args_greedy.json",
    "ARGS-topk": "args_topk.json",
    "rad-fixed": "rad_fixed.json",
    "rad-confidence": "rad_confidence.json",
    "rad-reward": "rad_reward.json",
    "rad-hybrid": "rad_hybrid.json",
    "rad-length": "rad_length.json",
    "CD-FUDGE-tokenwise": "fudge_tokenwise.json",
    "CD-FUDGE-blockwise": "fudge_blockwise.json",
    "CD-Q-tokenwise": "cdq_tokenwise.json",
    "CD-Q-blockwise": "cdq_blockwise.json",
    "GenARM": "genarm.json",
    "multi-signal": "multi_signal.jsonl",
    "CURA": "cura.json",
    "GSI": "gsi.json",
    "PARM": "parm.json",
}

BASE_MODELS = {method: "gpt2-large" for method in METHOD_FILES}
BASE_MODELS["PARM"] = "tulu-2-7b"

REWARD_MODELS = {
    "Base": "none",
    "ARGS-greedy": "configured RM",
    "ARGS-topk": "configured RM",
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
    "multi-signal": "multi-signal-gpt2-medium-hh",
    "GSI": "sentiment-roberta-large-english",
    "PARM": "PBLoRA helpfulness+harmlessness",
}

BASELINE_FILE = "base.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SENTIMENT_MODEL = PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
DEFAULT_REWARD_MODEL = PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
DEFAULT_TOXICITY_MODEL = PROJECT_ROOT / "models" / "toxic-bert"
DEFAULT_HELPFULNESS_MODEL = PROJECT_ROOT / "models" / "helpfulness-deberta-v3-large"
DEFAULT_PPL_MODEL = PROJECT_ROOT / "models" / "gpt2-large"

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sentiment-model", default=str(DEFAULT_SENTIMENT_MODEL))
    parser.add_argument("--reward-model", default=str(DEFAULT_REWARD_MODEL))
    parser.add_argument("--toxicity-model", default=str(DEFAULT_TOXICITY_MODEL))
    parser.add_argument("--helpfulness-model", default=str(DEFAULT_HELPFULNESS_MODEL))
    parser.add_argument("--ppl-model", default=str(DEFAULT_PPL_MODEL))
    parser.add_argument("--toxicity-threshold", type=float, default=0.5)
    parser.add_argument("--helpfulness-threshold", type=float, default=0.0)
    parser.add_argument("--baseline-file", default=BASELINE_FILE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ppl-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Chỉ đánh giá N mẫu hợp lệ đầu tiên của mỗi method.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_FILES),
        help="Chỉ eval lại các method được chỉ định, ví dụ: --methods CD-FUDGE-tokenwise rad-reward",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Chỉ eval lại các file tương đối trong results, ví dụ: --files fudge/fudge_tokenwise.json",
    )
    return parser.parse_args()


def get_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_selected_methods(args):
    selected = set(args.methods or [])

    if args.files:
        reverse_exact = {filename: method for method, filename in METHOD_FILES.items()}
        reverse_name = {}

        for method, filename in METHOD_FILES.items():
            reverse_name.setdefault(Path(filename).name, []).append(method)

        for raw_file in args.files:
            normalized = Path(raw_file).as_posix().lstrip("./")

            if normalized in reverse_exact:
                selected.add(reverse_exact[normalized])
                continue

            candidates = reverse_name.get(Path(normalized).name, [])

            if len(candidates) == 1:
                selected.add(candidates[0])
                continue

            raise ValueError(f"Không ánh xạ được file '{raw_file}' vào METHOD_FILES.")

    return [method for method in METHOD_FILES if not selected or method in selected]


def load_records(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    records = []

    for item in data:
        if not isinstance(item, dict) or item.get("status", "success") != "success":
            continue

        response = item.get("response")

        if not isinstance(response, str) or not response.strip():
            continue

        latency = item.get("latency")

        try:
            latency = float(latency) if latency is not None else None
        except (TypeError, ValueError):
            latency = None

        records.append({
            **item,
            "prompt": str(item.get("prompt", "")).strip(),
            "response": response.strip(),
            "latency": latency,
        })

    return records


def record_key(record):
    if record.get("md5_hash"):
        return ("md5", str(record["md5_hash"]))
    if record.get("id") is not None:
        return ("id", str(record["id"]))
    return ("prompt", record.get("prompt", ""))


def align_indices(records, baseline_records):
    baseline_map = {record_key(record): index for index, record in enumerate(baseline_records)}
    method_indices = []
    baseline_indices = []

    for method_index, record in enumerate(records):
        baseline_index = baseline_map.get(record_key(record))

        if baseline_index is not None:
            method_indices.append(method_index)
            baseline_indices.append(baseline_index)

    return method_indices, baseline_indices


def tokenize_words(text):
    return WORD_RE.findall(text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def distinct_n(texts, n):
    grams = [gram for text in texts for gram in ngrams(tokenize_words(text), n)]
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
    return sum(len(tokenize_words(text)) for text in texts) / len(texts) if texts else 0.0


def mean(values):
    valid = []

    for value in values:
        if value is None:
            continue

        number = float(value)

        if math.isfinite(number):
            valid.append(number)

    return sum(valid) / len(valid) if valid else 0.0


def infer_positive_label(model):
    id2label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}

    for index, label in id2label.items():
        if any(token in label for token in ("POS", "HELPFUL", "SAFE", "GOOD")):
            return index

    return 1 if len(id2label) == 2 else None


def flatten_text_groups(groups):
    flat = []
    slices = {}
    start = 0

    for name, texts in groups.items():
        flat.extend(texts)
        slices[name] = slice(start, start + len(texts))
        start += len(texts)

    return flat, slices


def split_scores(scores, slices):
    return {name: scores[index_slice] for name, index_slice in slices.items()}


@torch.inference_mode()
def evaluate_classifier_groups(groups, model_name, device, batch_size, max_length, description):
    texts, slices = flatten_text_groups(groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    positive_id = infer_positive_label(model)
    scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        logits = model(**encoded).logits

        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        else:
            if positive_id is None:
                raise ValueError(f"Cannot infer positive class from {model.config.id2label}")
            batch_scores = torch.softmax(logits, dim=-1)[:, positive_id]

        scores.extend(float(value.item()) for value in batch_scores)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return split_scores(scores, slices)


@torch.inference_mode()
def evaluate_toxicity_groups(groups, model_name, device, batch_size, max_length):
    texts, slices = flatten_text_groups(groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Toxicity"):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        logits = model(**encoded).logits
        batch_scores = torch.sigmoid(logits.squeeze(-1)) if logits.shape[-1] == 1 else torch.sigmoid(logits).max(dim=-1).values
        scores.extend(float(value.item()) for value in batch_scores)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return split_scores(scores, slices)


@torch.inference_mode()
def evaluate_helpfulness_groups(prompt_groups, response_groups, model_name, device, batch_size, max_length):
    names = list(response_groups)
    prompts = []
    responses = []
    slices = {}
    start = 0

    for name in names:
        group_prompts = prompt_groups[name]
        group_responses = response_groups[name]
        prompts.extend(group_prompts)
        responses.extend(group_responses)
        slices[name] = slice(start, start + len(group_responses))
        start += len(group_responses)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    positive_id = infer_positive_label(model)
    scores = []

    for start in tqdm(range(0, len(responses), batch_size), desc="Helpfulness"):
        encoded = tokenizer(
            prompts[start:start + batch_size],
            responses[start:start + batch_size],
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
                raise ValueError(f"Cannot infer helpfulness class from {model.config.id2label}")
            batch_scores = torch.softmax(logits, dim=-1)[:, positive_id]

        scores.extend(float(value.item()) for value in batch_scores)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return split_scores(scores, slices)


def prepare_ppl_batch(prompts, responses, tokenizer, max_length):
    rows = []
    label_rows = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id]

        if len(prompt_ids) + len(response_ids) > max_length:
            overflow = len(prompt_ids) + len(response_ids) - max_length

            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                response_ids = response_ids[overflow - len(prompt_ids):]
                prompt_ids = []

        rows.append(prompt_ids + response_ids)
        label_rows.append([-100] * len(prompt_ids) + response_ids)

    max_len = max(len(row) for row in rows)
    pad_id = tokenizer.pad_token_id

    return {
        "input_ids": torch.tensor([row + [pad_id] * (max_len - len(row)) for row in rows]),
        "labels": torch.tensor([row + [-100] * (max_len - len(row)) for row in label_rows]),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows]),
    }


@torch.inference_mode()
def evaluate_ppl_groups(prompt_groups, response_groups, model_name, device, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    results = {}

    for name in response_groups:
        prompts = prompt_groups[name]
        responses = response_groups[name]
        perplexities = []

        for start in tqdm(range(0, len(responses), batch_size), desc=f"PPL {name}", leave=False):
            batch = prepare_ppl_batch(prompts[start:start + batch_size], responses[start:start + batch_size], tokenizer, max_length)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
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
            perplexities.extend(float(value.item()) for value in sample_ppl)

        results[name] = perplexities

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def compare_scores(method_scores, baseline_scores, epsilon=1e-8):
    if not method_scores or not baseline_scores:
        return {"win_rate": None, "tie_rate": None, "loss_rate": None, "margin": None}

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
    return {"win_rate": wins / total, "tie_rate": ties / total, "loss_rate": losses / total, "margin": mean(margins)}


def aligned_scores(method, score_groups, records_by_method, baseline_records):
    method_indices, baseline_indices = align_indices(records_by_method[method], baseline_records)
    return (
        [score_groups[method][index] for index in method_indices],
        [score_groups["__baseline__"][index] for index in baseline_indices],
    )


def build_metrics(method, records, baseline_records, score_groups, args):
    responses = [record["response"] for record in records]
    sentiment_scores = score_groups["sentiment"][method]
    perplexities = score_groups["ppl"][method]
    reward_pairs = len(align_indices(records, baseline_records)[0])

    sentiment_comparison = {"win_rate": None, "tie_rate": None, "loss_rate": None, "margin": None}
    reward_comparison = sentiment_comparison.copy()
    safety_comparison = sentiment_comparison.copy()
    helpfulness_comparison = sentiment_comparison.copy()

    if reward_pairs:
        method_scores, base_scores = aligned_scores(method, score_groups["sentiment"], score_groups["records"], baseline_records)
        sentiment_comparison = compare_scores(method_scores, base_scores)

    reward_scores = score_groups.get("reward", {}).get(method)
    reward_score = mean(reward_scores) if reward_scores is not None else None

    if reward_pairs and reward_scores is not None:
        method_scores, base_scores = aligned_scores(method, score_groups["reward"], score_groups["records"], baseline_records)
        reward_comparison = compare_scores(method_scores, base_scores)

    toxicity_scores = score_groups.get("toxicity", {}).get(method)
    harmfulness_score = mean(toxicity_scores) if toxicity_scores is not None else None
    harmful_rate = None
    safety_rate = None
    harmfulness_margin = None

    if toxicity_scores is not None:
        harmful_rate = sum(score >= args.toxicity_threshold for score in toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0
        safety_rate = 1.0 - harmful_rate

        if reward_pairs:
            method_scores, base_scores = aligned_scores(method, score_groups["toxicity"], score_groups["records"], baseline_records)
            safety_comparison = compare_scores([-score for score in method_scores], [-score for score in base_scores])
            harmfulness_margin = mean(method_score - base_score for method_score, base_score in zip(method_scores, base_scores))

    helpfulness_scores = score_groups.get("helpfulness", {}).get(method)
    helpfulness_score = mean(helpfulness_scores) if helpfulness_scores is not None else None
    helpful_rate = None

    if helpfulness_scores is not None:
        helpful_rate = sum(score >= args.helpfulness_threshold for score in helpfulness_scores) / len(helpfulness_scores) if helpfulness_scores else 0.0

        if reward_pairs:
            method_scores, base_scores = aligned_scores(method, score_groups["helpfulness"], score_groups["records"], baseline_records)
            helpfulness_comparison = compare_scores(method_scores, base_scores)

    return {
        "Method": method,
        "Base LM": BASE_MODELS.get(method, "unknown"),
        "Reward Model (RM)": REWARD_MODELS.get(method, "unknown"),
        "Samples": len(records),
        "Positive Rate": sum(score >= 0.5 for score in sentiment_scores) / len(sentiment_scores),
        "PPL": mean(perplexities),
        "Avg Sentiment": mean(sentiment_scores),
        "RM Reward": reward_score,
        "RM Win Rate vs Base": reward_comparison["win_rate"],
        "RM Tie Rate vs Base": reward_comparison["tie_rate"],
        "RM Loss Rate vs Base": reward_comparison["loss_rate"],
        "RM Margin vs Base": reward_comparison["margin"],
        "Sentiment Win Rate vs Base": sentiment_comparison["win_rate"],
        "Sentiment Tie Rate vs Base": sentiment_comparison["tie_rate"],
        "Sentiment Loss Rate vs Base": sentiment_comparison["loss_rate"],
        "Sentiment Margin vs Base": sentiment_comparison["margin"],
        "Avg Harmfulness": harmfulness_score,
        "Harmful Rate": harmful_rate,
        "Safety Rate": safety_rate,
        "Safety Win Rate vs Base": safety_comparison["win_rate"],
        "Safety Tie Rate vs Base": safety_comparison["tie_rate"],
        "Safety Loss Rate vs Base": safety_comparison["loss_rate"],
        "Harmfulness Margin vs Base": harmfulness_margin,
        "Avg Helpfulness": helpfulness_score,
        "Helpful Rate": helpful_rate,
        "Helpfulness Win Rate vs Base": helpfulness_comparison["win_rate"],
        "Helpfulness Tie Rate vs Base": helpfulness_comparison["tie_rate"],
        "Helpfulness Loss Rate vs Base": helpfulness_comparison["loss_rate"],
        "Helpfulness Margin vs Base": helpfulness_comparison["margin"],
        "Reward Pairs": reward_pairs,
        "Avg Length": average_length(responses),
        "Dist-1": distinct_n(responses, 1),
        "Dist-2": distinct_n(responses, 2),
        "Dist-3": distinct_n(responses, 3),
        "Repetition": repetition_rate(responses, 4),
        "Latency (s/sample)": mean(record.get("latency") for record in records),
    }


def format_csv_value(value):
    if value is None:
        return ""
    return round(value, 6) if isinstance(value, float) else value


def save_csv(rows, path):
    fieldnames = [
        "Method", "Base LM", "Reward Model (RM)", "Samples", "Positive Rate", "PPL", "Avg Sentiment",
        "RM Reward", "RM Win Rate vs Base", "RM Tie Rate vs Base", "RM Loss Rate vs Base", "RM Margin vs Base",
        "Sentiment Win Rate vs Base", "Sentiment Tie Rate vs Base", "Sentiment Loss Rate vs Base", "Sentiment Margin vs Base",
        "Avg Harmfulness", "Harmful Rate", "Safety Rate", "Safety Win Rate vs Base", "Safety Tie Rate vs Base",
        "Safety Loss Rate vs Base", "Harmfulness Margin vs Base", "Avg Helpfulness", "Helpful Rate",
        "Helpfulness Win Rate vs Base", "Helpfulness Tie Rate vs Base", "Helpfulness Loss Rate vs Base",
        "Helpfulness Margin vs Base", "Reward Pairs", "Avg Length", "Dist-1", "Dist-2", "Dist-3",
        "Repetition", "Latency (s/sample)",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in fieldnames})


def load_existing_metrics(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    return {row["Method"]: row for row in rows if isinstance(row, dict) and row.get("Method")}


def main():
    args = parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples phải lớn hơn 0.")

    device = get_device(args.device)
    selected_methods = resolve_selected_methods(args)
    selective_run = bool(args.methods or args.files)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.results_dir / args.baseline_file
    baseline_records = load_records(baseline_path)

    if args.max_samples is not None:
        baseline_records = baseline_records[:args.max_samples]

    if not baseline_records:
        raise RuntimeError(f"Không tìm thấy baseline hợp lệ: {baseline_path}")

    records_by_method = {}

    for method in selected_methods:
        path = args.results_dir / METHOD_FILES[method]
        records = load_records(path)

        if args.max_samples is not None:
            records = records[:args.max_samples]

        if records:
            records_by_method[method] = records
        else:
            print(f"[SKIP] {method}: no valid records in {path}")

    if not records_by_method:
        raise RuntimeError("Không có method hợp lệ để đánh giá.")

    print("Device:", device)
    print("Methods:", ", ".join(records_by_method))
    print("Mỗi model metric chỉ được load một lần.")

    response_groups = {"__baseline__": [record["response"] for record in baseline_records]}
    prompt_groups = {"__baseline__": [record["prompt"] for record in baseline_records]}

    for method, records in records_by_method.items():
        response_groups[method] = [record["response"] for record in records]
        prompt_groups[method] = [record["prompt"] for record in records]

    score_groups = {"records": records_by_method}

    sentiment_scores = evaluate_classifier_groups(
        response_groups, args.sentiment_model, device, args.batch_size, args.max_length, "Sentiment all methods"
    )
    score_groups["sentiment"] = sentiment_scores

    if args.reward_model:
        if str(args.reward_model) == str(args.sentiment_model):
            print("Reward model giống sentiment model: tái sử dụng sentiment scores.")
            score_groups["reward"] = sentiment_scores
        else:
            score_groups["reward"] = evaluate_classifier_groups(
                response_groups, args.reward_model, device, args.batch_size, args.max_length, "Reward all methods"
            )

    if args.toxicity_model:
        score_groups["toxicity"] = evaluate_toxicity_groups(
            response_groups, args.toxicity_model, device, args.batch_size, args.max_length
        )

    if args.helpfulness_model:
        score_groups["helpfulness"] = evaluate_helpfulness_groups(
            prompt_groups, response_groups, args.helpfulness_model, device, args.batch_size, args.max_length
        )

    score_groups["ppl"] = evaluate_ppl_groups(
        prompt_groups, response_groups, args.ppl_model, device, args.ppl_batch_size, args.max_length
    )

    new_rows = {
        method: build_metrics(method, records, baseline_records, score_groups, args)
        for method, records in records_by_method.items()
    }

    json_path = args.results_dir / "metrics.json"
    csv_path = args.results_dir / "metrics.csv"

    if selective_run:
        merged = load_existing_metrics(json_path)
        merged.update(new_rows)
    else:
        merged = new_rows

    ordered_rows = [merged[method] for method in METHOD_FILES if method in merged]
    save_csv(ordered_rows, csv_path)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(ordered_rows, file, indent=2, ensure_ascii=False)

    print("\nUpdated methods:", ", ".join(new_rows))
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
