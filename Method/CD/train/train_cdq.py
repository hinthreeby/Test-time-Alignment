#!/usr/bin/env python3

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = PROJECT_ROOT / "dataset" / "cd_train" / "hh_prompts.jsonl"
BASE_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
SCORER_BACKBONE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
REWARD_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
OUTPUT_PATH = PROJECT_ROOT / "Method" / "CD" / "checkpoints" / "cd_q.pt"

sys.path.insert(0, str(PROJECT_ROOT / "Method" / "CD" / "models"))
from prefix_scorer import PrefixScorer


NUM_PROMPTS = 1000
MAX_NEW_TOKENS = 16
MAX_PREFIX_LENGTH = 128
LEARNING_RATE = 1e-5
EPOCHS = 4
TARGET_UPDATE_INTERVAL = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CD-Q prefix scorer.")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_prompts(limit):
    prompts = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            prompt = sample.get("prompt", "")
            prompt = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
            prompt = prompt.strip()

            if prompt:
                prompts.append(prompt)

            if len(prompts) >= limit:
                break

    return prompts


@torch.inference_mode()
def generate_response_ids(prompt, tokenizer, model, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_length = inputs["input_ids"].shape[1]
    return outputs[0, prompt_length:].cpu()


@torch.inference_mode()
def get_reward(prompt, response, tokenizer, model, device):
    text = prompt + " " + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


def create_prefix_pairs(prompt, response_ids, tokenizer):
    current_prefixes = []
    next_prefixes = []

    for index in range(len(response_ids)):
        current_text = tokenizer.decode(response_ids[:index], skip_special_tokens=True)
        next_text = tokenizer.decode(response_ids[:index + 1], skip_special_tokens=True)

        current_prefixes.append((prompt + " " + current_text).strip())
        next_prefixes.append((prompt + " " + next_text).strip())

    return current_prefixes, next_prefixes


def encode(texts, tokenizer):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_PREFIX_LENGTH,
        return_tensors="pt",
    )


def update_target_network(scorer, target_scorer):
    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in scorer.state_dict().items()}
    target_scorer.load_state_dict(cpu_state_dict)


def main():
    args = parse_args()
    if args.num_prompts <= 0 or args.epochs <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num-prompts, epochs và max-new-tokens phải lớn hơn 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    print("Device:", device)

    base_tokenizer = AutoTokenizer.from_pretrained(str(BASE_LM_PATH), local_files_only=True)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(str(REWARD_MODEL_PATH), local_files_only=True)

    scorer_tokenizer = AutoTokenizer.from_pretrained(str(SCORER_BACKBONE_PATH), local_files_only=True)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_tokenizer.padding_side = "right"

    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_lm = AutoModelForCausalLM.from_pretrained(
        str(BASE_LM_PATH), local_files_only=True, torch_dtype=model_dtype,
    ).to(device).eval()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(REWARD_MODEL_PATH), local_files_only=True, torch_dtype=model_dtype,
    ).to(device).eval()

    scorer = PrefixScorer(str(SCORER_BACKBONE_PATH)).to(device)
    target_scorer = deepcopy(scorer).to(cpu_device).eval()

    for parameter in target_scorer.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=LEARNING_RATE)

    prompts = load_prompts(args.num_prompts)
    train_step = 0

    print(f"Prompts: {len(prompts)} | Epochs: {args.epochs}")
    print(f"Checkpoint: {args.output_path}")

    for epoch in range(args.epochs):
        scorer.train()
        total_loss = 0.0
        trained_samples = 0

        for prompt in tqdm(prompts, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            response_ids = generate_response_ids(
                prompt, base_tokenizer, base_lm, device, args.max_new_tokens
            )

            if len(response_ids) == 0:
                continue

            response = base_tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            if not response:
                continue

            final_reward = get_reward(prompt, response, reward_tokenizer, reward_model, device)

            current_prefixes, next_prefixes = create_prefix_pairs(prompt, response_ids, base_tokenizer)

            current_inputs = encode(current_prefixes, scorer_tokenizer)
            next_inputs = encode(next_prefixes, scorer_tokenizer)

            with torch.no_grad():
                target_values = target_scorer(
                    next_inputs["input_ids"],
                    next_inputs["attention_mask"],
                )

                target_values[-1] = final_reward
                targets = target_values.to(device)

            current_input_ids = current_inputs["input_ids"].to(device)
            current_attention_mask = current_inputs["attention_mask"].to(device)

            predicted_values = scorer(current_input_ids, current_attention_mask)
            loss = F.mse_loss(predicted_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step += 1
            total_loss += loss.item()
            trained_samples += 1

            if train_step % TARGET_UPDATE_INTERVAL == 0:
                update_target_network(scorer, target_scorer)

            del current_inputs, next_inputs
            del current_input_ids, current_attention_mask
            del predicted_values, targets, target_values

        average_loss = total_loss / max(trained_samples, 1)
        print(f"Epoch {epoch + 1} loss: {average_loss:.4f}")

    update_target_network(scorer, target_scorer)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scorer.state_dict(), args.output_path)

    print("Saved:", args.output_path)


if __name__ == "__main__":
    main()
