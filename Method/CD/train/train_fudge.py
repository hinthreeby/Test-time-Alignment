#!/usr/bin/env python3

import json
import sys
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
OUTPUT_PATH = PROJECT_ROOT / "Method" / "CD_new" / "checkpoints" / "cd_fudge.pt"

sys.path.insert(0, str(PROJECT_ROOT / "Method" / "CD_new" / "models"))
from prefix_scorer import PrefixScorer


NUM_PROMPTS = 1000
MAX_NEW_TOKENS = 16
PREFIX_STEP = 4
LEARNING_RATE = 1e-5
EPOCHS = 4


def load_prompts():
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

            if len(prompts) >= NUM_PROMPTS:
                break

    return prompts


@torch.inference_mode()
def generate_response(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def get_final_reward(prompt, response, tokenizer, model, device):
    text = prompt + " " + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


def create_prefixes(prompt, response, tokenizer):
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    prefixes = []

    for end in range(PREFIX_STEP, len(response_ids) + 1, PREFIX_STEP):
        prefix = tokenizer.decode(response_ids[:end], skip_special_tokens=True)
        prefixes.append(prompt + " " + prefix)

    if not prefixes:
        prefixes.append(prompt + " " + response)

    return prefixes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_tokenizer = AutoTokenizer.from_pretrained(str(BASE_LM_PATH), local_files_only=True)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    base_lm = AutoModelForCausalLM.from_pretrained(str(BASE_LM_PATH), local_files_only=True).to(device).eval()

    reward_tokenizer = AutoTokenizer.from_pretrained(str(REWARD_MODEL_PATH), local_files_only=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(REWARD_MODEL_PATH),
        local_files_only=True,
    ).to(device).eval()

    scorer_tokenizer = AutoTokenizer.from_pretrained(str(SCORER_BACKBONE_PATH), local_files_only=True)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_tokenizer.padding_side = "right"

    scorer = PrefixScorer(str(SCORER_BACKBONE_PATH)).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=LEARNING_RATE)

    prompts = load_prompts()

    for epoch in range(EPOCHS):
        scorer.train()
        total_loss = 0.0
        trained_samples = 0

        for prompt in tqdm(prompts, desc=f"Epoch {epoch + 1}"):
            response = generate_response(prompt, base_tokenizer, base_lm, device)

            if not response:
                continue

            final_reward = get_final_reward(prompt, response, reward_tokenizer, reward_model, device)
            prefixes = create_prefixes(prompt, response, scorer_tokenizer)

            inputs = scorer_tokenizer(
                prefixes,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            predicted_values = scorer(inputs["input_ids"], inputs["attention_mask"])
            targets = torch.full_like(predicted_values, final_reward)
            loss = F.mse_loss(predicted_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            trained_samples += 1

        average_loss = total_loss / max(trained_samples, 1)
        print(f"Epoch {epoch + 1} loss: {average_loss:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scorer.state_dict(), OUTPUT_PATH)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()