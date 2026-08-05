#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
SCORER_BACKBONE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
SCORER_CHECKPOINT_PATH = PROJECT_ROOT / "Method" / "CD_new" / "checkpoints" / "cd_fudge.pt"

sys.path.insert(0, str(PROJECT_ROOT / "Method" / "CD_new" / "models"))
from prefix_scorer import PrefixScorer


TOP_K = 20
LAMBDA_WEIGHT = 1.0
MAX_NEW_TOKENS = 64


@torch.inference_mode()
def score_candidates(input_ids, candidate_ids, lm_tokenizer, scorer_tokenizer, scorer, device):
    candidates = torch.cat([input_ids.repeat(candidate_ids.size(0), 1), candidate_ids.unsqueeze(1)], dim=1)
    texts = lm_tokenizer.batch_decode(candidates, skip_special_tokens=True)

    inputs = scorer_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    return scorer(inputs["input_ids"], inputs["attention_mask"])


@torch.inference_mode()
def generate_tokenwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device):
    inputs = lm_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    for _ in range(MAX_NEW_TOKENS):
        logits = lm(input_ids=input_ids).logits[:, -1, :]
        top_logits, top_ids = torch.topk(logits[0], TOP_K)

        values = score_candidates(input_ids, top_ids, lm_tokenizer, scorer_tokenizer, scorer, device)
        aligned_logits = top_logits + LAMBDA_WEIGHT * values

        best_index = torch.argmax(aligned_logits)
        next_token = top_ids[best_index].view(1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == lm_tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, inputs["input_ids"].shape[1]:]
    return lm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lm_tokenizer = AutoTokenizer.from_pretrained(str(BASE_LM_PATH), local_files_only=True)
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

    lm = AutoModelForCausalLM.from_pretrained(str(BASE_LM_PATH), local_files_only=True).to(device).eval()

    scorer_tokenizer = AutoTokenizer.from_pretrained(str(SCORER_BACKBONE_PATH), local_files_only=True)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer_tokenizer.padding_side = "right"

    scorer = PrefixScorer(str(SCORER_BACKBONE_PATH)).to(device)
    scorer.load_state_dict(torch.load(SCORER_CHECKPOINT_PATH, map_location=device))
    scorer.eval()

    prompt = "This movie was"
    response = generate_tokenwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device)

    print("Prompt:", prompt)
    print("Response:", response)


if __name__ == "__main__":
    main()