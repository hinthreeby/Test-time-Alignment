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


NUM_CANDIDATES = 4
BLOCK_SIZE = 8
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95


@torch.inference_mode()
def score_candidates(candidate_ids, lm_tokenizer, scorer_tokenizer, scorer, device):
    texts = lm_tokenizer.batch_decode(candidate_ids, skip_special_tokens=True)
    inputs = scorer_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    return scorer(inputs["input_ids"], inputs["attention_mask"])


@torch.inference_mode()
def generate_blockwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device):
    inputs = lm_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    generated_tokens = 0

    while generated_tokens < MAX_NEW_TOKENS:
        current_block_size = min(BLOCK_SIZE, MAX_NEW_TOKENS - generated_tokens)

        candidates = lm.generate(
            input_ids=input_ids,
            max_new_tokens=current_block_size,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_CANDIDATES,
            pad_token_id=lm_tokenizer.eos_token_id,
        )

        values = score_candidates(candidates, lm_tokenizer, scorer_tokenizer, scorer, device)
        best_index = torch.argmax(values)
        best_candidate = candidates[best_index].unsqueeze(0)

        added_tokens = best_candidate.shape[1] - input_ids.shape[1]
        input_ids = best_candidate
        generated_tokens += added_tokens

        if lm_tokenizer.eos_token_id in input_ids[0, -added_tokens:]:
            break

    response_ids = input_ids[0, prompt_length:]
    return lm_tokenizer.decode(response_ids, skip_special_tokens=True).strip()


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
    response = generate_blockwise(prompt, lm, lm_tokenizer, scorer, scorer_tokenizer, device)

    print("Prompt:", prompt)
    print("Response:", response)


if __name__ == "__main__":
    main()