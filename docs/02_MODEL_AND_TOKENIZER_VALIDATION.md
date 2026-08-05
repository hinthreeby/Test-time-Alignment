# Stage 2 — Validate Models, Tokenizers, and Freezing

## Goal

Confirm that the GPT-2 Large base LM and GPT-2 Small RAD sentiment reward model can be used together safely before feature extraction.

## Required models

- Base LM: GPT-2 Large.
- Reward model: GPT-2 Small RAD sentiment reward checkpoint.
- Both models must be inference-only during router training.

## Checks

### 1. Tokenizer compatibility

Verify:

- same vocabulary size;
- same token-to-ID mapping;
- same EOS token ID;
- same BOS behavior;
- same byte-level BPE behavior;
- compatible padding strategy.

Required assertions:

```python
assert base_tokenizer.get_vocab() == reward_tokenizer.get_vocab()
assert base_tokenizer.eos_token_id == reward_tokenizer.eos_token_id
```

If exact equality fails, stop and document the mismatch. Do not silently remap tokens.

### 2. Model freezing

Set both models to evaluation mode and disable gradients:

```python
base_model.eval()
reward_model.eval()

for p in base_model.parameters():
    p.requires_grad = False

for p in reward_model.parameters():
    p.requires_grad = False
```

Add a test that checks every parameter remains frozen before and after one router optimization step.

### 3. Reward-model interface

Document exactly how the RAD reward model scores candidate continuations:

- input text or token IDs;
- expected tensor shape;
- returned scalar/logit/probability;
- whether higher means more positive;
- whether scores should be transformed before use.

Do not assume the reward output is calibrated. Record score statistics on a small sample.

### 4. Candidate scoring sanity check

For several prefixes:

1. Get top-20 candidates from GPT-2 Large.
2. Append each candidate to the prefix.
3. Score each candidate with the reward model.
4. Print a readable table with:
   - token string;
   - token ID;
   - base logit;
   - reward score.

Manually verify that candidate ordering and token decoding are correct.

## Deliverables

```text
scripts/validate_models.py
reports/model_validation.json
reports/candidate_sanity_samples.jsonl
```

`model_validation.json` must include:

- model identifiers;
- tokenizer identifiers;
- vocabulary size;
- special token IDs;
- frozen parameter counts;
- reward score mean/std/min/max;
- pass/fail results for every check.

## Acceptance criteria

- Token mappings are exactly compatible.
- Both models stay frozen.
- Candidate token IDs are scored without retokenization mistakes.
- Reward direction is verified.
- The script fails loudly on incompatible tokenizers.

## Prompt for the coding AI

Implement model and tokenizer validation for GPT-2 Large and the RAD GPT-2 Small sentiment reward model. Verify exact vocabulary compatibility, freeze both models, inspect the reward-model interface, and generate candidate-level sanity examples. Add explicit assertions and a JSON validation report. Do not implement the router yet.
