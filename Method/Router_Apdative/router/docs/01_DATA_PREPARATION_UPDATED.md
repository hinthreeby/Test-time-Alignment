# Stage 1 — Prepare Amazon Polarity Router Training Data

## Goal

Convert the existing local Amazon Polarity corpus into clean `train.jsonl` and `validation.jsonl` files for teacher-forced router training while preserving strict separation from the RAD benchmark.

This stage must use the Amazon Polarity data already available in the repository. Do not download another copy, do not use SST-2 for version 1, and do not modify any benchmark file.

## Existing inputs

Expected paths:

```text
dataset/
├── RAD_train/
│   ├── amazon_polarity/
│   └── sst2/
└── rad_benchmark/
    ├── negative_prompts.jsonl
    ├── neutral_prompts.jsonl
    ├── positive_prompts.jsonl
    └── nontoxic_prompts-10k.jsonl
```

Use:

- `dataset/RAD_train/amazon_polarity/` as the router-data source.
- `negative_prompts.jsonl`, `neutral_prompts.jsonl`, and `positive_prompts.jsonl` only for leakage checking and later held-out evaluation.

Do not use:

- `dataset/RAD_train/sst2/` in version 1.
- `nontoxic_prompts-10k.jsonl` for sentiment-router training or evaluation; reserve it for a later detoxification experiment.

## First required action: inspect the local source format

Before implementing preprocessing, inspect all files under:

```text
dataset/RAD_train/amazon_polarity/
```

The coding AI must report:

- discovered filenames and formats;
- row count per source split/file;
- available fields or columns;
- label representation and which value means positive;
- whether title and review body are stored separately;
- whether official train/test splits already exist;
- sample rows with long text truncated in logs.

Do not assume Hugging Face Arrow, CSV, JSONL, or a particular column schema until the local files are inspected.

## Source-selection policy

For version 1:

1. Keep only positive Amazon Polarity examples.
2. Use the existing Amazon Polarity training portion as the source when an official split is available.
3. Do not use the official Amazon Polarity test portion for router training. Keep it unused or reserve it for later auxiliary analysis.
4. If there is no official split, create deterministic source-level splits before converting texts into prompt–continuation examples.
5. Never use RAD benchmark samples as router-training targets.

## Building the source text

Use the actual available text fields:

- If one review-text field exists, use it directly.
- If both title and body exist, combine them deterministically, for example:

```text
<title>. <body>
```

Do not insert synthetic sentiment words or rewrite the review using an LLM.

Normalize only:

- Unicode representation when necessary;
- repeated whitespace;
- leading/trailing whitespace.

Do not aggressively remove punctuation, capitalization, negation, or sentiment-bearing words.

## Filtering rules

Tokenize lengths using the same GPT-2 tokenizer planned for the base LM and reward model.

Recommended initial filters:

```text
Complete source text: 20–150 GPT-2 tokens
Prompt:                at least 5 GPT-2 tokens
Continuation:          8–80 GPT-2 tokens
Language:              English
Label:                 positive only
```

Also remove:

- empty or malformed rows;
- reviews containing only punctuation or markup;
- exact duplicates;
- obvious template/spam duplicates;
- samples whose continuation becomes trivial after splitting.

Document every removal category in `data_report.json`.

## Prompt–continuation splitting

For every accepted positive review, create exactly one prompt–continuation example in the first implementation.

Choose a deterministic split point from a configurable interval, initially 30%–60% of GPT-2 token length.

Requirements:

- split on a token boundary;
- prefer a nearby whitespace or sentence boundary when possible;
- retain the original text exactly apart from whitespace normalization;
- ensure prompt and continuation concatenate back to the normalized source text;
- ensure the continuation has enough meaningful content;
- use a fixed random seed when selecting among valid split locations.

The script must support configurable arguments such as:

```text
--min-total-tokens 20
--max-total-tokens 150
--min-prompt-tokens 5
--min-continuation-tokens 8
--max-continuation-tokens 80
--min-split-ratio 0.30
--max-split-ratio 0.60
--seed 42
```

## Required output schema

Each processed router example must use:

```json
{
  "id": "amazon-polarity-positive-000001",
  "prompt": "The beginning was a little slow, but",
  "continuation": " the product eventually proved useful and reliable.",
  "label": 1,
  "source": "amazon_polarity",
  "source_split": "train",
  "source_index": 12345
}
```

Required fields:

- `id`: unique and stable identifier.
- `prompt`: prefix supplied to the base LM.
- `continuation`: gold continuation used for teacher forcing.
- `label`: normalized positive label, fixed to `1` in version 1.
- `source`: exactly `amazon_polarity`.
- `source_split`: original local source split/file role where available.
- `source_index`: stable original row index or record identifier.

Optional audit fields may be stored in an auxiliary manifest rather than the training JSONL:

- source-file name;
- normalized-text hash;
- prompt hash;
- continuation hash;
- total GPT-2 token count;
- split-token index.

## Recommended output split

Final proof-of-concept target:

```text
Train:       20,000 examples
Validation:   2,000 examples
Optional dev:   500 examples
```

For the first smoke test, support a small mode such as:

```text
Train:       1,000 examples
Validation:    200 examples
```

Split at the source-record level after deduplication and leakage removal. Use a deterministic seed. A source review must appear in only one output split.

## Leakage prevention against RAD benchmark

Treat these files as held-out:

```text
dataset/rad_benchmark/negative_prompts.jsonl
dataset/rad_benchmark/neutral_prompts.jsonl
dataset/rad_benchmark/positive_prompts.jsonl
```

The script must tolerate the benchmark's nested JSON schema, including fields such as:

```json
{
  "prompt": {"text": "..."},
  "continuation": {"text": "..."}
}
```

Extract all available benchmark strings:

- prompt text;
- reference continuation text when present;
- prompt + continuation.

Create normalized fingerprints for each Amazon sample's:

- complete source text;
- prompt;
- continuation;
- prompt + continuation.

Minimum normalization for matching:

- Unicode normalization;
- lowercase;
- collapse whitespace;
- strip surrounding whitespace and punctuation.

Remove exact matches. Also perform a documented near-duplicate check using a practical method such as token-set Jaccard, character n-gram similarity, or MinHash. The threshold must be configurable and recorded.

Do not tune data rules by inspecting downstream results on the RAD benchmark.

## Deliverables

Write processed outputs to:

```text
dataset/RAD_train/router_amazon_polarity/
├── train.jsonl
├── validation.jsonl
├── dev.jsonl                 # optional
├── data_report.json
└── manifest.jsonl            # recommended audit metadata
```

Recommended implementation files:

```text
router/scripts/data/
├── inspect_amazon_polarity.py
├── build_router_amazon_polarity.py
└── check_router_data.py

router/tests/
└── test_router_data_preparation.py
```

## Required `data_report.json`

Include at least:

```text
source directory and discovered files
source format and field mapping
positive-label mapping
raw row counts by source file/split
positive rows before filtering
removed malformed rows
removed length violations
removed exact duplicates
removed near duplicates
removed benchmark leakage
final train/validation/dev counts
average/min/max total tokens
average/min/max prompt tokens
average/min/max continuation tokens
split-ratio statistics
source-file hashes
processed-output hashes
random seed
all preprocessing arguments
```

## Acceptance criteria

- Every output line parses as JSON.
- All IDs are unique and stable for the same configuration.
- Every example originates from a positive Amazon Polarity row.
- No source row appears in more than one processed split.
- No train/validation/dev overlap under normalized hashes.
- No overlap with negative, neutral, or positive RAD benchmarks under the selected leakage checks.
- Every continuation contains at least the configured minimum number of non-special GPT-2 tokens.
- `prompt + continuation` reconstructs the normalized source text.
- Counts and hashes are documented.
- Re-running with the same inputs and seed produces identical outputs.
- Existing Amazon Polarity files and RAD benchmark files are never modified.

## Prompt for the coding AI

Inspect the existing local dataset under `dataset/RAD_train/amazon_polarity/` and implement a deterministic pipeline that converts positive Amazon Polarity reviews into prompt–continuation examples for token-level router training. Do not download data and do not use SST-2. Detect the actual local file format and field names before coding the conversion. Use the GPT-2 tokenizer for length filtering and token-boundary splitting. Create 20,000 training examples and 2,000 validation examples when enough valid rows exist, with an optional 500-example dev split and a configurable smoke-test mode. Add exact and near-duplicate leakage checks against `dataset/rad_benchmark/negative_prompts.jsonl`, `neutral_prompts.jsonl`, and `positive_prompts.jsonl`, including nested `prompt.text` and `continuation.text` fields. Produce `train.jsonl`, `validation.jsonl`, optional `dev.jsonl`, `data_report.json`, and an audit manifest under `dataset/RAD_train/router_amazon_polarity/`. Include CLI arguments, type hints, logging, seed control, source/output hashes, and unit tests for source inspection, positive-label mapping, token splitting, reconstruction, deduplication, deterministic splitting, and benchmark leakage detection. Do not modify model, router, reward-model, inference, or benchmark code in this stage.
