# Stage 3 — Build the Offline Token-Level Feature Cache

## Goal

Convert router train and validation examples into cached token-level tensors so the router can be trained repeatedly without rerunning GPT-2 Large and the reward model every epoch.

## Teacher-forcing procedure

For each sample:

```text
context = prompt
for each gold continuation token y_t:
    run base LM on context
    select top-k base candidates
    add y_t if absent
    score every candidate with reward model
    save one token-level training record
    append y_t to context
```

Only continuation tokens produce training records. Prompt tokens must never contribute to router NLL.

## Candidate-set policy

Inference uses exactly `top_k = 20`.

Training must ensure that the gold token is present:

\[
C_t=\operatorname{TopK}_{20}(z_t^{base})\cup\{y_t^*\}.
\]

To maintain a fixed dimension of 20:

- if the gold token is already in top-20, retain top-20;
- otherwise replace the lowest-ranked top-20 candidate with the gold token;
- store whether replacement occurred.

This keeps the router input fixed at 40 dimensions.

## Cached fields

Each step should contain:

```python
{
    "sample_id": str,
    "position": int,
    "candidate_ids": LongTensor[20],
    "base_logits": FloatTensor[20],
    "reward_scores": FloatTensor[20],
    "gold_token_id": int,
    "gold_index": int,
    "gold_was_in_topk": bool,
    "attention_length": int,
}
```

Optional diagnostic fields:

- decoded candidate strings;
- base entropy;
- reward range;
- original top-k rank of gold when available.

## Normalization policy

Cache raw candidate values. Perform normalization in the router data loader or feature builder:

\[
\tilde\ell_t=\frac{\ell_t-\mu(\ell_t)}{\sigma(\ell_t)+\epsilon},
\]

\[
\tilde r_t=\frac{r_t-\mu(r_t)}{\sigma(r_t)+\epsilon}.
\]

Keeping raw values makes future ablations possible.

## Efficiency requirements

- Use `torch.inference_mode()`.
- Batch candidate scoring where possible.
- Support resume from partial progress.
- Save shards rather than one huge file.
- Record a manifest with shard hashes.
- Avoid recomputing samples already completed.

Recommended structure:

```text
cache/router_features/
├── train/
│   ├── shard_00000.pt
│   ├── shard_00001.pt
│   └── manifest.json
└── validation/
    ├── shard_00000.pt
    └── manifest.json
```

## Required statistics

Manifest must report:

- number of source samples;
- number of cached token steps;
- gold-in-original-top-k rate;
- average continuation length;
- base logit statistics;
- reward score statistics;
- number of skipped or failed samples;
- tokenizer/model identifiers;
- top-k value;
- source dataset hash.

## Acceptance criteria

- Every cached record has exactly 20 candidates.
- `candidate_ids[gold_index] == gold_token_id`.
- No prompt tokens are cached as targets.
- Values contain no NaN or infinity.
- Re-running with resume does not duplicate records.
- Randomly reconstructed records match direct online computation.

## Prompt for the coding AI

Implement an offline feature extraction pipeline for teacher-forced RAD router training. Use top-k=20, force the gold continuation token into the fixed-size candidate set by replacing the lowest candidate when necessary, and cache raw base logits plus reward scores in resumable tensor shards. Generate manifests with hashes and statistics. Add integrity tests that compare cached records against online recomputation.
