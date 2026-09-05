# Stage 9 Tokenization Boundary Report

## Status

```text
TOKENIZATION FIX PASS
STAGE 9 NOT PASS - TARO training has not completed
```

The real Tulu-2 tokenizer was audited over every Stage 8 train and validation
response after the boundary fix. All 17,000 responses tokenize without an
unexpected error under the production Stage 9 configuration.

## Root Cause

Stage 9 tokenizes the exact concatenated string:

```text
BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}
```

For `pku_round0_01557/response_0`, the response starts with `"` and there is no
space after the prompt template's final colon. Llama SentencePiece therefore
emits one token spanning both sides of the character boundary:

```text
boundary character index       = 81
crossing token index           = 28
crossing token ID              = 6160
crossing tokenizer token/text  = :"
crossing character span        = [80, 82]
first response-loss token      = 29
```

The old code rejected every crossing span. The fixed code preserves token 28
in the model input and causal context, but response NLL begins at token 29,
the first positive-length token whose span starts at or after character 81.
No prompt-only, crossing, or zero-length token becomes a response target.

## Exact Mapping Rule

The full concatenated string is still tokenized once. For tokenizer offset
`[start_i, end_i]` and prompt/response boundary `b`:

```text
prompt-only:       end_i <= b
crossing:          start_i < b < end_i
response target:   start_i >= b and end_i > start_i
zero-length:       start_i == end_i
```

All emitted token IDs remain in causal order in `input_ids`. Only the
`response target` set is used to construct `gold_token_ids` and
`logit_positions`. This leaves clean-boundary examples byte-for-byte and
index-for-index equivalent to the previous target construction.

## Full Audit

| Split | Examples | Responses | Affected examples | Affected responses | Unexpected errors |
|---|---:|---:|---:|---:|---:|
| train | 8,000 | 16,000 | 3 (0.0375%) | 3 (0.01875%) | 0 |
| validation | 500 | 1,000 | 0 (0%) | 0 (0%) | 0 |

Affected train responses:

| Sample | Response | Boundary | Crossing token | Span | First loss token |
|---|---:|---:|---:|---|---:|
| `pku_round0_01054` | 0 | 81 | `6160` (`:"`) | `[80, 82]` | 29 |
| `pku_round0_01557` | 0 | 81 | `6160` (`:"`) | `[80, 82]` | 29 |
| `pku_round0_07805` | 0 | 175 | `17178` (`:$`) | `[174, 176]` | 55 |

Each affected response has exactly one crossing token. There are no
unexpected tokenization errors and no sequence-too-long records at
`max_length=512`, `max_continuation_tokens=128`.

Machine-readable diagnostics are in
`PARM_TARO/reports/stage9_tokenization_boundary_audit.json`.

## Verification

```text
router_v2/tests = 120 passed
PARM_TARO/tests = 59 passed
total           = 179 passed
```

Tests cover clean-boundary equivalence, one crossing token, zero-length
special-token offsets, truncation at the response boundary, deterministic
mapping, prompt/crossing target isolation, and causal next-token alignment.

Protected artifacts remain unchanged:

```text
PARM tree SHA-256 = dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
Stage 8 train SHA-256 = 9a1ca7c8dc559cb6a78d4dc773c40eee085b61588f6edc8f0f87a1f9f9875285
Stage 8 validation SHA-256 = cdbb914443591817d056e76ecd3cc59e28c7e95de526c5c0397c35763d2fc9ba
Stage 8 manifest SHA-256 = dff8b3ff4762d1c864763e365188df4efb8a52adb02309cf2d5712deb423e564
PBLORA weights SHA-256 = 1012790985d4eeb5229efa75a5cdee124fdf526ff226ddb239d1a2d29018ac93
```

## Checkpoint State

The failed host run reached optimizer step 80. Its configuration saves
`latest.pt` every 250 optimizer steps, so no Router/optimizer checkpoint was
written. The existing `frozen_audit_before.json` and
`resolved_training_config.json` are provenance only and cannot reconstruct
the in-memory Router state. A true `--resume` is therefore impossible for
this run; restarting TARO is unavoidable. Preserve the failed provenance by
moving the incomplete directory aside before starting the canonical TARO
output directory again.
