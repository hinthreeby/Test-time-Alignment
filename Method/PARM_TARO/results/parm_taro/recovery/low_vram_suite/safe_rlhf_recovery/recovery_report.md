# Safe-RLHF scorer runtime recovery

## Result

Phase 07 scorer preflight now passes without loading either 7B model. The
official Safe-RLHF score-model runtime is pinned at commit
`e8cca16665ef2340ac92c6514f05519310251581` under
`models/safe-rlhf-source`.

This is a **REPRODUCED_RUNTIME**, not a claim of byte-identical historical
source. The Stage-10 protocol recorded only a full-directory hash that included
mutable `.git` state; no upstream commit was recorded. Searches of local source,
caches, Conda environments, logs, shell history, and project Git history found
no surviving checkout or installed package.

## Why source is required

The Beaver directories contain weights, tokenizer, and LLaMA configuration,
but Transformers does not provide the custom scalar-score architecture. The
historical PARM evaluator imports `safe_rlhf.models.AutoModelForScore`, which
maps the config to `LlamaForScore`, attaches the score head, and returns
`ScoreModelOutput.end_scores` at the final attended token.

The upstream package root eagerly imports training modules and therefore asks
for DeepSpeed, which is absent from `genarm`. Phase 07/09 do not require those
trainers. The recovery suite now loads only the pinned upstream `score_model`
and `normalizer` namespaces; those implementation files and scoring equations
are unchanged.

## Semantics preserved

- Template: `BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}`
- Helpful reward: higher is better.
- Harm/cost: lower is safer; higher is more harmful.
- Phase 09 safety coordinate: `harmlessness_raw = -cost_raw`.
- `end_scores`: score at the last non-padding position from `attention_mask`.
- Reward and cost models remain sequential, never resident together.

The local cost checkpoint's `config.json` says `score_type=reward`. This field is
not rewritten: `do_normalize=false`, so it does not change the emitted raw score;
the cost direction is established by the checkpoint role and the historical
PARM evaluator (`harm_score (low better)`).

## Validation

- `AutoModelForScore` scorer-only import: PASS
- Both configs resolve to `LlamaForScore`: PASS
- Both local tokenizers resolve to `LlamaTokenizerFast`: PASS
- Reward shards: 7/7 present and non-empty
- Cost shards: 7/7 present and non-empty
- Phase 07 `--preflight-only`: PASS
- Heavy 7B model load: deliberately not run by Codex

## Host command

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
export GPU_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10
$GPU_PYTHON PARM_TARO/recovery/low_vram_suite/07_scorer_sanity.py --limit 10
```
