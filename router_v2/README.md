# Router V2: TARO Baseline and Smart Router

This package is independent from `router/`, `router/evaluation/`,
`Method/RAD/`, and `PARM/`.

For each token step, `select_taro_topk` independently selects:

```text
base Top-K:   (base_token_id_i, base_logit_i) x K
reward Top-K: (reward_token_id_j, reward_logit_j) x K
```

Gold token IDs are not accepted by the selector and never enter router features.
They are used only by the full-vocabulary NLL objective.

Each pair becomes `[logit; token_embedding]`. Base pairs are flattened in Top-K
order, reward pairs are flattened in Top-K order, and both streams are
concatenated:

```text
feature_dim = 2 * top_k * (token_embedding_dim + 1)
Linear(feature_dim, 128) -> Tanh -> Linear(128, 1) -> Sigmoid
```

The gate routes complete vocabulary logits:

```text
guided = (1 - alpha) * base + alpha * reward
```

Supported modes are `taro_topk_nll` and `taro_topk_nll_entropy`. Config and
checkpoint schema version 2 pin the flatten/Tanh architecture. Obsolete
schema-1 pooled/GELU checkpoints fail validation and are not silently loaded.

Checkpoint writes are atomic and refuse overwrite by default. Runtime selection
supports `device=auto|cuda|cpu`; `auto` prefers CUDA and falls back safely to
CPU when configured.

## Smart Router V2

`SmartTokenRouter` extends the frozen TARO input contract without changing the
faithful baseline. Each base/guide candidate is encoded independently from
`[logit; token_embedding]`; separate stream encoders are pooled with mean and
max. Optional current-token groups add 11 confidence/disagreement statistics
and normalized position.

The history group is a small router-only `GRUCell`. It receives previous
`lambda`, selected score, base entropy, and JS divergence. Current or future
selected scores are never used for the current token. The preference group is
a separate MLP over a real supplied preference vector. `preference_dim` is
configurable and must equal `len(alpha)` for later multi-objective integration.
The current RAD cache has no preference vector, so non-alpha RAD presets require
no preference input and alpha training rejects that cache instead of inventing
a constant preference.

All groups are controlled by `SmartRouterConfig`. Named configs provide:

```text
taro_topk                         existing faithful TAROTokenRouter
v2_topk_confidence               token-aware Top-K + confidence
v2_topk_confidence_position      previous groups + position
v2_topk_state_history            previous groups + router GRU
v2_topk_state_history_alpha      previous groups + preference MLP
```

Arbitrary feature ablations use `variant=custom`. A single scalar is emitted
at each step and applied universally across the complete vocabulary:

```text
lambda_t = lambda_max * sigmoid(fusion_features)
guided = base + lambda_t * (guide - base)
```

This interpolation is the TARO/RAD equation only. A later `PARM_TARO/`
implementation must use PARM-specific log-probability guidance semantics and
must not blindly reuse this interpolation helper.

Backbone hidden states are not part of any Smart Router input or method
signature. Smart configs and checkpoints have a separate schema and cannot be
silently loaded as Stage 2 TARO checkpoints.

## Feature cache

The isolated cache contract lives in `router_v2.cache`. It stores independent
base/guide Top-K token IDs and logits, gold targets in a separate target field,
and reconstructable target-independent confidence/disagreement features.

No existing local autoregressive checkpoint is sentiment aligned. Stage 3
therefore trains a GPT-2-medium LoRA guide with continuation-only causal NLL on
the positive RAD train split. The official Amazon Polarity test split supplies
held-out negative reviews for the direction audit.

The real extraction path is:

```text
train_sentiment_guide
  -> validate_sentiment_guide
  -> extract_full_logits (20/20 rad_smoke_fp32)
  -> audit_feature_cache
  -> extract_full_logits (20K/2K rad)
  -> audit_feature_cache
  -> finalize_stage3
```

`extract_full_logits` runs both frozen causal LMs with teacher forcing. Full
vocabulary logits exist only for the current batch; each model's Top-K is
selected independently and moved to CPU before the shard is built. The script
supports `--device auto|cuda|cpu`, bounded batches, atomic shards, and
deterministic `--resume` recovery.

Scientific smoke and persistent RAD caches require FP32 model weights, FP32
full logits, strict FP32 recomputation tolerance, explicit position IDs, and
the exact extraction batch/padding context. The earlier `rad_smoke/` FP16
cache and its failed audit are retained only as diagnostic evidence; scientific
smoke writes to `rad_smoke_fp32/` and never overwrites it.

Exact host commands and the current evidence status are in
`router_v2/reports/stage3_feature_cache_report.md`.

All Stage 3 writers reject destinations outside `results/router_v2/`,
`dataset/router_v2_cache/`, and `router_v2/reports/`. Cache and checkpoint
writes are atomic and refuse overwrite.

## Stage 5 training

`router_v2.training` trains only router parameters. Stage 3 Top-K tensors are
the router input, while base and guide GPT-2 models are loaded frozen and run
online under `torch.no_grad()` to obtain transient FP32 full-vocabulary logits
for exact gold-token NLL. Full logits are never written to a checkpoint or
cache.

The ordered RAD stages are `taro`, `state`, then `history`. Each later stage
requires the predecessor `run_status.json` to be `PASS`; architectures train as
separate experiments rather than loading incompatible predecessor weights.
The `alpha` stage is implemented but rejects the current RAD cache because its
`preference_vector` is absent. It must not be run with invented or constant
multi-objective preferences.

Optional objective terms are Bernoulli gate entropy, adjacent-token lambda
smoothness, and normalized lambda strength. Validation always records the base
`lambda=0` baseline and, by default, performs a second online pass at the
adaptive router's exact mean lambda.

The history selected score is the detached base log-probability of the selected
teacher-forced token. It is shifted inside `SmartTokenRouter`, so score at `t`
can affect only lambda at `t+1` and later. Before/after checksums cover the base
checkpoint, guide checkpoint, and complete `PARM/` tree.

## Stage 6 RAD validation

`router_v2.evaluation.rad` evaluates the trained TARO, Smart state, and Smart
history checkpoints without modifying V1. It uses a two-phase protocol: fixed
lambda, heuristic choice, and TARO/history mean lambda are selected on the
validation prompts; the held-out test phase then evaluates Base, selected Fixed,
selected Heuristic, V1, TARO, state/history V2, and validation-selected
same-average-lambda controls.

V1 rows are streamed from the complete read-only 32-token report after source,
checkpoint, classifier, prompt, seed, and output hashes are verified. The V1
checkpoint is also loaded and run through the current V1 inference code on CPU.
All new artifacts are restricted to `results/router_v2/rad/`.

The 32-token requirement applies to the configured `max_new_tokens` budget.
Realized outputs may be shorter only when decoding terminates on EOS. V1 stops
on EOS, while the current V2 protocol uses `stop_on_eos=false`; the evaluator
therefore audits both length bounds and termination metadata instead of padding
legacy output or requiring every realized sequence to contain 32 tokens.

V1 uses GPT-2 Large with the scalar RAD reward model; V2 uses GPT-2 Medium with
the autoregressive sentiment guide. Raw PPL from these protocols is not treated
as directly comparable. The primary Pareto x-axis is per-sample PPL degradation
relative to the base method from the same family, while alignment uses the same
hashed local sentiment classifier for both families.
