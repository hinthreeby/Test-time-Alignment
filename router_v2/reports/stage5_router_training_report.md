# Stage 5 Router V2 Training Report

Generated: 2026-08-16

## Status

**HOST EXECUTION REQUIRED - Stage 5 training implementation is complete but no
scientific training run is claimed yet.**

Stage 3 remains PASS with the real 20K/2K FP32 RAD cache. Stage 4 remains PASS
and ready for training. No V1, PARM, RAD, or Stage 3 cache file was modified.

## Exact NLL Data Path

The Stage 3 cache intentionally stores independent Top-K tensors rather than
full-vocabulary logits. Top-K alone cannot recover the normalizer required by
gold-token cross entropy. Stage 5 therefore uses:

~~~text
Stage 3 FP32 Top-K cache -> router features
frozen base GPT-2 online -> transient full base logits
frozen sentiment guide online -> transient full guide logits
gold token IDs -> loss target only
~~~

Both causal LMs run in evaluation mode under `torch.no_grad()` with FP32 model
outputs, explicit attention masks and position IDs. Full logits exist only for
the current source batch and are never stored in cache or checkpoints.

For every valid token:

~~~text
guided = base + lambda_t * (guide - base)
L_NLL = cross_entropy(guided, gold_token_id)
~~~

Padding is excluded by a boolean sequence mask. Base/guide full logits and
cached Top-K logits are detached; the optimizer contains only router parameters.

## Optional Objective Terms

~~~text
H(g_t) = -g_t log(g_t) - (1-g_t) log(1-g_t)
L_smooth = mean((lambda_t - lambda_(t-1))^2)
L_strength = mean((lambda_t / lambda_max)^2)

L = L_NLL
  + entropy_weight * H
  + smoothness_weight * L_smooth
  + strength_weight * L_strength
~~~

Smoothness uses only adjacent valid tokens from the same sample. TARO configs
reject Smart smoothness/strength terms. The first recommended runs leave both
Smart regularizers off; a separate history regularization config is provided
for ablation.

## History Semantics

For history training, `selected_score[t]` is the detached base log-probability
assigned to teacher-forced token `t`. `SmartTokenRouter` shifts this internally:

~~~text
selected_score[t] may affect lambda[t+1] and later
selected_score[t] cannot affect lambda[t] or any earlier token
~~~

This uses a previously selected token exactly as online decoding can. Current
gold identity never enters the current router feature.

## Ordered Stages

| Stage | Router config | Preference |
|---|---|---|
| `taro` | faithful flattened TARO | none |
| `state` | Top-K + confidence + position | none |
| `history` | previous groups + router GRU | none |
| `alpha` | previous groups + preference MLP | required from cache |

Every stage after TARO requires its predecessor `run_status.json` to be PASS.
Architectures are separate experiments and do not load structurally
incompatible predecessor weights.

The current RAD cache has `preference_vector=None`. Alpha training is therefore
implemented but intentionally blocked. It will not substitute a constant or
synthetic vector and call that a multi-objective result.

## Diagnostics

Each train/validation epoch records:

~~~text
NLL and total loss
token accuracy
entropy, smoothness, strength
lambda mean/std/min/max
p05/p25/p50/p75/p95
fraction near zero and near lambda_max
lambda by exact continuation position
Pearson(lambda, base entropy)
Pearson(lambda, JS divergence)
lambda by every alpha dimension and bucket, when available
~~~

Diagnostic state is checkpointed, so an interrupted mid-epoch resume does not
discard earlier observations.

## Same-Average Control

After adaptive validation, Stage 5 computes the exact validation mean lambda.
It then performs a second frozen online pass using this constant for every
token. The run status includes:

~~~text
base lambda=0 NLL
adaptive NLL
same-average fixed-lambda NLL
~~~

A scientific stage is PASS only when validation is finite, adaptive NLL beats
base and same-average fixed routing, lambda is non-constant, and all frozen
artifact checks pass.

## Frozen Audit

Before and after training, SHA-256 descriptors are recomputed for:

~~~text
models/gpt2-medium/
results/router_v2/sentiment_guide/final_adapter/
PARM/
~~~

Cache manifest model/tokenizer hashes must match the live checkpoints before
training starts. Both online models have `requires_grad=False`; optimizer
membership is checked against router parameter identities.

## Files Added Or Changed

~~~text
router_v2/training/__init__.py
router_v2/training/config.py
router_v2/training/data.py
router_v2/training/objective.py
router_v2/training/diagnostics.py
router_v2/training/online.py
router_v2/training/audit.py
router_v2/training/engine.py
router_v2/scripts/train_router_v2.py
router_v2/schemas/router_training_config.schema.json
router_v2/configs/train_stage5_taro_rad.json
router_v2/configs/train_stage5_taro_rad_entropy.json
router_v2/configs/train_stage5_state_rad.json
router_v2/configs/train_stage5_history_rad.json
router_v2/configs/train_stage5_history_rad_regularized.json
router_v2/configs/train_stage5_alpha_rad_blocked.json
router_v2/tests/test_router_training.py
router_v2/README.md
router_v2/reports/stage5_router_training_report.md
~~~

## CPU Verification

~~~text
Ran 100 tests in 4.038s
OK
~~~

The 11 Stage 5 tests cover config/stage contracts, exact masked NLL, all
regularizers, sequence padding and boundaries, diagnostics/resume state,
same-average full-logit evaluation, FP32 teacher forcing, optimizer isolation,
and frozen tree comparison. Existing Stage 2-4 and protected-tree regressions
also pass.

Real-cache read-only sanity:

~~~text
batch: 4 complete source samples
Top-K shape: [4, 59, 20]
lengths: [59, 32, 17, 27]
positions contiguous: true for every sample
preference: None
~~~

## Host Environment

~~~bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
set -o pipefail

export PY=/home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p results/router_v2/logs
nvidia-smi
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
~~~

## CPU Tests

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m unittest discover -s router_v2/tests -v \
  2>&1 | tee results/router_v2/logs/stage5_cpu_tests.log
~~~

## GPU Engineering Smoke

This smoke uses real models/cache but only 8 train and 8 validation samples.
Its small-sample scientific status may be NOT_PASS; the purpose is execution,
gradient, checkpoint, diagnostic, and checksum validation.

~~~bash
"$PY" -m router_v2.scripts.train_router_v2 \
  --config router_v2/configs/train_stage5_taro_rad.json \
  --device cuda --no-cpu-fallback \
  --max-train-samples 8 --max-validation-samples 8 \
  --output-dir results/router_v2/training_smoke/taro \
  2>&1 | tee results/router_v2/logs/stage5_taro_smoke.log

"$PY" -m json.tool \
  results/router_v2/training_smoke/taro/run_status.json
~~~

Resume that smoke after interruption:

~~~bash
"$PY" -m router_v2.scripts.train_router_v2 \
  --config router_v2/configs/train_stage5_taro_rad.json \
  --device cuda --no-cpu-fallback \
  --max-train-samples 8 --max-validation-samples 8 \
  --output-dir results/router_v2/training_smoke/taro \
  --resume \
  2>&1 | tee -a results/router_v2/logs/stage5_taro_smoke.log
~~~

## Full Ordered Training

Run TARO first:

~~~bash
"$PY" -m router_v2.scripts.train_router_v2 \
  --config router_v2/configs/train_stage5_taro_rad.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/router_v2/logs/stage5_taro.log

"$PY" -m json.tool results/router_v2/training/taro/run_status.json
~~~

Only if TARO is PASS, run state:

~~~bash
"$PY" -m router_v2.scripts.train_router_v2 \
  --config router_v2/configs/train_stage5_state_rad.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/router_v2/logs/stage5_state.log

"$PY" -m json.tool results/router_v2/training/state/run_status.json
~~~

Only if state is PASS, run history:

~~~bash
"$PY" -m router_v2.scripts.train_router_v2 \
  --config router_v2/configs/train_stage5_history_rad.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/router_v2/logs/stage5_history.log

"$PY" -m json.tool results/router_v2/training/history/run_status.json
~~~

Resume any full stage by repeating its exact command with `--resume` and
changing `tee` to `tee -a`.

Do not run `train_stage5_alpha_rad_blocked.json` on the current cache. Alpha
requires real preference vectors and remains blocked by design.

## Monitoring

~~~bash
watch -n 1 nvidia-smi
~~~

~~~bash
tail -F \
  results/router_v2/logs/stage5_taro.log \
  results/router_v2/logs/stage5_state.log \
  results/router_v2/logs/stage5_history.log
~~~

## Files To Return After Host Smoke

~~~text
results/router_v2/logs/stage5_taro_smoke.log
results/router_v2/training_smoke/taro/run_status.json
results/router_v2/training_smoke/taro/epoch_000.json
results/router_v2/training_smoke/taro/frozen_audit_before.json
results/router_v2/training_smoke/taro/frozen_audit_after.json
~~~

Do not send checkpoint tensors unless a load/resume failure needs diagnosis.
