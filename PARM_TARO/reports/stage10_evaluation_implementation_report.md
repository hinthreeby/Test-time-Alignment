# Stage 10 PARM-TARO Evaluation Implementation

## Status

```text
IMPLEMENTATION PASS - GPU smoke not launched
HOST PREREQUISITES REQUIRED - official Beaver reward/cost scorers are not local
```

Stage 9 is officially PASS and the final alpha-aware checkpoint is pinned to:

```text
results/parm_taro/training/v2_alpha_preference/final.pt
SHA256 56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20
```

No Stage 9 checkpoint was trained or overwritten. No final test prompt was
loaded by the implementation/preflight path.

## Existing-Code Audit

- Reused the frozen one-backbone NF4 Tulu/PBLORA runtime from Stage 9.
- Reused the PARM-specific full-vocabulary equation and causal decoder from
  `PARM_TARO/decoding/`.
- Reused Router V2 checkpoint loaders, independent Top-K construction, and the
  RAD text-only coherence/distinct/repetition primitives.
- Did not reuse RAD's single-objective Pareto definition for multi-objective
  PARM evaluation.
- Implemented HV, MIP, PCS, shared-pool Preference Regret, paired inference,
  and Holm correction in a new `PARM_TARO/evaluation/` package.
- Original `PARM/code/evaluation/compute_reward.py` confirms the intended
  scorers are `PKU-Alignment/beaver-7b-v1.0-reward` and
  `PKU-Alignment/beaver-7b-v1.0-cost`.

The official Safe-RLHF project describes reward maximization and cost
minimization and publishes the corresponding v1 checkpoints. Its source and
both checkpoint trees are hashed into the protocol lock.

## Frozen Methods

```text
parm_static
parm_taro
parm_v2_no_alpha
parm_v2_full_alpha
parm_v2_full_alpha_same_average
parm_v2_full_alpha_shuffled_alpha
parm_v2_full_alpha_fixed_alpha
```

`parm_static` uses the original static coefficient `lambda=1`. TARO and
no-alpha load their final Stage 9 checkpoints. Full-alpha and both alpha
controls load only the pinned production checkpoint.

For shuffled/fixed controls, PBLORA receives the requested alpha while only
the Router alpha is changed. This isolates the learned Router alpha pathway.
Shuffling is the deterministic cyclic grid offset `+1`; fixed alpha is
`[0.5, 0.5]`. Same-average uses the token-weighted mean lambda of full-alpha
generation in the same run. It does not inspect rewards or labels.

## Frozen Evaluation Protocol

Evaluation split:

```text
dataset/parm_taro/test_prompt_only.json
1,500 prompts
SHA256 1ab1b59e3127d0077ed6908e538072bdfe541266affae51267d90d2ec8255472
```

The test payload may be opened only after a valid protocol lock exists.
Calibration uses the 500-example validation split and both reference responses
(`1,000` score records). The test split is never used for normalization or
threshold selection.

Alpha grid, in `[helpfulness, harmlessness]` order:

```text
[0.00, 1.00]
[0.25, 0.75]
[0.50, 0.50]
[0.75, 0.25]
[1.00, 0.00]
```

Generation is greedy, one deterministic seed (`2026`), temperature `1.0`,
normal EOS termination, and `max_new_tokens=64` for every method.

## Objectives And Normalization

Raw vector:

```text
r_raw = [Beaver reward, -Beaver cost]
```

Both coordinates therefore use higher-is-better semantics. Each coordinate is
fit independently on the validation reference pool using its 1st/99th
percentiles, transformed by affine min-max scaling, then clipped to `[0,1]`.
The exact numeric anchors, model tree hashes, Safe-RLHF source hash, and
protocol specification hash are written once to:

```text
results/parm_taro/evaluation/protocol/protocol_lock.json
```

The lock is no-overwrite and is revalidated before smoke/full generation.

## Pareto, HV, MIP, PCS, And Regret

- Pareto points are per-method/per-alpha mean normalized objective vectors.
- Both objectives are maximized.
- Exact two-dimensional HV uses the shared normalized reference `[0,0]`.
- MIP is `mean(alpha^T r_normalized)`.
- PCS is cosine similarity between alpha and the same normalized vector.
- Preference Regret uses one oracle candidate pool shared by all methods for
  each identical `(prompt, requested alpha, seed)` task. The oracle is the
  maximum MIP among all seven outputs; no method-specific oracle is allowed.
- Raw and normalized objective vectors are retained for recomputation.

Quality output includes base-conditional PPL, text coherence proxy,
distinct-1/2/3, 4-gram repetition, generation length, total latency, latency
per token, and complete lambda histories.

## Statistics

Primary comparisons are full-alpha against static PARM, TARO, V2 no-alpha,
and same-average-lambda. Prompt ID is the paired resampling cluster; alpha
tasks and deterministic seeds are not treated as independent prompt samples.

Each comparison reports paired 95% bootstrap CI, paired sign permutation test,
paired Cohen `d_z` for additive metrics, and Holm-adjusted p-values across all
primary comparisons/metrics. HV uses prompt-cluster bootstrap and paired
method-label swaps because HV is non-additive.

## Files Created Or Changed

Created:

```text
PARM_TARO/evaluation/__init__.py
PARM_TARO/evaluation/config.py
PARM_TARO/evaluation/engine.py
PARM_TARO/evaluation/generation.py
PARM_TARO/evaluation/io.py
PARM_TARO/evaluation/metrics.py
PARM_TARO/evaluation/protocol.py
PARM_TARO/evaluation/scoring.py
PARM_TARO/evaluation/statistics.py
PARM_TARO/configs/evaluate_stage10_parm_taro.json
PARM_TARO/schemas/stage10_evaluation_record.schema.json
PARM_TARO/scripts/evaluate_stage10.py
PARM_TARO/tests/test_stage10_evaluation.py
PARM_TARO/reports/stage10_evaluation_implementation_report.md
results/parm_taro/evaluation/preflight.json
```

Changed:

```text
PARM_TARO/decoding/adaptive.py
PARM_TARO/tests/test_parm_taro.py
```

The decoder change is backward compatible: `router_preference` is optional and
defaults to the same alpha passed to PBLORA.

## CPU Verification

```text
PARM_TARO tests: 109 PASS
router_v2 tests: 120 PASS
total: 229 PASS
```

Protected current hashes after implementation:

```text
PARM      dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
Router V1 8595fb3e48c2b1771a07a6efba09c29560785a581d5490f582286c58fbf5ffd3
Method/RAD e0c8907885b80cde9c12f0e1f6e3134b0781e811ee8edafb9f0d690988d36bbe
```

Existing V1 regression tests and Stage 9 PARM hash tests pass. The production
checkpoint remains byte-identical.

## Current Preflight

Passed:

```text
protocol_specification_valid
stage9_official_pass
stage9_production_checkpoint_matches
```

Missing locally:

```text
models/safe-rlhf-source/
models/beaver-7b-v1.0-reward/
models/beaver-7b-v1.0-cost/
```

No local classifier is accepted as a silent replacement.

## Host Commands

One-time official scorer materialization (large downloads):

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
export PY=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
export HFCLI=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/huggingface-cli

git clone --depth 1 https://github.com/PKU-Alignment/safe-rlhf.git \
  models/safe-rlhf-source

"$HFCLI" download PKU-Alignment/beaver-7b-v1.0-reward \
  --local-dir models/beaver-7b-v1.0-reward \
  --local-dir-use-symlinks False --resume-download

"$HFCLI" download PKU-Alignment/beaver-7b-v1.0-cost \
  --local-dir models/beaver-7b-v1.0-cost \
  --local-dir-use-symlinks False --resume-download
```

Preflight and validation-only protocol freeze:

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase preflight

"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase freeze --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage10_protocol_freeze.log
```

Small GPU smoke (`2 validation prompts x 5 alpha x 7 methods = 70 records`):

```bash
"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase smoke --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage10_smoke.log
```

Resume smoke:

```bash
"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase smoke --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/parm_taro/logs/stage10_smoke.log
```

Full evaluation command, intentionally not launched by the agent:

```bash
"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase full --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage10_full.log
```

Resume full evaluation:

```bash
"$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config PARM_TARO/configs/evaluate_stage10_parm_taro.json \
  --phase full --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/parm_taro/logs/stage10_full.log
```

GPU monitoring:

```bash
watch -n 2 nvidia-smi
```

Do not run full evaluation until protocol freeze and smoke both report `PASS`.
