# Stage 3 Faithful TARO Feature Cache Report

Generated: 2026-08-15

## Final Status

**HOST EXECUTION REQUIRED — FP32 smoke and persistent cache**

The sentiment guide is trained and validated. The first real 20/20 smoke cache
was extracted in FP16 but failed strict online recomputation. It is preserved
unchanged as diagnostic evidence and is excluded from scientific Stage 3.

No FP32 scientific smoke or persistent 20K/2K cache exists yet, so Stage 3 is
not marked PASS.

## Guide Evidence

Base:

~~~text
path: models/gpt2-medium
architecture: GPT2LMHeadModel
vocabulary: 50,257
weight SHA-256: fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd
~~~

Guide:

~~~text
path: results/router_v2/sentiment_guide/final_adapter
architecture: GPT2LMHeadModel plus LoRA
adapter SHA-256: 17b0e1b5594bbd7780015434b927d36a6f0f78be370e8cfc966f15e597d4c6dd
checkpoint descriptor: 224b3151332911c5089e56c8fdf7d9e285d10503865a04d3529f528a3a8032ca
trainable parameters: 3,145,728
objective: positive-continuation causal NLL
~~~

Guide validation:

~~~text
status: PASS
full vocabulary output: PASS
exact tokenizer compatibility: PASS
models frozen: PASS
positive validation NLL improvement: 0.1687728713 nats/token
positive-minus-negative direction gap: 0.0472727877 nats/token
report SHA-256: e48712f22e896593ef173d37b2770c362cbadddcd1eb0a652d115f4880b7bb85
~~~

The base and guide tokenizer vocabulary hashes are both:

~~~text
79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
~~~

## FP16 Smoke Evidence

Preserved path:

~~~text
dataset/router_v2_cache/rad_smoke/
~~~

Observed:

~~~text
train source examples: 20
train token records: 780
validation source examples: 20
validation token records: 800
tensor shards: 2
precision: fp16
audit status: FAIL
audit SHA-256: a1bae897f9dac3ca79c5c01720912ffb718d1c9cb16ace2f19410b558cac35bd
~~~

Tokenizer, shard hashes, leakage controls, and independent Top-K logic passed.
Online values failed because FP16 numerical changes altered logits and near-tied
Top-K IDs. Inspection also found that the original audit recomputed singleton
samples while extraction used batch size 4, so padding and GEMM shapes were not
identical. Increasing tolerance would not repair either issue.

The FP16 cache and failed report are not deleted, overwritten, finalized, or
counted toward Stage 3.

## Scientific FP32 Runtime

Scientific paths are now:

~~~text
smoke: dataset/router_v2_cache/rad_smoke_fp32/
full:  dataset/router_v2_cache/rad/
~~~

Both paths enforce:

- base and guide parameters loaded as FP32;
- model outputs and full-vocabulary logits are FP32;
- independent Top-K selected directly from transient FP32 full logits;
- only Top-K CPU float32 tensors are persisted;
- derived features are computed in float32;
- model.eval, torch.inference_mode, fixed seeds, deterministic algorithms;
- TF32 disabled and float32 matmul precision set to highest;
- explicit position IDs;
- identical tokenizer/model loader shared by extraction and audit;
- audit recreates the exact source batch, padding, attention mask, and position
  IDs recorded by extraction;
- shard boundaries preserve complete extraction batches, so interruption and
  resume cannot change padding/GEMM context for a partially saved batch;
- base/guide Top-K IDs and gold IDs must match exactly;
- logits and derived fields use rtol 0 and atol at most 1e-5.

The full-vocabulary tensors exist only inside the current model forward and are
discarded immediately after Top-K selection.

## Mismatch Diagnostics

Audit schema version 2 records the following for every mismatch:

~~~text
sample_id and continuation position
cached and online base Top-K IDs/logits
cached and online guide Top-K IDs/logits
base and guide K/K+1 boundary margins
minimum boundary margin
prefix token IDs and SHA-256
prefix length and model logit position
padded batch length and attention length
position ID
model output dtypes
base and guide checkpoint hashes
~~~

This diagnostic is evidence only. Top-K ID mismatch always fails the audit,
regardless of boundary margin or numeric tolerance.

## Files Changed For FP32 Reproducibility

~~~text
router_v2/guide_model/runtime.py
router_v2/guide_model/extraction.py
router_v2/scripts/extract_full_logits.py
router_v2/scripts/audit_feature_cache.py
router_v2/scripts/finalize_stage3.py
router_v2/tests/test_guide_model_pipeline.py
router_v2/README.md
dataset/router_v2_cache/cache_status.json
router_v2/reports/stage3_feature_cache_report.md
~~~

The earlier AMP training fix remains active: frozen GPT-2 weights may stay
FP16 during training, every trainable LoRA parameter and gradient is FP32,
forward uses FP16 autocast, and GradScaler remains enabled.

## CPU Tests

~~~bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" \
/home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python \
  -m unittest discover -s router_v2/tests -v
~~~

Result:

~~~text
Ran 70 tests in 3.900s
OK
~~~

New regressions cover FP32 output enforcement, explicit position IDs, K/K+1
boundary margin, exact extraction-batch grouping, batch-atomic shard resume,
required mismatch diagnostic fields, real PEFT LoRA AMP dtypes/gradients, and
all prior cache contracts.

## Read-Only Regression

All protected hashes still match:

| Protected tree | SHA-256 |
|---|---|
| router/ | 7b5ceafc493c605eb13346c557ad8f768399fcecfe3dceb6ee1137dbf518b127 |
| router/evaluation/ | 1af858c224e2d196af925e047b97f8ec1392db7289670ce97dbea2c0e7482fd9 |
| PARM/ | aef1aa5fbc30816a97c1c1bb82d7dbc6554138f8e1a267630b6498d14a02289e |
| Method/RAD/ | 3012ada626eabd9dc72dfde349cd4ffc7d0041ec0404cb796e253974b2f5de31 |
| dataset/router_cache/rad/ | 7922216808b1e4614b982d522b7ac73f0c85c1eaf1006484e09d01f03909a36d |
| dataset/router_train/rad/ | 665bef1b2949c45a1ac6d02889ae861723ac16a3e008c809706991c45e4b2ab6 |
| dataset/rad_benchmark/ | 46ed6143daa76fc66a5f6f06e1e3901dde36c4c283698d33e3517accca3880a6 |

## Exact Host Commands

### 1. Environment and regression tests

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

CUDA_VISIBLE_DEVICES="" "$PY" -m unittest discover -s router_v2/tests -v \
  2>&1 | tee results/router_v2/logs/cpu_tests_fp32_cache.log
~~~

### 2. FP32 smoke extraction: train

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split train \
  --output-root dataset/router_v2_cache/rad_smoke_fp32 \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-samples 20 --max-continuation-tokens 80 \
  --shard-size 4096 \
  2>&1 | tee results/router_v2/logs/rad_smoke_fp32_train.log
~~~

### 3. FP32 smoke extraction: validation

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split validation \
  --output-root dataset/router_v2_cache/rad_smoke_fp32 \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-samples 20 --max-continuation-tokens 80 \
  --shard-size 4096 \
  2>&1 | tee results/router_v2/logs/rad_smoke_fp32_validation.log
~~~

Resume an interrupted smoke split with the identical command plus resume:

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split train \
  --output-root dataset/router_v2_cache/rad_smoke_fp32 \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-samples 20 --max-continuation-tokens 80 \
  --shard-size 4096 --resume \
  2>&1 | tee -a results/router_v2/logs/rad_smoke_fp32_train.log
~~~

Use split validation and its validation log for validation resume.

### 4. FP32 smoke audit

~~~bash
"$PY" -m router_v2.scripts.audit_feature_cache \
  --cache-root dataset/router_v2_cache/rad_smoke_fp32 \
  --records-per-split 20 \
  --device cuda --no-cpu-fallback --precision fp32 \
  --fp32-atol 1e-5 \
  --output router_v2/reports/stage3_smoke_fp32_audit.json \
  --overwrite-report \
  2>&1 | tee results/router_v2/logs/rad_smoke_fp32_audit.log

"$PY" -m json.tool router_v2/reports/stage3_smoke_fp32_audit.json
~~~

Do not continue unless the audit status is PASS, both split mismatch counts are
zero, and online_values_match_cache is true.

### 5. Finalize FP32 smoke

~~~bash
"$PY" -m router_v2.scripts.finalize_stage3 --phase smoke \
  2>&1 | tee results/router_v2/logs/rad_smoke_fp32_finalize.log
~~~

### 6. Full FP32 extraction: 20K train

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split train \
  --output-root dataset/router_v2_cache/rad \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-continuation-tokens 80 --shard-size 4096 \
  2>&1 | tee results/router_v2/logs/rad_full_fp32_train.log
~~~

### 7. Full FP32 extraction: 2K validation

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split validation \
  --output-root dataset/router_v2_cache/rad \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-continuation-tokens 80 --shard-size 4096 \
  2>&1 | tee results/router_v2/logs/rad_full_fp32_validation.log
~~~

### 8. Resume full extraction

~~~bash
"$PY" -m router_v2.scripts.extract_full_logits \
  --split train \
  --output-root dataset/router_v2_cache/rad \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-continuation-tokens 80 --shard-size 4096 \
  --resume \
  2>&1 | tee -a results/router_v2/logs/rad_full_fp32_train.log

"$PY" -m router_v2.scripts.extract_full_logits \
  --split validation \
  --output-root dataset/router_v2_cache/rad \
  --device cuda --no-cpu-fallback --precision fp32 \
  --batch-size 4 --max-continuation-tokens 80 --shard-size 4096 \
  --resume \
  2>&1 | tee -a results/router_v2/logs/rad_full_fp32_validation.log
~~~

### 9. Final FP32 recomputation audit

~~~bash
"$PY" -m router_v2.scripts.audit_feature_cache \
  --cache-root dataset/router_v2_cache/rad \
  --records-per-split 20 \
  --device cuda --no-cpu-fallback --precision fp32 \
  --fp32-atol 1e-5 \
  --output router_v2/reports/stage3_recomputation_audit.json \
  --overwrite-report \
  2>&1 | tee results/router_v2/logs/rad_full_fp32_audit.log

"$PY" -m json.tool router_v2/reports/stage3_recomputation_audit.json
~~~

### 10. Finalize and inspect Stage 3

~~~bash
"$PY" -m router_v2.scripts.finalize_stage3 --phase full \
  2>&1 | tee results/router_v2/logs/rad_full_fp32_finalize.log

"$PY" -m json.tool dataset/router_v2_cache/cache_status.json
~~~

The final status is valid only when it contains:

~~~text
guide_logits_ready = true
smoke_cache_ready = true
train_cache_ready = true
validation_cache_ready = true
tensor_shards_generated > 0
stage3_status = PASS
~~~

### 11. Monitor

~~~bash
watch -n 1 nvidia-smi
~~~

~~~bash
tail -F \
  results/router_v2/logs/rad_smoke_fp32_train.log \
  results/router_v2/logs/rad_smoke_fp32_validation.log \
  results/router_v2/logs/rad_full_fp32_train.log \
  results/router_v2/logs/rad_full_fp32_validation.log
~~~

## Files To Return

After FP32 smoke:

~~~text
results/router_v2/logs/rad_smoke_fp32_train.log
results/router_v2/logs/rad_smoke_fp32_validation.log
results/router_v2/logs/rad_smoke_fp32_audit.log
dataset/router_v2_cache/rad_smoke_fp32/train/manifest.json
dataset/router_v2_cache/rad_smoke_fp32/validation/manifest.json
router_v2/reports/stage3_smoke_fp32_audit.json
dataset/router_v2_cache/cache_status.json
~~~

After full extraction:

~~~text
results/router_v2/logs/rad_full_fp32_train.log
results/router_v2/logs/rad_full_fp32_validation.log
results/router_v2/logs/rad_full_fp32_audit.log
dataset/router_v2_cache/rad/train/manifest.json
dataset/router_v2_cache/rad/validation/manifest.json
router_v2/reports/stage3_recomputation_audit.json
dataset/router_v2_cache/cache_status.json
~~~

Do not send large shard tensors unless a specific diagnostic requires one.
