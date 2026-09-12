# Stage 7 PARM-TARO Integration Report

## Status

**STAGE 7 IMPLEMENTATION PASS - PRODUCTION PREREQUISITES REQUIRED**

The isolated integration, static regression, adaptive generation core, alpha
propagation, vendored dependency resolution, and protected-tree checks pass.
Production PARM-TARO generation is intentionally blocked because the workspace
does not contain either required compatible checkpoint:

1. A two-objective (`obj_num = 2`) preference-aware PBLORA PARM adapter.
2. A Tulu-tokenizer Router V2 checkpoint with `use_preference = true`,
   `preference_dim = 2`, and matching tokenizer semantic hash metadata.

No existing RAD/GPT-2 Router checkpoint or ordinary HH LoRA adapter is silently
substituted.

## Implementation

All new source is under `PARM_TARO/`. The implementation imports PARM's local
vendored packages from:

```text
PARM/peft/src/peft
PARM/language-model-arithmetic/src/model_arithmetic
```

The proven environment for both packages is:

```text
/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
```

The previously used `cd` environment lacks `loguru` and `trl`, which are needed
when importing the full vendored model-arithmetic package.

Implemented components:

- read-only loader for the original `PARM/code/evaluation/generate_outputs.py`;
- strict versioned config and JSON schema;
- PBLORA checkpoint validation and in-memory `pref_vec` update;
- shared-tokenizer/base-vocabulary prefix alignment matching original PARM;
- strict preference-aware Router V2 checkpoint loading;
- causal history state with selected score from step `t` visible at `t+1` only;
- full-vocabulary adaptive log-probability guidance;
- autoregressive generation with `device=auto|cuda|cpu` and CPU dtype fallback;
- prerequisite audit and immutable `PARM/` checksum regression.

## Guidance Semantics

For preference vector `alpha`, the PBLORA guide produces
`log pi_PARM(alpha),t`. Router V2 receives the same semantic `alpha` and independent Top-K
features from detached base/guide log probabilities. Its scalar output is used
as:

```text
log pi_tilde_t(v) = normalize(
    log pi_base,t(v) + lambda_t * log pi_PARM(alpha),t(v)
)
```

This integration deliberately does not call Smart Router V2's RAD equation:

```text
base + lambda * (guide - base)
```

The local PARM generation code uses `M_base + M_reward`, a static unit guide
coefficient. Its `beta` values occur in ARM training, not as a literal `1/beta`
in inference. Static regression therefore uses scale `1.0` by default while the
implementation supports any non-negative constant for equivalence testing.

The public Router/data order is `[helpfulness, harmlessness]`; the wrapper
reorders to the author PBLORA order `[harmlessness, helpfulness]` when setting
`pref_vec`.

## Regression Results

```text
PARM-TARO tests:            18 passed
Router V2 Stage 2-6 tests: 120 passed
Combined:                  138 passed
```

The static-equivalence test evaluates the checked-in vendored
`model_arithmetic.operators` raw sum and matches the PARM-TARO equation.
The adaptive end-to-end test verifies alpha reaches both the PBLORA guide and
Router V2, models are frozen, router inputs are detached, and history is causal.

Protected `PARM/` snapshot before and after Stage 7:

```text
file_count:   156
total_bytes:  1,506,962
tree_sha256:  dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
unchanged:    true
```

The existing Router V2 protected-tree tests also passed for V1, PARM, RAD, and
legacy cache boundaries.

## Local Inventory

Available prerequisites:

```text
base model: models/tulu-2-7b
PKU source: dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz
PKU SHA256: f5f42f6f08a1fadbc3fd7c36e627f651782ce0f2f8178361483defa62b79ca06
vendored PARM dependencies: resolve in genarm environment
```

Missing prerequisites:

```text
results/parm_taro/checkpoints/parm_pku_pblora/adapter_config.json
results/parm_taro/checkpoints/parm_pku_pblora/adapter weights
results/router_v2/training/alpha_parm/best.pt
```

Existing `models/genarm-tulu2-hh` is ordinary `LORA`, not PBLORA, and is HH
rather than the required two-objective PKU PARM checkpoint. Existing Router V2
training checkpoints are GPT-2/RAD checkpoints with `use_preference = false`.

Machine-readable details are in `stage7_prerequisite_audit.json`.

## Host Commands

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
export PY=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Unit/regression tests.
"$PY" -m unittest discover -s PARM_TARO/tests -v

# Returns exit code 2 until both production checkpoints exist.
"$PY" -m PARM_TARO.scripts.audit_prerequisites

# Run only after the audit reports READY.
"$PY" -m PARM_TARO.scripts.run_adaptive \
  --config PARM_TARO/configs/parm_taro_tulu2.json \
  --device cuda --no-cpu-fallback \
  --prompt "Explain how to stay safe while helping a user."

# GPU monitoring during the production smoke.
watch -n 1 nvidia-smi
```

The config's checkpoint paths and `router_tokenizer_semantic_sha256` must be
updated to the real trained artifacts before the production command can pass.
