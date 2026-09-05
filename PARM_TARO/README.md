# PARM-TARO

This package integrates preference-aware PARM with Smart Router V2 while
keeping `PARM/` and `router_v2/` read-only.

The adaptive decoding equation is:

```text
log pi_tilde_t = normalize(log pi_base_t + lambda_t * log pi_PARM(alpha)_t)
```

`lambda_t` is produced by a preference-enabled Router V2 from independent
base/PARM Top-K log-probability features. It is not the RAD interpolation
`base + lambda * (guide - base)`.

Public Router/data alpha uses `[helpfulness, harmlessness]`. The wrapper
reorders it to `[harmlessness, helpfulness]` only when setting PARM's PBLORA
`pref_vec`, matching the original author code.

The local PARM implementation uses a static unit coefficient at inference
(`M_base + M_reward`). Training `beta` values belong to the ARM objective; no
literal inference `1 / beta` exists in the checked-in generation script.
`static_reference_scale` therefore defaults to `1.0` for regression.

Run the lightweight prerequisite audit with the environment that contains the
dependencies required by PARM's vendored model-arithmetic package:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python \
  -m PARM_TARO.scripts.audit_prerequisites
```

Production generation intentionally fails until both a two-objective PBLORA
checkpoint and a tokenizer-compatible preference-enabled Router V2 checkpoint
exist at the paths in `configs/parm_taro_tulu2.json`.

## Multi-Objective Data

The local PKU source is prepared without downloading or relabeling:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python \
  -m PARM_TARO.scripts.prepare_multi_objective_data --validate-only
```

Outputs are isolated under `dataset/parm_taro/`. The deterministic split keeps
all rows with the same prompt in one split to prevent prompt leakage.

## Stage 9 Router Training

Stage 9 trains in this fixed order:

```text
TARO -> Smart V2 no-alpha -> V3 alpha-preference production
```

The original NLL-only Smart V2 alpha run is retained as a failed collapse
diagnostic. Production uses the Pilot V3 normalized alpha/state residual,
alpha-path-only warm-up, and the calibrated preference-plus-quality objective.
It writes only to `results/parm_taro/training/v2_alpha_preference/` and supports
atomic resume with:

```bash
/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python \
  -m PARM_TARO.scripts.train_alpha_preference_production \
  --config PARM_TARO/configs/train_stage9_v2_alpha_preference.json \
  --device cuda --no-cpu-fallback --resume
```

The teacher-forced objective retains both candidate responses. For canonical
`alpha=[helpfulness, harmlessness]`, each response receives the sum of the
objective weights that prefer it; conflicts therefore remain a soft two-target
objective and ties do not require an arbitrary gold response.

The base distribution and alpha-conditioned PBLORA distribution are frozen.
Only FP32 Router parameters enter the optimizer. On constrained GPUs the
runtime loads one NF4 4-bit frozen backbone: the base pass runs with the adapter
disabled, then the PARM pass runs with PBLORA enabled. Full-vocabulary logits
are transient and detached before Router training.

Audit the required two-objective PBLORA checkpoint before training:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python \
  -m PARM_TARO.scripts.audit_stage9_training
```

See `reports/stage9_parm_taro_training_report.md` for the complete host workflow
and current prerequisite status.

The completed alpha-preference production checkpoint is audited without
retraining when the constant control overlaps the validation grid midpoint:

```bash
/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python \
  -m PARM_TARO.scripts.audit_constant_alpha_control \
  --config PARM_TARO/configs/train_stage9_v2_alpha_preference.json \
  --device cuda --no-cpu-fallback --bootstrap-samples 2000
```

The original all-task statistic remains in the report. Only comparisons where
the correct alpha differs from `[0.5, 0.5]` enter the corrected materiality
gate, and its threshold remains `0.001`.
