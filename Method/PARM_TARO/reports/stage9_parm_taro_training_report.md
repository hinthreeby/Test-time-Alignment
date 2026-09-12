# Stage 9 PARM-TARO Router Training Report

## Status

```text
CUDA SMOKE PASS
TOKENIZATION BOUNDARY FIX PASS
TARO PASS
V2 NO-ALPHA PASS
V2 ALPHA-PREFERENCE PILOT V3 PASS
FULL 8,000-EXAMPLE ALPHA-PREFERENCE PRODUCTION RUN COMPLETE
STAGE 9 NOT PASS - read-only non-identical constant-control audit pending
```

The one-backbone NF4 CUDA smoke, TARO training, V2 no-alpha training, and the
preference-aware Pilot V3 have passed. The full production run completed with
all checks passing except the original all-task constant-alpha control. Its
constant is `[0.5, 0.5]`, which is exactly equal to one of the five validation
grid values. The production checkpoint and NOT_PASS `run_status.json` remain
unchanged while a separate full-validation audit tests the conditional
non-identical control statistic at the same `0.001` threshold.

## Implemented Contract

- Train order is enforced as `taro -> v2_no_alpha -> v2_alpha`.
- Base LM, PARM backbone, and PBLORA are frozen and execute under
  `torch.inference_mode()`.
- One shared NF4 4-bit backbone is used on CUDA. The base pass uses
  `PeftModel.disable_adapter()` and the PARM pass enables PBLORA. Shared mode
  calls `AutoModelForCausalLM.from_pretrained()` exactly once.
- Base/PARM full-vocabulary logits are converted to detached FP32 log
  probabilities. Independent Top-K sets are Router inputs only.
- Every optimizer parameter must belong to the Router and must be FP32. Frozen
  model gradients are asserted absent and model/PARM hashes are compared before
  and after each run.
- Public alpha order is `[helpfulness, harmlessness]`; only the PBLORA boundary
  reorders it to the author order `[harmlessness, helpfulness]`.
- Alpha is sampled deterministically from `SHA256(seed, epoch, sample_id)`, so
  batching and resume do not change sampled preferences.
- Validation uses only the fixed helpfulness grid
  `[0, 0.25, 0.5, 0.75, 1]`. The test split cannot be loaded by Stage 9 code.
- Atomic `latest.pt` and `best.pt` checkpoints include optimizer state,
  provenance, tokenizer hash, objective, alpha orders, and Stage 8 manifest
  hash. Existing non-resume output is never overwritten.

## Objective

For response `r` and teacher-forced token `t`:

```text
b_t        = log_softmax(z_base_t)
p_t(alpha) = log_softmax(z_PBLORA(alpha)_t)
log pi_t   = log_softmax(b_t + lambda_t * p_t(alpha))
NLL(r)     = mean_t[-log pi_t(y_t^r)]
```

The response weight is:

```text
q_r(alpha) = alpha_help * 1[r = better_response_id]
           + alpha_safe * 1[r = safer_response_id]

L(alpha) = q_0(alpha) * NLL(response_0)
         + q_1(alpha) * NLL(response_1)
```

Thus agreeing objectives yield one preferred response; conflicting objectives
retain both responses; exact and near ties are logged. The configured rule is
`alpha_weighted_response_mean_nll`.

## Comparisons And Controls

Each stage reports validation token NLL and lambda count/mean/std/min/max plus
lambda-alpha correlation for all available methods:

```text
static_parm
taro
v2_no_alpha
v2_alpha
v2_alpha_shuffled
v2_alpha_constant
```

For shuffled/constant controls, the PBLORA guide remains conditioned on the
correct alpha and only the Router alpha is altered. PASS requires shuffled and
constant alpha to change lambda; the production config also requires shuffled
alpha to degrade validation loss.

Production Router parameter counts:

| Stage | Parameters |
|---|---:|
| TARO | 1,193,217 |
| Smart V2 no-alpha | 1,079,449 |
| Smart V2 alpha | 1,081,849 |

## Files Added Or Updated

- `PARM_TARO/training/{alpha,config,data,engine,objective,online,routing,runtime}.py`
- `PARM_TARO/scripts/{audit_stage9_training,train_parm_router}.py`
- `PARM_TARO/scripts/smoke_stage9_cuda.py`
- `PARM_TARO/configs/train_stage9_{taro,v2_no_alpha,v2_alpha}.json`
- `PARM_TARO/configs/router_{taro,v2_no_alpha,v2_alpha}_tulu2.json`
- `PARM_TARO/schemas/stage9_training_config.schema.json`
- `PARM_TARO/tests/test_stage9_training.py`
- `PARM_TARO/adapters/parm_adapter.py`
- `PARM_TARO/README.md`
- `PARM_TARO/reports/stage9_prerequisite_audit.json`
- `PARM_TARO/reports/stage9_cuda_oom_fix_report.md`

No file under `PARM/`, `router/`, `Method/RAD/`, or a V1/Stage 3 cache was
modified.

## Verification

CPU suites:

```text
router_v2/tests: 120 passed
PARM_TARO/tests: 91 passed
total: 211 passed
```

The CPU suite includes a complete tiny one-epoch training/checkpoint run,
dual-response loss checks, alpha controls, causal history, frozen-gradient and
optimizer boundaries, shared-backbone adapter disable/restore, no test loading,
resume-stable alpha, and protected-tree regression.

Current protected PARM snapshot:

```text
file_count  = 156
total_bytes = 1,506,962
tree_sha256 = dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
```

Current prerequisite audit:

```text
base_model_present                 = true
tokenizer_present                  = true
stage8_data_valid                  = true
router_config_present              = true
parm_tree_matches_baseline         = true
frozen_two_objective_pblora_present = true
shared_backbone_compatible          = true
status                              = READY
```

PBLORA checkpoint protection:

```text
adapter_config.json SHA-256 = a32ef80e2d487abee84a10cf1d87ea9da2aa15636f4eec61050f35c08f4aa69e
adapter_model.safetensors SHA-256 = 1012790985d4eeb5229efa75a5cdee124fdf526ff226ddb239d1a2d29018ac93
```

## Historical Host Workflow

Set the environment and verify CUDA:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
set -euo pipefail
export ROOT="$PWD"
export PY=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p results/parm_taro/logs

nvidia-smi
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

Audit the checkpoint. Exit code must be zero before Router training:

```bash
"$PY" -m PARM_TARO.scripts.audit_stage9_training \
  --config PARM_TARO/configs/train_stage9_taro.json \
  --output PARM_TARO/reports/stage9_prerequisite_audit.json \
  2>&1 | tee results/parm_taro/logs/stage9_prerequisite_audit.log
```

Run the bounded production-dimension CUDA smoke before training. It executes
two teacher-forced samples and a Router backward pass without writing a Router
checkpoint:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m PARM_TARO.scripts.smoke_stage9_cuda \
  --config PARM_TARO/configs/train_stage9_taro.json \
  --max-samples 2 \
  --output results/parm_taro/smoke/stage9_cuda_taro.json \
  2>&1 | tee results/parm_taro/logs/stage9_cuda_taro_smoke.log
"$PY" -m json.tool results/parm_taro/smoke/stage9_cuda_taro.json
```

Expected RTX 3080 peak is approximately 5-7 GiB; the smoke JSON records actual
peak allocated and reserved VRAM. Do not start full training unless smoke
status is `PASS` and `physical_backbone_count` is `1`.

Train in the enforced order:

```bash
"$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_taro.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage9_taro_4bit.log

"$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_v2_no_alpha.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage9_v2_no_alpha.log

"$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_v2_alpha.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage9_v2_alpha.log
```

Resume an interrupted stage with the same config and overrides, for example:

```bash
"$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_v2_alpha.json \
  --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/parm_taro/logs/stage9_v2_alpha.log
```

Monitor and inspect results:

```bash
watch -n 2 nvidia-smi
tail -f results/parm_taro/logs/stage9_v2_alpha.log

for stage in taro v2_no_alpha v2_alpha; do
  "$PY" -m json.tool "results/parm_taro/training/$stage/run_status.json"
  sha256sum "results/parm_taro/training/$stage/best.pt"
done

"$PY" - <<'PY'
import json
from pathlib import Path
for stage in ("taro", "v2_no_alpha", "v2_alpha"):
    path = Path("results/parm_taro/training") / stage / "run_status.json"
    status = json.loads(path.read_text())
    print(stage, status["status"], status["checks"])
    assert status["status"] == "PASS"
PY
```

Return these files after host execution:

```text
PARM_TARO/reports/stage9_prerequisite_audit.json
results/parm_taro/logs/pblora_train.log
results/parm_taro/logs/stage9_cuda_taro_smoke.log
results/parm_taro/smoke/stage9_cuda_taro.json
results/parm_taro/logs/stage9_taro_4bit.log
results/parm_taro/logs/stage9_v2_no_alpha.log
results/parm_taro/logs/stage9_v2_alpha.log
results/parm_taro/training/taro/run_status.json
results/parm_taro/training/v2_no_alpha/run_status.json
results/parm_taro/training/v2_alpha/run_status.json
```

## Alpha-Preference Production Promotion

The production run is isolated from and does not overwrite the failed legacy
`v2_alpha` run or any pilot. It starts deterministically from the same Pilot V2
source and a fresh V3 residual, preserving the validated initialization near
lambda 0.03 without using validation-selected pilot weights as a warm start.

The production configuration locks:

```text
train examples                 = 8000
validation examples            = 500
test examples                  = 0
warm-up                        = alpha path only
quality phase                  = full Router + calibrated NLL
min control lambda delta       = 0.001
checkpoint interval            = 100 optimizer steps
output                         = results/parm_taro/training/v2_alpha_preference/
```

Atomic `latest.pt` checkpoints contain the phase, epoch, next deterministic
sample index, optimizer state, accumulated metrics, calibration, validation
baseline, and protected-artifact snapshot. Resume is accepted only when the
config, tokenizer, Pilot V3 authorization, model/PBLORA/data hashes, and all
previous Stage 9 run hashes still match.

The final gate accepts only:

```text
TARO PASS
V2 no-alpha PASS
V2 alpha-preference production PASS
```

Pilot V1/V2/V3 and the failed legacy `v2_alpha` remain read-only evidence. The
current final-gate status is recorded in
`PARM_TARO/reports/stage9_final_gate.json` and is
`STAGE 9 NOT PASS` until the read-only constant-control audit completes.

Do not retrain or resume production for this control audit. The immutable
checkpoint under audit is:

```text
results/parm_taro/training/v2_alpha_preference/final.pt
SHA-256 = 56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20
```

Run only the full-validation audit:

```bash
"$PY" -m PARM_TARO.scripts.audit_constant_alpha_control \
  --config PARM_TARO/configs/train_stage9_v2_alpha_preference.json \
  --device cuda --no-cpu-fallback \
  --bootstrap-samples 2000 \
  --output PARM_TARO/reports/stage9_constant_alpha_control_audit.json \
  2>&1 | tee results/parm_taro/logs/stage9_constant_alpha_control_audit.log
```

After the read-only constant-control audit returns `PASS`, materialize the final gate:

```bash
"$PY" -m PARM_TARO.scripts.finalize_stage9 \
  --config PARM_TARO/configs/train_stage9_v2_alpha_preference.json \
  --output PARM_TARO/reports/stage9_final_gate.json \
  --overwrite \
  2>&1 | tee results/parm_taro/logs/stage9_final_gate.log
```

## Constant-Control Audit

The completed production run reported:

```text
all-task correct-vs-constant delta = 0.0008731123345
correct-vs-shuffled delta          = 0.0010523890413
endpoint-alpha delta               = 0.0026256577858
threshold                         = 0.001
```

The validation grid has 2,500 tasks, of which 500 use the same `[0.5, 0.5]`
alpha as the constant control. Those comparisons are structurally
non-informative and must produce zero delta. Because every alpha grid point
uses the same examples and token counts, the aggregate implies a preliminary
non-identical token-weighted estimate of `0.0010913904181`. This estimate does
not authorize PASS by itself.

`audit_constant_alpha_control` re-runs the immutable `final.pt` over all 500
validation examples and reports all-task, non-identical, and per-alpha deltas;
task-level median/quartiles; a deterministic 95% bootstrap CI; and endpoint
sensitivity. It hashes the checkpoint, original NOT_PASS run status, complete
production output tree, frozen base/PBLORA, PARM tree, data, and predecessor
runs before and after. It never loads the test split and writes only under
`PARM_TARO/reports/`.

The final gate accepts the correction only when the original production run
failed exactly the constant-control check and the recomputed non-identical
token-weighted mean passes the unchanged `0.001` threshold. Otherwise Stage 9
remains NOT PASS.
