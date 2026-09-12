# Stage 9 CUDA OOM Fix

## Root Cause

The shared-backbone branch already created one Python backbone object, but it
loaded that 7B object in FP16. `device_map="auto"` was allowed to consume the
configured 8 GiB before PBLORA attachment, and Accelerate then failed on a
20 MiB allocation with only 12 MiB free. The report's claimed 4-bit runtime had
not been implemented in the model-loading kwargs.

## Fix

- CUDA shared mode now constructs `BitsAndBytesConfig` with NF4, double
  quantization, FP16 compute, and `load_in_4bit=True`.
- `AutoModelForCausalLM.from_pretrained()` is called exactly once in shared
  mode, using `models/tulu-2-7b`.
- Vendored PARM `PeftModel.from_pretrained()` attaches the frozen PBLORA to that
  object.
- `FrozenBaseModelView` uses `disable_adapter()` for the base pass; the context
  restores PBLORA before the guide pass.
- The runtime rejects a CUDA model that does not report
  `is_loaded_in_4bit=True`.
- Logit positions are selected in model dtype before conversion to FP32. This
  avoids a transient FP32 `[sequence_length, vocab_size]` allocation.
- A CUDA smoke performs real base/PBLORA forwards and Router backward, checks
  one physical backbone, adapter restoration, distribution difference, frozen
  gradients, Router gradients, hashes, and peak VRAM.

Expected RTX 3080 peak: approximately 5-7 GiB. The measured source of truth is
`cuda.peak_reserved_gib` in the smoke report.

## Protected Artifacts

```text
PARM tree SHA-256 = dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
PBLORA weights SHA-256 = 1012790985d4eeb5229efa75a5cdee124fdf526ff226ddb239d1a2d29018ac93
```

Both hashes were unchanged after implementation and CPU tests.

CPU regression result:

```text
router_v2/tests = 120 passed
PARM_TARO/tests = 53 passed
total           = 173 passed
```

## Host Commands

CUDA smoke:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
set -euo pipefail
export PY=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/parm_taro/logs

"$PY" -m PARM_TARO.scripts.smoke_stage9_cuda \
  --config PARM_TARO/configs/train_stage9_taro.json \
  --max-samples 2 \
  --output results/parm_taro/smoke/stage9_cuda_taro.json \
  2>&1 | tee results/parm_taro/logs/stage9_cuda_taro_smoke.log

"$PY" -m json.tool results/parm_taro/smoke/stage9_cuda_taro.json
```

Full TARO training after smoke `PASS`:

```bash
"$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_taro.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/parm_taro/logs/stage9_taro_4bit.log
```

## Status

```text
HOST CUDA SMOKE REQUIRED
```
