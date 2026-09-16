#!/usr/bin/env bash
# Shared environment selection/preflight for recovery GPU jobs.
# Source this file; it intentionally performs no work by itself.

resolve_gpu_python() {
  local preferred_variable="${1:?preferred environment variable is required}"
  local configured="${!preferred_variable:-${TTA_GPU_PYTHON:-}}"

  if [[ -z "$configured" ]]; then
    configured="$(command -v python 2>/dev/null || true)"
  fi
  [[ -n "$configured" && -x "$configured" ]] || {
    echo "GPU Python is unavailable. Activate genarm or set $preferred_variable." >&2
    return 2
  }

  GPU_PYTHON_BIN="$(readlink -f "$configured")"
  if [[ "$GPU_PYTHON_BIN" == *'/envs/tta/'* ]]; then
    echo "Refusing CPU-only tta interpreter for a GPU job: $GPU_PYTHON_BIN" >&2
    echo "Activate genarm or set $preferred_variable explicitly." >&2
    return 2
  fi
}

gpu_env_preflight() {
  local python_bin="${1:?python executable is required}"
  local require_dpo="${2:-false}"
  local allow_no_cuda="${3:-false}"

  "$python_bin" - "$require_dpo" "$allow_no_cuda" <<'PY'
import importlib
import json
import sys

require_dpo = sys.argv[1].lower() == "true"
allow_no_cuda = sys.argv[2].lower() == "true"

try:
    import torch
    import transformers
    import peft
    import trl
    import bitsandbytes
except Exception as exc:
    print(f"GPU_ENV_PREFLIGHT_FAIL: required import failed: {exc!r}", file=sys.stderr)
    raise SystemExit(20)

cuda_available = bool(torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
payload = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu_name": gpu_name,
    "trl_version": getattr(trl, "__version__", "UNKNOWN"),
    "transformers_version": getattr(transformers, "__version__", "UNKNOWN"),
    "peft_version": getattr(peft, "__version__", "UNKNOWN"),
    "bitsandbytes_version": getattr(bitsandbytes, "__version__", "UNKNOWN"),
}
print(json.dumps(payload, sort_keys=True))

if require_dpo:
    try:
        from trl import DPOConfig, DPOTrainer  # noqa: F401
    except Exception as exc:
        print(f"GPU_ENV_PREFLIGHT_FAIL: TRL DPO import failed: {exc!r}", file=sys.stderr)
        raise SystemExit(21)
    print("TRL DPO import PASS")

if not cuda_available:
    if allow_no_cuda:
        print("GPU_ENV_PREFLIGHT_DRY_RUN: CUDA is hidden from this process; real execution remains fail-closed.")
    else:
        print("GPU_ENV_PREFLIGHT_FAIL: torch.cuda.is_available() is false", file=sys.stderr)
        raise SystemExit(22)
elif "5090" not in gpu_name:
    print(f"GPU_ENV_PREFLIGHT_FAIL: expected RTX 5090, got {gpu_name!r}", file=sys.stderr)
    raise SystemExit(23)
else:
    print("GPU_ENV_PREFLIGHT_PASS")
PY
}
