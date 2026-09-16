#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$SCRIPT_DIR/gpu_env.sh"
resolve_gpu_python FEASIBILITY_PYTHON
PY="$GPU_PYTHON_BIN"
gpu_env_preflight "$PY" false false

cd "$ROOT"
exec "$PY" PARM_TARO/recovery/run_feasibility.py --device cuda "$@"
