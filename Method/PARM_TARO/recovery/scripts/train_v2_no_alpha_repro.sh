#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$SCRIPT_DIR/gpu_env.sh"
resolve_gpu_python V2_PYTHON
PY="$GPU_PYTHON_BIN"
gpu_env_preflight "$PY" false false
OUT="$ROOT/results/parm_taro/recovery/reproduced/v2_no_alpha"
mkdir -p "$OUT"
[[ -e "$OUT/last.pt" ]] || ln -s latest.pt "$OUT/last.pt"
resume=()
if [[ "${1:-}" == "--resume" ]]; then resume=(--resume); shift; fi
cd "$ROOT"
exec "$PY" -m PARM_TARO.scripts.train_parm_router \
  --config PARM_TARO/configs/train_stage9_v2_no_alpha.json --device cuda \
  --no-cpu-fallback --output-dir "$OUT" "${resume[@]}" "$@"
