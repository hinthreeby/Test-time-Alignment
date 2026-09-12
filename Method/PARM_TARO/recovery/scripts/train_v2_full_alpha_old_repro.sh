#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PY="${TTA_PYTHON:-$(command -v python3)}"
[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
OUT="$ROOT/results/parm_taro/recovery/reproduced/v2_alpha_preference"
CONFIG="${V2_OLD_REPRO_CONFIG:-$ROOT/PARM_TARO/recovery/configs/train_v2_full_alpha_old_repro.json}"
[[ -f "$CONFIG" ]] || {
  echo "Resolved recovery config is absent: $CONFIG" >&2
  echo "Pilot/checkpoint provenance must be reconstructed before this job is authorized." >&2
  exit 2
}
mkdir -p "$OUT"
[[ -e "$OUT/last.pt" ]] || ln -s latest.pt "$OUT/last.pt"
[[ -e "$OUT/best.pt" ]] || ln -s final.pt "$OUT/best.pt"
resume=()
if [[ "${1:-}" == "--resume" ]]; then resume=(--resume); shift; fi
cd "$ROOT"
exec "$PY" -m PARM_TARO.scripts.train_alpha_preference_production \
  --config "$CONFIG" --device cuda --no-cpu-fallback "${resume[@]}" "$@"
