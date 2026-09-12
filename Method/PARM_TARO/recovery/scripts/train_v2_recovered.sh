#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PY="${TTA_PYTHON:-$(command -v python3)}"
[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
ENTRY="$ROOT/PARM_TARO/recovery/train_recovered_router.py"
CONFIG="${RECOVERED_CONFIG:-$ROOT/PARM_TARO/recovery/configs/train_v2_recovered.json}"
[[ -f "$ENTRY" && -f "$CONFIG" ]] || {
  echo "Recovered training remains gated until fresh D1-D7 confirms the fix family." >&2
  echo "Expected later: $ENTRY and $CONFIG" >&2
  exit 2
}
resume=()
if [[ "${1:-}" == "--resume" ]]; then resume=(--resume); shift; fi
cd "$ROOT"
exec "$PY" "$ENTRY" --config "$CONFIG" --device cuda --no-cpu-fallback \
  "${resume[@]}" "$@"
