#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PY="${TTA_PYTHON:-$(command -v python3)}"
HF="${HF_CLI:-$(dirname "$PY")/hf}"
[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
[[ -x "$HF" ]] || { echo "Set HF_CLI to an executable Hugging Face CLI" >&2; exit 2; }
CHECK="$ROOT/results/parm_taro/recovery/reconstruction/scripts/check_downloads.py"
LOG_DIR="$ROOT/results/parm_taro/recovery/reconstruction/logs"
LOG="$LOG_DIR/download_public_artifacts.log"
mkdir -p "$LOG_DIR"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/models/.hf-control}"

exec > >(tee -a "$LOG") 2>&1
echo "download attempt UTC: $(date -u +%FT%TZ)"

# Three 7B snapshots need about 40 GiB; retain headroom for Hub temporary files.
"$PY" "$CHECK" --require-free-gib 55

download_if_needed() {
  local key="$1" repo="$2" revision="$3" target="$4"
  if "$PY" "$CHECK" --artifact "$key" >/dev/null 2>&1; then
    echo "verified, skip: $key"
    return 0
  fi
  echo "resume/download: $repo@$revision -> $target"
  "$HF" download "$repo" --revision "$revision" --local-dir "$target"
  "$PY" "$CHECK" --mark-revision "$key"
  "$PY" "$CHECK" --artifact "$key"
}

download_if_needed tulu allenai/tulu-2-7b \
  3c6e328ae91fabdd0daf09de16887de9615c1f66 models/tulu-2-7b
download_if_needed beaver_reward PKU-Alignment/beaver-7b-v1.0-reward \
  375cd6a9f0d7e339d2199b05ba129a4a8906596d models/beaver-7b-v1.0-reward
download_if_needed beaver_cost PKU-Alignment/beaver-7b-v1.0-cost \
  c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 models/beaver-7b-v1.0-cost

echo "Safe-RLHF is required by current Stage-10 scoring source, but its historical commit is unresolved."
echo "Refusing an unpinned clone; this does not fail the approved three-artifact public-core gate."
"$PY" "$CHECK" --public-core --write-report \
  results/parm_taro/recovery/public_restore
