#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$SCRIPT_DIR/gpu_env.sh"
resolve_gpu_python EVALUATION_PYTHON
PY="$GPU_PYTHON_BIN"
gpu_env_preflight "$PY" false false
stage=""
resume=false
skip_existing=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --resume) resume=true; shift ;;
    --skip-existing|--skip_existing) skip_existing=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$stage" in
  val200) count=200 ;;
  val500) count=500 ;;
  test1500) count=1500 ;;
  *) echo "--stage must be val200, val500, or test1500" >&2; exit 2 ;;
esac
config="$ROOT/PARM_TARO/recovery/configs/evaluate_${stage}.json"
[[ -f "$config" ]] || {
  echo "Frozen staged config is not yet authorized/present: $config" >&2
  exit 2
}
if $skip_existing && ! $resume; then
  echo "--skip-existing requires --resume so the existing journal is opened safely" >&2
  exit 2
fi
args=()
if $resume; then args+=(--resume); fi
cd "$ROOT"
# The engine's append-only journal deduplicates the tuple
# (phase, prompt_id, alpha_index, method, seed), so --resume is skip-existing.
exec "$PY" -m PARM_TARO.scripts.evaluate_stage10 \
  --config "$config" --phase full --device cuda --no-cpu-fallback \
  --max-prompts "$count" "${args[@]}"
