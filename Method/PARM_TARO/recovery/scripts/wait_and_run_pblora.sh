#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
TRAIN_WRAPPER="$ROOT/PARM_TARO/recovery/scripts/train_pblora_repro.sh"
REPORT_DIR="$ROOT/results/parm_taro/recovery/pblora_repro"
GPU_WAIT_LOG="$REPORT_DIR/gpu_wait.log"
WATCHER_LOG="$REPORT_DIR/watcher.log"
WATCHER_PID_FILE="$REPORT_DIR/watcher.pid"
LAST_ATTEMPT_LOG="$REPORT_DIR/last_attempt.log"
LOCK_FILE="$REPORT_DIR/watcher.lock"

MIN_FREE_MIB="${MIN_FREE_MIB:-14336}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"
OOM_BACKOFF="${OOM_BACKOFF:-120}"
PBLORA_PYTHON="${PBLORA_PYTHON:-/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10}"
PBLORA_SEED="${PBLORA_SEED:-42}"
MODE="smoke"
DRY_RUN=false
STATUS_ONLY=false
FORWARD_ARGS=()
TRAINING_PID=""

usage() {
  cat <<'EOF'
Usage: wait_and_run_pblora.sh [OPTIONS] [-- extra PBLoRA wrapper arguments]

  --mode smoke|one-epoch|full
  --dry-run       validate configuration; do not poll, wait, or train
  --status        show watcher PID, current GPU telemetry, and latest log tails

Environment overrides:
  MIN_FREE_MIB=14336  CHECK_INTERVAL=60  STABLE_CHECKS=3  OOM_BACKOFF=120
  PBLORA_PYTHON=...   PBLORA_SEED=42
EOF
}

while (($#)); do
  case "$1" in
    --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --status) STATUS_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; FORWARD_ARGS+=("$@"); break ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

case "$MODE" in smoke|one-epoch|full) ;; *) echo "Invalid --mode: $MODE" >&2; exit 2 ;; esac
for value_name in MIN_FREE_MIB CHECK_INTERVAL STABLE_CHECKS OOM_BACKOFF; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$value_name must be a non-negative integer" >&2; exit 2; }
done
(( MIN_FREE_MIB > 0 && CHECK_INTERVAL > 0 && STABLE_CHECKS > 0 )) || {
  echo "MIN_FREE_MIB, CHECK_INTERVAL and STABLE_CHECKS must be positive" >&2
  exit 2
}

mkdir -p "$REPORT_DIR"

timestamp() { date --iso-8601=seconds; }

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

gpu_query() {
  nvidia-smi --id=0 \
    --query-gpu=memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

show_status() {
  if [[ -s "$WATCHER_PID_FILE" ]]; then
    watcher_pid="$(tr -cd '0-9' < "$WATCHER_PID_FILE")"
    if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null; then
      echo "watcher: RUNNING pid=$watcher_pid"
      ps -fp "$watcher_pid" || true
    else
      echo "watcher: NOT RUNNING (stale pid file: ${watcher_pid:-invalid})"
    fi
  else
    echo "watcher: NOT RUNNING"
  fi
  echo "gpu0:"
  gpu_query || echo "nvidia-smi query unavailable"
  echo "gpu_wait.log (tail):"
  tail -n 8 "$GPU_WAIT_LOG" 2>/dev/null || echo "not created"
  echo "last_attempt.log (tail):"
  tail -n 12 "$LAST_ATTEMPT_LOG" 2>/dev/null || echo "not created"
}

if $STATUS_ONLY; then
  show_status
  exit 0
fi

[[ -x "$PBLORA_PYTHON" ]] || { echo "PBLORA_PYTHON is not executable: $PBLORA_PYTHON" >&2; exit 2; }
[[ -x "$TRAIN_WRAPPER" ]] || { echo "PBLoRA training wrapper is not executable: $TRAIN_WRAPPER" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is unavailable" >&2; exit 2; }

export PBLORA_PYTHON
export PBLORA_SEED
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

checkpoint_output() {
  if [[ -n "${PBLORA_OUTPUT_DIR:-}" ]]; then
    printf '%s\n' "$PBLORA_OUTPUT_DIR"
  elif [[ "$MODE" == smoke ]]; then
    printf '%s\n' "$ROOT/results/parm_taro/recovery/reproduced/pblora_smoke"
  else
    printf '%s\n' "$ROOT/results/parm_taro/recovery/reproduced/pblora"
  fi
}

latest_valid_checkpoint() {
  local output latest candidate
  output="$(checkpoint_output)"
  latest=""
  if [[ -L "$output/last" && -f "$output/last/trainer_state.json" ]]; then
    readlink -f "$output/last"
    return 0
  fi
  while IFS= read -r candidate; do
    [[ -f "$candidate/trainer_state.json" ]] && latest="$candidate"
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

build_command() {
  local checkpoint
  RUN_COMMAND=(bash "$TRAIN_WRAPPER" --mode "$MODE")
  checkpoint="$(latest_valid_checkpoint || true)"
  if [[ -n "$checkpoint" ]]; then
    RUN_COMMAND+=(--resume)
    RESUME_DECISION="resume from $checkpoint"
  else
    RESUME_DECISION="fresh start (no valid checkpoint)"
  fi
  RUN_COMMAND+=("${FORWARD_ARGS[@]}")
}

build_command
if $DRY_RUN; then
  echo "SHARED_GPU_WATCHER_DRY_RUN"
  echo "project_root=$ROOT"
  echo "mode=$MODE"
  echo "python=$(readlink -f "$PBLORA_PYTHON")"
  echo "min_free_mib=$MIN_FREE_MIB check_interval=$CHECK_INTERVAL stable_checks=$STABLE_CHECKS oom_backoff=$OOM_BACKOFF"
  echo "resume_decision=$RESUME_DECISION"
  printf 'launch_command:'
  printf ' %q' "${RUN_COMMAND[@]}"
  printf '\n'
  echo "No polling, sleeping, model loading, or training was performed."
  exit 0
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another PBLoRA watcher holds $LOCK_FILE; refusing duplicate watcher." >&2
  exit 4
fi

cleanup() {
  status=$?
  if [[ -n "$TRAINING_PID" ]] && kill -0 "$TRAINING_PID" 2>/dev/null; then
    kill -SIGTERM "$TRAINING_PID" 2>/dev/null || true
    wait "$TRAINING_PID" 2>/dev/null || true
  fi
  current_pid="$(tr -cd '0-9' < "$WATCHER_PID_FILE" 2>/dev/null || true)"
  if [[ "$current_pid" == "$$" ]]; then
    rm -f "$WATCHER_PID_FILE"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

pid_tmp="$WATCHER_PID_FILE.tmp.$$"
printf '%s\n' "$$" > "$pid_tmp"
mv -f "$pid_tmp" "$WATCHER_PID_FILE"
touch "$WATCHER_LOG" "$GPU_WAIT_LOG"

log "watcher started pid=$$ mode=$MODE threshold=${MIN_FREE_MIB}MiB stable_required=$STABLE_CHECKS"
attempt=0
stable=0

while true; do
  telemetry="$(gpu_query 2>/dev/null || true)"
  if [[ -z "$telemetry" ]]; then
    stable=0
    line="$(timestamp), free_mib=UNKNOWN, used_mib=UNKNOWN, utilization_pct=UNKNOWN, stable=0/$STABLE_CHECKS, status=NVIDIA_SMI_UNAVAILABLE"
    printf '%s\n' "$line" | tee -a "$GPU_WAIT_LOG"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  IFS=',' read -r free_mib used_mib utilization_pct <<< "$telemetry"
  free_mib="${free_mib//[[:space:]]/}"
  used_mib="${used_mib//[[:space:]]/}"
  utilization_pct="${utilization_pct//[[:space:]]/}"
  if [[ ! "$free_mib" =~ ^[0-9]+$ || ! "$used_mib" =~ ^[0-9]+$ || ! "$utilization_pct" =~ ^[0-9]+$ ]]; then
    stable=0
    line="$(timestamp), free_mib=$free_mib, used_mib=$used_mib, utilization_pct=$utilization_pct, stable=0/$STABLE_CHECKS, status=PARSE_ERROR"
    printf '%s\n' "$line" | tee -a "$GPU_WAIT_LOG"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  if (( free_mib >= MIN_FREE_MIB )); then
    ((stable += 1))
  else
    stable=0
  fi
  line="$(timestamp), free_mib=$free_mib, used_mib=$used_mib, utilization_pct=$utilization_pct, stable=$stable/$STABLE_CHECKS"
  printf '%s\n' "$line" | tee -a "$GPU_WAIT_LOG"

  if (( stable < STABLE_CHECKS )); then
    sleep "$CHECK_INTERVAL"
    continue
  fi

  ((attempt += 1))
  build_command
  log "attempt=$attempt launching; $RESUME_DECISION"
  : > "$LAST_ATTEMPT_LOG"
  set +e
  "${RUN_COMMAND[@]}" > >(tee "$LAST_ATTEMPT_LOG") 2>&1 &
  TRAINING_PID=$!
  wait "$TRAINING_PID"
  training_status=$?
  TRAINING_PID=""
  set -e

  if (( training_status == 0 )); then
    log "attempt=$attempt completed successfully"
    exit 0
  fi

  if grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$LAST_ATTEMPT_LOG"; then
    log "attempt=$attempt exited=$training_status with CUDA OOM; waiting ${OOM_BACKOFF}s before resuming GPU polling"
    stable=0
    sleep "$OOM_BACKOFF"
    continue
  fi

  log "attempt=$attempt exited=$training_status with non-OOM failure; watcher stopping"
  exit "$training_status"
done
