#!/usr/bin/env bash
set -u

PROJECT="/home/jupyter-iec2024se10/Test-time-Alignment"
GENARM_PYTHON="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"

MIN_FREE_MIB="${MIN_FREE_MIB:-14336}"      # 14 GiB
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"     # seconds
STABLE_CHECKS="${STABLE_CHECKS:-3}"

LOG="$PROJECT/results/parm_taro/recovery/pblora_repro/smoke.log"
PID_FILE="$PROJECT/results/parm_taro/recovery/pblora_repro/smoke.pid"
WAIT_LOG="$PROJECT/results/parm_taro/recovery/pblora_repro/gpu_wait.log"

mkdir -p "$(dirname "$LOG")"
cd "$PROJECT" || exit 1

stable=0

echo "[$(date)] Waiting for GPU 0: >= ${MIN_FREE_MIB} MiB free" | tee -a "$WAIT_LOG"

while true; do
    FREE_MIB=$(nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        -i 0 | head -n1 | tr -d ' ')

    UTIL=$(nvidia-smi \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits \
        -i 0 | head -n1 | tr -d ' ')

    echo "[$(date)] free=${FREE_MIB}MiB util=${UTIL}% stable=${stable}/${STABLE_CHECKS}" \
        | tee -a "$WAIT_LOG"

    if [ "$FREE_MIB" -ge "$MIN_FREE_MIB" ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi

    if [ "$stable" -ge "$STABLE_CHECKS" ]; then
        echo "[$(date)] GPU ready. Starting PBLoRA smoke." | tee -a "$WAIT_LOG"
        break
    fi

    sleep "$CHECK_INTERVAL"
done

export PBLORA_PYTHON="$GENARM_PYTHON"
export PBLORA_SEED=42
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash PARM_TARO/recovery/scripts/train_pblora_repro.sh \
    --mode smoke \
    >> "$LOG" 2>&1

EXIT_CODE=$?

echo "[$(date)] PBLoRA smoke exited code=$EXIT_CODE" | tee -a "$WAIT_LOG"

exit "$EXIT_CODE"
