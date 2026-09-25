#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX:-/home/jupyter-iec2024se10/.conda/envs/genarm}/bin/python}
PIPELINE_MODE=${PIPELINE_MODE:-inference-tune}

case "$PIPELINE_MODE" in
  inference-tune)
    RUN_TAG=${RUN_TAG:-cura_inference_tune_v1}
    NUM_PROMPTS=${NUM_PROMPTS:-250}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
    BASE_DEVICE=${BASE_DEVICE:-auto}
    SIGNAL_DEVICE=${SIGNAL_DEVICE:-cpu}
    ;;
  full-train)
    RUN_TAG=${RUN_TAG:-cura_astar_full_v1}
    NUM_PROMPTS=${NUM_PROMPTS:-1000}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
    BASE_DEVICE=${BASE_DEVICE:-cuda}
    SIGNAL_DEVICE=${SIGNAL_DEVICE:-cuda}
    ;;
  *)
    echo "Unknown PIPELINE_MODE='$PIPELINE_MODE' (expected inference-tune or full-train)." >&2
    exit 2
    ;;
esac

CONFIG=${CONFIG:-Method/CURA/configs/sentiment_astar.json}
SOURCE_DIR=${SOURCE_DIR:-dataset/cura_astar_source}
CACHE_DIR=${CACHE_DIR:-dataset/${RUN_TAG}_cache}
CHECKPOINT=${CHECKPOINT:-Method/CURA/checkpoints/${RUN_TAG}.pt}
REPORT_DIR=${REPORT_DIR:-Method/CURA/reports/${RUN_TAG}}
BENCHMARK=${BENCHMARK:-dataset/rad_benchmark/all.jsonl}
OUTPUT=${OUTPUT:-results/cura.json}

TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-2000}
VALIDATION_PER_CLASS=${VALIDATION_PER_CLASS:-400}
TOP_K=${TOP_K:-20}
ROLLOUTS=${ROLLOUTS:-2}
ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-5}
TARGET_SAVE_EVERY=${TARGET_SAVE_EVERY:-100}
TARGET_EVALUATOR=${TARGET_EVALUATOR:-models/sentiment-roberta-large-english}
SIGNAL_BUDGET=${SIGNAL_BUDGET:-2}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-Method/CURA/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log}

# Safe, disjoint inference-tuning paths. These never point at results/cura.json.
EXPERIMENT_DIR=${EXPERIMENT_DIR:-results/experiments/${RUN_TAG}}
TUNE_INPUT_DIR=${TUNE_INPUT_DIR:-${EXPERIMENT_DIR}/inputs}
TUNE_INPUT=${TUNE_INPUT:-${TUNE_INPUT_DIR}/validation_prompts.jsonl}
TUNE_OUTPUT_DIR=${TUNE_OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs}
TUNE_REPORT_DIR=${TUNE_REPORT_DIR:-${EXPERIMENT_DIR}/reports}
TUNE_PER_CLASS=${TUNE_PER_CLASS:-125}
TUNE_SEED=${TUNE_SEED:-20260921}
TUNE_TRAIN=${TUNE_TRAIN:-dataset/cura_astar_source/train.jsonl}
TUNE_VALIDATION=${TUNE_VALIDATION:-dataset/cura_astar_source/validation.jsonl}
FINAL_TEST=${FINAL_TEST:-dataset/rad_benchmark/all.jsonl}
RUN_EVALUATION=${RUN_EVALUATION:-1}

if [[ "$PIPELINE_MODE" == "inference-tune" && "$CHECKPOINT" == "Method/CURA/checkpoints/${RUN_TAG}.pt" ]]; then
  CHECKPOINT=Method/CURA/checkpoints/cura_astar_full_v1.pt
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1

TOTAL_STAGES=16
CURRENT_STAGE=0
CURRENT_STAGE_NAME="initialization"
PIPELINE_STARTED=$SECONDS

duration() {
  local total=${1:-0}
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

artifact_status() {
  local label=$1
  local path=$2
  if [[ -e "$path" ]]; then
    printf '  [FOUND]   %-20s %s\n' "$label" "$path"
  else
    printf '  [MISSING] %-20s %s\n' "$label" "$path"
  fi
}

run() {
  printf '  [CMD] '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_stage() {
  local name=$1
  shift
  CURRENT_STAGE=$((CURRENT_STAGE + 1))
  CURRENT_STAGE_NAME=$name
  local started=$SECONDS
  printf '\n================================================================================\n'
  printf '[%02d/%02d] %s\n' "$CURRENT_STAGE" "$TOTAL_STAGES" "$name"
  printf '[START] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s\n' '--------------------------------------------------------------------------------'
  if run "$@"; then
    printf '[DONE]  %s | elapsed=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$((SECONDS - started))s"
  else
    local status=$?
    printf '[FAIL]  stage=%02d/%02d name=%s exit=%d elapsed=%s\n' \
      "$CURRENT_STAGE" "$TOTAL_STAGES" "$name" "$status" "$((SECONDS - started))s"
    exit "$status"
  fi
}

pipeline_exit() {
  local status=$?
  printf '\n================================================================================\n'
  if [[ "$status" -eq 0 ]]; then
    printf '[PIPELINE SUCCESS] total_elapsed=%s\n' "$(duration "$((SECONDS - PIPELINE_STARTED))")"
  else
    printf '[PIPELINE STOPPED] stage=%02d/%02d name=%s exit=%d total_elapsed=%s\n' \
      "$CURRENT_STAGE" "$TOTAL_STAGES" "$CURRENT_STAGE_NAME" "$status" \
      "$(duration "$((SECONDS - PIPELINE_STARTED))")"
    printf '[RESUME] Run the same command again after fixing the reported error.\n'
  fi
  printf '[LOG] %s\n' "$LOG_FILE"
  printf '================================================================================\n'
}
trap pipeline_exit EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

run_inference_tune() {
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "CURA checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
  if [[ "$NUM_PROMPTS" -ne $((TUNE_PER_CLASS * 2)) ]]; then
    echo "NUM_PROMPTS must equal 2 * TUNE_PER_CLASS for the balanced tuning split." >&2
    echo "Got NUM_PROMPTS=$NUM_PROMPTS and TUNE_PER_CLASS=$TUNE_PER_CLASS." >&2
    exit 2
  fi
  if [[ "$RUN_EVALUATION" != "0" && "$RUN_EVALUATION" != "1" ]]; then
    echo "RUN_EVALUATION must be 0 or 1." >&2
    exit 2
  fi

  local experiment_abs="$EXPERIMENT_DIR"
  if [[ "$experiment_abs" != /* ]]; then
    experiment_abs="$REPO_ROOT/$experiment_abs"
  fi
  local input_abs="$TUNE_INPUT"
  if [[ "$input_abs" != /* ]]; then
    input_abs="$REPO_ROOT/$input_abs"
  fi
  local output_abs="$TUNE_OUTPUT_DIR"
  if [[ "$output_abs" != /* ]]; then
    output_abs="$REPO_ROOT/$output_abs"
  fi
  local report_abs="$TUNE_REPORT_DIR"
  if [[ "$report_abs" != /* ]]; then
    report_abs="$REPO_ROOT/$report_abs"
  fi

  if [[ "$RUN_EVALUATION" == "1" ]]; then
    TOTAL_STAGES=15
  else
    TOTAL_STAGES=8
  fi

  printf '================================================================================\n'
  printf 'CURA INFERENCE-TUNING PIPELINE\n'
  printf '================================================================================\n'
  printf '  Mode                : %s\n' "$PIPELINE_MODE"
  printf '  Experiment          : %s\n' "$experiment_abs"
  printf '  Checkpoint          : %s\n' "$CHECKPOINT"
  printf '  Validation input    : %s\n' "$input_abs"
  printf '  Prompts / tokens    : %s / %s\n' "$NUM_PROMPTS" "$MAX_NEW_TOKENS"
  printf '  Top-k / seed        : %s / %s\n' "$TOP_K" "$SEED"
  printf '  Devices             : base=%s signal=%s\n' "$BASE_DEVICE" "$SIGNAL_DEVICE"
  printf '  Run evaluation      : %s\n' "$RUN_EVALUATION"
  printf '  Mode                : %s\n' "$([[ "$DRY_RUN" == "1" ]] && echo DRY-RUN || echo RUN/RESUME)"
  printf '  Log                 : %s\n' "$LOG_FILE"
  printf '\nExisting artifacts:\n'
  artifact_status "split manifest" "$TUNE_INPUT_DIR/split_manifest.json"
  artifact_status "base output" "$TUNE_OUTPUT_DIR/base.json"
  artifact_status "learned output" "$TUNE_OUTPUT_DIR/cura_learned.json"
  artifact_status "comparison CSV" "$EXPERIMENT_DIR/comparison.csv"

  run mkdir -p "$TUNE_INPUT_DIR" "$TUNE_OUTPUT_DIR" "$TUNE_REPORT_DIR"

  run_stage "Prepare fixed disjoint validation split" \
    "$PYTHON_BIN" Method/CURA/scripts/prepare_inference_validation.py \
    --train "$TUNE_TRAIN" \
    --validation "$TUNE_VALIDATION" \
    --final-test "$FINAL_TEST" \
    --output-dir "$TUNE_INPUT_DIR" \
    --per-class "$TUNE_PER_CLASS" \
    --seed "$TUNE_SEED"

  run_stage "Generate/resume Base validation outputs" \
    "$PYTHON_BIN" Method/generate_base.py \
    --dataset-path "$TUNE_INPUT" \
    --output-path "$TUNE_OUTPUT_DIR/base.json" \
    --num-prompts "$NUM_PROMPTS" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  local common=(
    --checkpoint "$CHECKPOINT"
    --input "$TUNE_INPUT"
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --top-k "$TOP_K"
    --base-device "$BASE_DEVICE"
    --signal-device "$SIGNAL_DEVICE"
    --signal-budget 2
    --seed "$SEED"
  )

  run_stage "Generate/resume CURA learned" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --output "$TUNE_OUTPUT_DIR/cura_learned.json"
  run_stage "Generate/resume CURA fixed lambda=2" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --fixed-lambda 2 --output "$TUNE_OUTPUT_DIR/cura_lambda2.json"
  run_stage "Generate/resume CURA fixed lambda=3" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --fixed-lambda 3 --output "$TUNE_OUTPUT_DIR/cura_lambda3.json"
  run_stage "Generate/resume CURA fixed lambda=4" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --fixed-lambda 4 --output "$TUNE_OUTPUT_DIR/cura_lambda4.json"
  run_stage "Generate/resume CURA fixed gate=0.75" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --fixed-gate 0.75 --output "$TUNE_OUTPUT_DIR/cura_gate075.json"
  run_stage "Generate/resume CURA no-gate" \
    "$PYTHON_BIN" Method/CURA/core/generate.py \
    "${common[@]}" --ablation no-gate --output "$TUNE_OUTPUT_DIR/cura_no_gate.json"

  if [[ "$RUN_EVALUATION" == "1" ]]; then
    local variant
    for variant in learned lambda2 lambda3 lambda4 gate075 no_gate; do
      local eval_dir="$report_abs/$variant/input"
      run mkdir -p "$eval_dir"
      run ln -sfn "$output_abs/base.json" "$eval_dir/base.json"
      run ln -sfn "$output_abs/cura_${variant}.json" "$eval_dir/cura.json"
      run_stage "Evaluate CURA ${variant} with local evaluators" \
        "$PYTHON_BIN" eval/evaluate_sota.py \
        --results-dir "$eval_dir" \
        --output-file "$report_abs/$variant/evaluation_report.csv" \
        --methods CURA \
        --max-samples "$NUM_PROMPTS" \
        --device "$BASE_DEVICE"
    done
    run_stage "Build tuning comparison CSV" \
      "$PYTHON_BIN" Method/CURA/scripts/summarize_inference_tuning.py \
      --experiment-dir "$experiment_abs" \
      --output "$experiment_abs/comparison.csv"
  fi

  printf '\nINFERENCE-TUNING ARTIFACTS\n'
  printf '  Split manifest : %s\n' "$TUNE_INPUT_DIR/split_manifest.json"
  printf '  Outputs        : %s\n' "$TUNE_OUTPUT_DIR"
  printf '  Reports        : %s\n' "$TUNE_REPORT_DIR"
  if [[ "$RUN_EVALUATION" == "1" ]]; then
    printf '  Comparison     : %s\n' "$EXPERIMENT_DIR/comparison.csv"
  fi
  printf '  Log            : %s\n' "$LOG_FILE"
}

if [[ "$PIPELINE_MODE" == "inference-tune" ]]; then
  run_inference_tune
  exit 0
fi

printf '================================================================================\n'
printf 'CURA A* PIPELINE\n'
printf '================================================================================\n'
printf '  Run tag             : %s\n' "$RUN_TAG"
printf '  Config              : %s\n' "$CONFIG"
printf '  Benchmark           : %s\n' "$BENCHMARK"
printf '  Output              : %s\n' "$OUTPUT"
printf '  Prompts / tokens    : %s / %s\n' "$NUM_PROMPTS" "$MAX_NEW_TOKENS"
printf '  Top-k / rollouts    : %s / %s x %s tokens\n' "$TOP_K" "$ROLLOUTS" "$ROLLOUT_TOKENS"
printf '  Target evaluator    : %s\n' "$TARGET_EVALUATOR"
printf '  Signal budget       : %s\n' "$SIGNAL_BUDGET"
printf '  Devices             : base=%s signal=%s\n' "$BASE_DEVICE" "$SIGNAL_DEVICE"
printf '  Mode                : %s\n' "$([[ "$DRY_RUN" == "1" ]] && echo DRY-RUN || echo RUN/RESUME)"
printf '  Log                  : %s\n' "$LOG_FILE"
printf '\nResume state before run:\n'
artifact_status "train cache" "$CACHE_DIR/train/manifest.json"
artifact_status "validation cache" "$CACHE_DIR/validation/manifest.json"
artifact_status "latest checkpoint" "${CHECKPOINT%.pt}.latest.pt"
artifact_status "best checkpoint" "$CHECKPOINT"
artifact_status "CURA result" "$OUTPUT"

run mkdir -p "$REPORT_DIR" "$(dirname "$OUTPUT")"

run_stage "Prepare balanced A* source data" \
  "$PYTHON_BIN" Method/CURA/core/prepare_astar_data.py \
  --output-dir "$SOURCE_DIR" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --validation-per-class "$VALIDATION_PER_CLASS" \
  --seed "$SEED"

run_stage "Audit reward signals and objective compatibility" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action audit-signals \
  --config "$CONFIG"

run_stage "Build/resume train token cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache --split train \
  --config "$CONFIG" \
  --input "$SOURCE_DIR/train.jsonl" \
  --cache-dir "$CACHE_DIR/train" \
  --shard-size 100 \
  --max-response-tokens 32 \
  --top-k "$TOP_K" \
  --base-device "$BASE_DEVICE" \
  --signal-device "$SIGNAL_DEVICE" \
  --resume

run_stage "Build/resume validation token cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache --split validation \
  --config "$CONFIG" \
  --input "$SOURCE_DIR/validation.jsonl" \
  --cache-dir "$CACHE_DIR/validation" \
  --shard-size 50 \
  --max-response-tokens 32 \
  --top-k "$TOP_K" \
  --base-device "$BASE_DEVICE" \
  --signal-device "$SIGNAL_DEVICE" \
  --resume

run_stage "Verify train cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache-verify \
  --cache-dir "$CACHE_DIR/train"

run_stage "Verify validation cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache-verify \
  --cache-dir "$CACHE_DIR/validation"

run_stage "Annotate train cache with rollout utility targets" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action annotate-targets --split train \
  --cache-dir "$CACHE_DIR/train" \
  --evaluator "$TARGET_EVALUATOR" \
  --rollouts "$ROLLOUTS" \
  --rollout-tokens "$ROLLOUT_TOKENS" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --target-save-every "$TARGET_SAVE_EVERY" \
  --base-device "$BASE_DEVICE" \
  --seed "$SEED"

run_stage "Annotate validation cache with rollout utility targets" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action annotate-targets --split validation \
  --cache-dir "$CACHE_DIR/validation" \
  --evaluator "$TARGET_EVALUATOR" \
  --rollouts "$ROLLOUTS" \
  --rollout-tokens "$ROLLOUT_TOKENS" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --target-save-every "$TARGET_SAVE_EVERY" \
  --base-device "$BASE_DEVICE" \
  --seed "$SEED"

run_stage "Verify annotated train cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache-verify \
  --cache-dir "$CACHE_DIR/train"

run_stage "Verify annotated validation cache" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action cache-verify \
  --cache-dir "$CACHE_DIR/validation"

run_stage "Validate signal calibration inputs" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action calibrate \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR/validation"

run_stage "Measure token-oracle headroom" \
  "$PYTHON_BIN" Method/CURA/core/oracle_study.py \
  --cache-dir "$CACHE_DIR/validation" \
  --output "$REPORT_DIR/token_oracle.json"

TRAIN_RESUME=()
LATEST_CHECKPOINT=${CHECKPOINT%.pt}.latest.pt
if [[ -f "$LATEST_CHECKPOINT" ]]; then
  TRAIN_RESUME=(--resume)
fi

run_stage "Train/resume CURA controller" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action train \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR/train" \
  --validation-cache-dir "$CACHE_DIR/validation" \
  --output "$CHECKPOINT" \
  --base-device "$BASE_DEVICE" \
  --seed "$SEED" \
  "${TRAIN_RESUME[@]}"

run_stage "Diagnose trained checkpoint" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action diagnose \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR/validation" \
  --output "$REPORT_DIR/checkpoint_diagnostic.json" \
  --base-device "$BASE_DEVICE"

run_stage "Generate CURA benchmark responses" \
  "$PYTHON_BIN" Method/run_method.py \
  --method cura --action generate \
  --checkpoint "$CHECKPOINT" \
  --input "$BENCHMARK" \
  --output "$OUTPUT" \
  --num-prompts "$NUM_PROMPTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-k "$TOP_K" \
  --base-device "$BASE_DEVICE" \
  --signal-device "$SIGNAL_DEVICE" \
  --signal-budget "$SIGNAL_BUDGET" \
  --seed "$SEED"

run_stage "Validate final CURA result" \
  "$PYTHON_BIN" Method/CURA/core/validate_run.py \
  --input "$OUTPUT" \
  --paper-mode

printf '\nFINAL ARTIFACTS\n'
printf '  Checkpoint : %s\n' "$CHECKPOINT"
printf '  Diagnostic : %s\n' "$REPORT_DIR/checkpoint_diagnostic.json"
printf '  Result     : %s\n' "$OUTPUT"
printf '  Log        : %s\n' "$LOG_FILE"
printf '\nEvaluation is separate. After every method JSON is ready, run:\n'
printf '  %q eval/evaluate_sota.py --results-dir results --output-file results/evaluation_report --device %q\n' \
  "$PYTHON_BIN" "$BASE_DEVICE"
