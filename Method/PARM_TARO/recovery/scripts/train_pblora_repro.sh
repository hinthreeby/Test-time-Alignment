#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$SCRIPT_DIR/gpu_env.sh"
MODE="full"
RESUME=false
DRY_RUN=false
TRAIN_SUBSET=""
MAX_STEPS=""
PASSTHROUGH=()

usage() {
  cat <<'EOF'
Usage: train_pblora_repro.sh [OPTIONS] [-- extra Trainer arguments]

  --mode smoke|one-epoch|full  smoke=256 rows/30 steps; one-epoch and full
                               share the canonical output so full can resume
  --train-subset N             deterministic first N training rows
  --max-steps N                override Trainer max_steps
  --resume                     resume latest valid checkpoint in output dir
  --dry-run                    validate and print command without training
EOF
}

while (($#)); do
  case "$1" in
    --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
    --train-subset) TRAIN_SUBSET="${2:?missing value for --train-subset}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?missing value for --max-steps}"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

case "$MODE" in
  smoke)
    TRAIN_SUBSET="${TRAIN_SUBSET:-256}"
    MAX_STEPS="${MAX_STEPS:-30}"
    EPOCHS=2
    SAVE_STEPS=25
    DEFAULT_OUT="$ROOT/results/parm_taro/recovery/reproduced/pblora_smoke"
    ;;
  one-epoch)
    EPOCHS=1
    SAVE_STEPS=50
    DEFAULT_OUT="$ROOT/results/parm_taro/recovery/reproduced/pblora"
    ;;
  full)
    EPOCHS=2
    SAVE_STEPS=50
    DEFAULT_OUT="$ROOT/results/parm_taro/recovery/reproduced/pblora"
    ;;
  *) echo "Unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

resolve_gpu_python PBLORA_PYTHON
PY="$GPU_PYTHON_BIN"
gpu_env_preflight "$PY" true "$DRY_RUN"

OUT="${PBLORA_OUTPUT_DIR:-$DEFAULT_OUT}"
WORK="${PBLORA_WORK_DIR:-$ROOT/results/parm_taro/recovery/reproduced/pblora_work}"
BASE="${PBLORA_BASE_MODEL:-$ROOT/models/tulu-2-7b}"
TRAIN="$ROOT/dataset/parm_taro/train.json"
VALIDATION="$ROOT/dataset/parm_taro/validation.json"
SEED="${PBLORA_SEED:-42}"
MICRO_BATCH="${PBLORA_MICRO_BATCH:-4}"
EFFECTIVE_BATCH="${PBLORA_EFFECTIVE_BATCH:-32}"

[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
[[ -f "$BASE/config.json" ]] || { echo "Base model is incomplete: $BASE" >&2; exit 2; }
[[ -f "$TRAIN" && -f "$VALIDATION" ]] || { echo "Project train/validation split is missing" >&2; exit 2; }
(( MICRO_BATCH > 0 && EFFECTIVE_BATCH % MICRO_BATCH == 0 )) || {
  echo "PBLORA_EFFECTIVE_BATCH must be divisible by PBLORA_MICRO_BATCH" >&2; exit 2;
}
GRAD_ACC=$((EFFECTIVE_BATCH / MICRO_BATCH))

mkdir -p "$OUT" "$WORK/training" "$WORK/data"
ln -sfn "$TRAIN" "$WORK/data/train.json"
ln -sfn "$VALIDATION" "$WORK/data/dev.json"

if ! $RESUME && find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  echo "Existing checkpoint found under $OUT; use --resume or choose PBLORA_OUTPUT_DIR." >&2
  exit 3
fi

RESUME_ARGS=()
if $RESUME; then
  latest=""
  if [[ -L "$OUT/last" && -f "$OUT/last/trainer_state.json" ]]; then
    latest="$(readlink -f "$OUT/last")"
  else
    while IFS= read -r candidate; do
      [[ -f "$candidate/trainer_state.json" ]] && latest="$candidate"
    done < <(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  fi
  [[ -n "$latest" ]] || { echo "No valid Trainer checkpoint under $OUT" >&2; exit 3; }
  RESUME_ARGS=(--resume-from-checkpoint "$latest")
fi

ENTRY_ARGS=("$PY" "$ROOT/PARM_TARO/recovery/pblora_resume_entrypoint.py" --recovery-output-dir "$OUT")
[[ -n "$TRAIN_SUBSET" ]] && ENTRY_ARGS+=(--train-subset "$TRAIN_SUBSET")
ENTRY_ARGS+=("${RESUME_ARGS[@]}" --)

TRAINER_ARGS=(
  --preference_dataset=PKU_SafeRLHF --pref_sample_p=0.5
  --lora_r=4 --lora_r2=4 --lora_alpha=8 --lora_dropout=0.05
  --safe_obj=true --help_obj=true --beta_safe=0.01 --beta_help=0.01
  --model_name_or_path="$BASE" --beta=0.5
  --learning_rate=0.0005 --num_train_epochs="$EPOCHS" --output_dir="$OUT"
  --run_name="parm_pblora_reproduction_${MODE}" --per_device_train_batch_size="$MICRO_BATCH"
  --gradient_accumulation_steps="$GRAD_ACC" --per_device_eval_batch_size=2
  --logging_steps=1 --evaluation_strategy=steps --eval_steps="$SAVE_STEPS"
  --save_strategy=steps --save_steps="$SAVE_STEPS"
  --lr_scheduler_type=cosine --warmup_steps=20 --weight_decay=0.05
  --gradient_checkpointing=true --bf16=true --load_in_4bit=true
  --max_prompt_length=512 --max_length=1024 --report_to=none
  --remove_unused_columns=false --seed="$SEED"
)
[[ -n "$MAX_STEPS" ]] && TRAINER_ARGS+=(--max_steps="$MAX_STEPS")
TRAINER_ARGS+=("${PASSTHROUGH[@]}")

echo "PBLoRA provenance: historical base=Tulu-2-7B; seed=$SEED is explicit fallback (historical seed unrecorded)."
echo "Mode=$MODE output=$OUT effective_batch=$EFFECTIVE_BATCH gradient_accumulation=$GRAD_ACC"
printf 'Command:'
printf ' %q' env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TOKENIZERS_PARALLELISM=false "${ENTRY_ARGS[@]}" "${TRAINER_ARGS[@]}"
printf '\n'

if $DRY_RUN; then
  exit 0
fi

cd "$WORK/training"
exec env TOKENIZERS_PARALLELISM=false "${ENTRY_ARGS[@]}" "${TRAINER_ARGS[@]}"
