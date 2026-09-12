#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${TTA_PROJECT_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PY="${TTA_PYTHON:-$(command -v python3)}"
[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
OUT="$ROOT/results/parm_taro/recovery/reproduced/pblora"
WORK="$ROOT/results/parm_taro/recovery/reproduced/pblora_work"
RESUME=false
if [[ "${1:-}" == "--resume" ]]; then RESUME=true; shift; fi
if [[ -z "${PBLORA_SEED:-}" ]]; then
  echo "PBLORA_SEED is unresolved by retained history; set it explicitly after protocol approval." >&2
  exit 2
fi
mkdir -p "$OUT" "$WORK/training" "$WORK/data"
ln -sfn "$ROOT/dataset/parm_taro/train.json" "$WORK/data/train.json"
ln -sfn "$ROOT/dataset/parm_taro/validation.json" "$WORK/data/dev.json"
cd "$WORK/training"
resume_args=()
if $RESUME; then
  latest=$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -t- -k2,2n | tail -1)
  [[ -n "$latest" ]] || { echo "No HF Trainer checkpoint exists under $OUT" >&2; exit 3; }
  resume_args=(--resume-from-checkpoint "$OUT/$latest")
fi
exec "$PY" "$ROOT/PARM_TARO/recovery/pblora_resume_entrypoint.py" "${resume_args[@]}" -- \
  --preference_dataset=PKU_SafeRLHF --pref_sample_p=0.5 \
  --lora_r=4 --lora_r2=4 --lora_alpha=8 --lora_dropout=0.05 \
  --safe_obj=true --help_obj=true --beta_safe=0.01 --beta_help=0.01 \
  --model_name_or_path="$ROOT/models/tulu-2-7b" --beta=0.5 \
  --learning_rate=0.0005 --num_train_epochs=2 --output_dir="$OUT" \
  --run_name=parm_pblora_reproduction --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=8 --per_device_eval_batch_size=2 \
  --logging_steps=10 --evaluation_strategy=steps --eval_steps=20 \
  --save_strategy=steps --save_steps=250 --save_total_limit=4 \
  --lr_scheduler_type=cosine --warmup_steps=20 --weight_decay=0.05 \
  --gradient_checkpointing=true --bf16=true --max_prompt_length=512 \
  --max_length=1024 --report_to=none --remove_unused_columns=false \
  --seed="$PBLORA_SEED" "$@"
