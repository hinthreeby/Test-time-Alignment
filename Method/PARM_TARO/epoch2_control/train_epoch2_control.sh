#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jupyter-iec2024se10/Test-time-Alignment"
PY="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"
SOURCE="$ROOT/results/parm_taro/recovery/reproduced/pblora/checkpoint-200"
OUT="$ROOT/results/parm_taro/recovery/reproduced/pblora_epoch2_control"
WORK="$ROOT/results/parm_taro/recovery/reproduced/pblora_epoch2_control_work"
BASE="$ROOT/models/tulu-2-7b"
TRAIN="$ROOT/dataset/parm_taro/train.json"
VALIDATION="$ROOT/dataset/parm_taro/validation.json"

[[ -f "$SOURCE/optimizer.pt" && -f "$SOURCE/scheduler.pt" && -f "$SOURCE/rng_state.pth" ]] || { echo "checkpoint-200 is incomplete" >&2; exit 2; }
[[ -f "$SOURCE/trainer_state.json" && -f "$SOURCE/adapter_model.safetensors" ]] || { echo "checkpoint-200 model/state missing" >&2; exit 2; }
mkdir -p "$OUT" "$WORK/training" "$WORK/data"
ln -sfn "$TRAIN" "$WORK/data/train.json"
ln -sfn "$VALIDATION" "$WORK/data/dev.json"

RESUME="$SOURCE"
while IFS= read -r candidate; do
  [[ -f "$candidate/trainer_state.json" && -f "$candidate/optimizer.pt" && -f "$candidate/scheduler.pt" ]] && RESUME="$candidate"
done < <(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

cd "$WORK/training"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" "$ROOT/PARM_TARO/epoch2_control/fixed_resume.py" \
  --recovery-output-dir "$OUT" --resume-from-checkpoint "$RESUME" -- \
  --preference_dataset=PKU_SafeRLHF --pref_sample_p=0.5 \
  --lora_r=4 --lora_r2=4 --lora_alpha=8 --lora_dropout=0.05 \
  --safe_obj=true --help_obj=true --beta_safe=0.01 --beta_help=0.01 \
  --model_name_or_path="$BASE" --beta=0.5 \
  --learning_rate=0.0005 --num_train_epochs=2 --output_dir="$OUT" \
  --run_name=parm_pblora_epoch2_control --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=8 --per_device_eval_batch_size=2 \
  --logging_steps=1 --evaluation_strategy=steps --eval_steps=50 \
  --save_strategy=steps --save_steps=50 --lr_scheduler_type=cosine \
  --warmup_steps=20 --weight_decay=0.05 --gradient_checkpointing=true \
  --bf16=true --load_in_4bit=true --max_prompt_length=512 --max_length=1024 \
  --report_to=none --remove_unused_columns=false --seed=42
