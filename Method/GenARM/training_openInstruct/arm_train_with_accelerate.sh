cuda=0,1,2,3,4,5,6,7 
main_process_port=29500

dpo_beta=0.1 # it is called dpo_beta but it is actually the beta parameter for training autoregressive RM
learning_rate=5e-7 
num_train_epochs=3

### the following is automatically set
exp_name=tulu2_arm_fullfinetune-beta_$dpo_beta-epoch_$num_train_epochs-lr_$learning_rate
NUM_GPUS=$(echo $cuda | awk -F, '{print NF}')
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "exp_name: $exp_name"

# full finetuning
accelerate launch --gpu_ids $cuda --main_process_port $main_process_port \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/arm_tune.py \
    --model_name_or_path allenai/tulu-2-7b \
    --use_flash_attn \
    --gradient_checkpointing \
    --tokenizer_name allenai/tulu-2-7b \
    --use_slow_tokenizer \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $learning_rate \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs $num_train_epochs \
    --output_dir checkpoints/tulu_v2_arm_fullfinetune/$exp_name \
    --with_tracking \
    --report_to wandb --wandb_exp_name $exp_name \
    --logging_steps 1 --checkpointing_steps "epoch" \
    --dpo_beta $dpo_beta


    