# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cuda=0,1,2,3,4,5,6,7
num_train_epochs=1 # default = 5
dpo_beta=0.1 # default = 0.1

lora_rank=64 # default = 64
lora_alpha=16 # default = 16
max_seq_length=1024 # default = 1024; full finetune uses 2048
learning_rate=1e-4 # default = 1e-4


### the following is automatically set
exp_name=tulu2_dpo_qlora-beta_$dpo_beta-epoch_$num_train_epochs-lr_$learning_rate-loraRank_$lora_rank-loraAlpha_$lora_alpha-maxSeq_$max_seq_length

# NUM_GPUS=8
NUM_GPUS=$(echo $cuda | awk -F, '{print NF}')
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "exp_name: $exp_name"

# Lora training
accelerate launch --gpu_ids $cuda \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/dpo_tune.py \
    --model_name_or_path allenai/tulu-2-7b \
    --use_qlora \
    --use_lora \
    --use_flash_attn \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout 0.1 \
    --tokenizer_name allenai/tulu-2-7b \
    --use_slow_tokenizer \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --max_seq_length $max_seq_length \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $learning_rate \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $num_train_epochs \
    --output_dir checkpoints/tulu_v2_dpo_qlora/$exp_name \
    --with_tracking \
    --report_to wandb --wandb_exp_name $exp_name \
    --logging_steps 1 \
    --dpo_beta $dpo_beta 

    # --logging_steps 1 &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path allenai/tulu-2-7b \
#     --lora_model_name_or_path checkpoints/tulu_v2_dpo_qlora/$exp_name \
#     --output_dir checkpoints/tulu_v2_dpo_qlora/merged-$exp_name \
#     --qlora \
#     --save_tokenizer
