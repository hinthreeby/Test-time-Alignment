cuda=0,1
algorithm=arm # arm or dpo
epoch=1
beta=0.05 # for DPO, use beta=0.1
learning_rate=5e-4
bs=32 # Total batch size, which is $per_device_train_batch_size * $gradient_accumulation_steps * $num_GPU
per_device_train_batch_size=2 # Adjust according to your GPU memory

model_name_or_path="argsearch/llama-7b-sft-float32" # sft model in the args paper
model_name_script=args-llama-sft-7b # only used for saving the model

###### the following is automatically set
num_GPU=$(echo $cuda | awk -F, '{print NF}')
gradient_accumulation_steps=$(($bs/$num_GPU/$per_device_train_batch_size))
exp_name=$model_name_script-$algorithm-HH-epoch_$epoch-beta_$beta-lr_$learning_rate-bs_$bs

output_dir=checkpoints/HH/$algorithm/$exp_name
if [ -d "${output_dir}" ]; then
    echo -e "\n\n"
    echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir." >&2
    exit 1
fi
echo "Output dir: $output_dir"

accelerate launch --gpu_ids $cuda --main_process_port 29500 training_trl/train_arm_llama.py --preference_dataset="HH_RLHF" \
    --algorithm=$algorithm --model_name_or_path=$model_name_or_path \
    --beta=$beta --learning_rate=$learning_rate --num_train_epochs=$epoch \
    --output_dir=$output_dir --run_name=$exp_name \
    --per_device_train_batch_size=$per_device_train_batch_size --gradient_accumulation_steps=$gradient_accumulation_steps --per_device_eval_batch_size=4 \
    --logging_steps=10 --evaluation_strategy="steps" --eval_steps=50 --save_strategy="steps" --save_steps=1000 \
    --lr_scheduler_type="cosine" --warmup_steps=100 --weight_decay=0.05 --gradient_checkpointing=True --bf16=True \
    --max_prompt_length=512 --max_length=1024 --report_to="wandb" --remove_unused_columns=False 

echo "Finished training $output_dir"
