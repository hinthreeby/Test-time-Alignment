# (when running DPO model, use DPO model as the base model, and set alpha = 0, temperature = 1)

cuda=0,1

base_model_pth=argsearch/llama-7b-sft-float32 # SFT model provided by ARGS
base_model_script_name=args-llama-sft-7b
arm_pth=Your_AutoregressiveRM_Path
arm_script_name=Your_AutoregressiveRM_Name_For_Logging

alpha=1 # 0; 1 
temperature=0.5 # set to 1 / (1 + alpha) to sample from pi_decode with temperature=1.

num_prompt=300

### automatically set
out_folder=model_outputs
if [ "$alpha" -eq 0 ]; then
    echo "alpha is zero; only use the base model"
    out_file=Base_$base_model_script_name-NoArm-temp_$temperature-promptNum_$num_prompt
else
    out_file=Base_$base_model_script_name-Arm_$arm_script_name-Alpha_$alpha-temp_$temperature-promptNum_$num_prompt
fi

echo "Output dir: $out_folder/$out_file"

CUDA_VISIBLE_DEVICES=$cuda python generate_outputs_HH.py \
    --base_model_pth=$base_model_pth --arm_pth=$arm_pth --alpha=$alpha --temperature=$temperature \
    --num_prompt=$num_prompt --out_folder=$out_folder --out_file=$out_file \
    --max_new_tokens 128 --max_prompt_length 2048