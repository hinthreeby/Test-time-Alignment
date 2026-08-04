# The evaluation data is from the supplemetary material of the safe-RLHF paper: https://openreview.net/forum?id=TyFrPOKYXw; This evaluation dataset is for GPT evaluation
datasets=./materials_ICLR/evaluation_prompts-500.json

# Case 1: single model (like DPO, RLHF trained models): set model_base_name_or_path, and set alpha_helpfulness=0, alpha_harmlessness=0
# Case 2: Rewarded Soups: Set rewarded_soups=True, set model_base_name_or_path=anything, model_arm_helpfulness_name_or_path & model_arm_harmlessness_name_or_path to be the DPO model paths, and set alpha list
# Case 3: Controlled Decoding using ARM: set model_base_name_or_path, model_arm_helpfulness_name_or_path & model_arm_harmlessness_name_or_path to be the ARM model paths, and set alpha list
# Case 3.1: If only use one of the ARM, set another one to "none"

gpu=0,1

model_base_name_or_path=PKU-Alignment/alpaca-7b-reproduced # or "TheBloke/alpaca-lora-65B-GPTQ"

# put model_arm_helpfulness_name_or_path or model_arm_harmlessness_name_or_path as none if necessary.
model_arm_helpfulness_name_or_path=Provide_Your_Path_Here 
model_arm_harmlessness_name_or_path=Provide_Your_Path_Here 

alpha_helpfulness_list=(0.2 0.8)
alpha_harmlessness_list=(4 1)

resume=False # True or False
rewarded_soups=False # True or False; When True, model_arm_helpfulness_name_or_path and model_arm_harmlessness_name_or_path should be LLMs (like DPO models) instead of reward models like ARMs. 

output_dir_base=./gpt-evaluation/generation

for i in "${!alpha_helpfulness_list[@]}"; do
    alpha_helpfulness=${alpha_helpfulness_list[$i]}
    alpha_harmlessness=${alpha_harmlessness_list[$i]}

    ########### The following are the automatic ###########
    model_name_for_logging=${model_base_name_or_path##*/}
    model_arm_helpfulness_name_for_logging=${model_arm_helpfulness_name_or_path#*Helpfulness-}
    model_arm_harmlessness_name_for_logging=${model_arm_harmlessness_name_or_path#*Harmlessness-}

    # if [[ "$alpha_helpfulness" -eq 0 && "$alpha_harmlessness" -eq 0 ]]; then
    if [[ $(echo "$alpha_helpfulness == 0" | bc) -eq 1 && $(echo "$alpha_harmlessness == 0" | bc) -eq 1 ]]; then
        # only base model is used
        echo Only use the base model $model_base_name_or_path
        if [[ "$model_base_name_or_path" == *"dpo"* ]]; then
            output_dir=$output_dir_base/dpo/${model_base_name_or_path##*/}
        else
            output_dir=$output_dir_base/single_models/${model_base_name_or_path##*/}
        fi
    else
        if [ "$rewarded_soups" = "True" ]; then
            # rewarded_soups
            folder_name_arm=Helpfulness-$model_arm_helpfulness_name_for_logging-Harmlessness-$model_arm_harmlessness_name_for_logging
            folder_name_alpha=alpha-Helpfulness-$alpha_helpfulness-Harmlessness-$alpha_harmlessness
            output_dir=$output_dir_base/Rewarded_Soups/$folder_name_arm/$folder_name_alpha
        else
            # controlled decoding using ARM
            folder_name_arm=Helpfulness-$model_arm_helpfulness_name_for_logging-Harmlessness-$model_arm_harmlessness_name_for_logging
            folder_name_alpha=alpha-Helpfulness-$alpha_helpfulness-Harmlessness-$alpha_harmlessness
            output_dir=$output_dir_base/ARM_decoding/$folder_name_arm/$folder_name_alpha
        fi
    fi

    echo $output_dir

    if [[ "$resume" != "True" ]]; then
        if [ -d "${output_dir}" ]; then
            echo -e "\n\n"
            echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir, or use resume=True" >&2
            exit 1
        fi
    fi

    mkdir -p "${output_dir}"
    output_dir="$(cd "${output_dir}" &>/dev/null && pwd)"
    if [[ ! -f "${output_dir}/.gitignore" ]]; then
        echo '*' >"${output_dir}/.gitignore"
    fi
    cp -f "$0" "${output_dir}/script.sh"


    CUDA_VISIBLE_DEVICES=$gpu python generate_outputs_Beaver.py \
        --model_base_name_or_path $model_base_name_or_path \
        --model_arm_helpfulness_name_or_path $model_arm_helpfulness_name_or_path \
        --model_arm_harmlessness_name_or_path $model_arm_harmlessness_name_or_path \
        --alpha_helpfulness $alpha_helpfulness \
        --alpha_harmlessness $alpha_harmlessness \
        --model_name_for_logging $model_name_for_logging \
        --model_arm_helpfulness_name_for_logging $model_arm_helpfulness_name_for_logging \
        --model_arm_harmlessness_name_for_logging $model_arm_harmlessness_name_for_logging \
        --datasets $datasets --output_dir $output_dir --resume $resume \
        --rewarded_soups $rewarded_soups 
done