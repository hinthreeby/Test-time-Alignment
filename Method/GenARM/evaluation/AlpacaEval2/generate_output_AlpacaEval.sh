cuda=0,1

### Case 1: GenARM: model_base + alpha * autoregressive RM
model_base=allenai/tulu-2-7b # some choices: allenai/tulu-2-7b, TheBloke/tulu-2-13B-GPTQ, TheBloke/tulu-2-70B-GPTQ, allenai/tulu-2-13b, allenai/tulu-2-70b
model_arm=Provide_The_Path_For_AutoregressiveRM 
alpha_arm=1 # the weight for the autoregressive RM
temperature=1 # for GenARM, the temperature input to model_arithmetic is temperature/(1 + alpha_arm) 

### Case 2: evaluation single model without guided decoding
# model_base=allenai/tulu-2-dpo-7b # some choices: allenai/tulu-2-7b, allenai/tulu-2-dpo-7b, TheBloke/tulu-2-13B-GPTQ, TheBloke/tulu-2-dpo-13B-GPTQ, TheBloke/tulu-2-70B-GPTQ, TheBloke/tulu-2-dpo-70B-GPTQ, allenai/tulu-2-13b, allenai/tulu-2-70b
# model_arm=none
# alpha_arm=0 
# temperature=0 # default = 0 in AlpacaEval2.0

###### the following are automatically set
max_new_tokens=7500 # default for tulu2
top_p=1 

output_dir=./AlpacaEval_results/model_generation/
# the json file will be saved to output_dir/model_save_name.json
if [ "$model_arm" = "none" ] || [ $(echo "$alpha_arm == 0" | bc) -eq 1 ]; then
    model_save_name=${model_base##*/}-temp_$temperature # get the last part of the model name
    echo Only use the base model. Results will be saved to $output_dir$model_save_name.json
else
    model_save_name=Base-${model_base##*/}-ARM-${model_arm##*/}-alpha_$alpha_arm-temp_$temperature
    echo GenARM decoding. Results will be saved to $output_dir$model_save_name.json
fi

CUDA_VISIBLE_DEVICES=$cuda python generate_output_AlpacaEval.py \
      --model_base $model_base --model_arm $model_arm --alpha_arm $alpha_arm \
      --temperature $temperature --max_new_tokens $max_new_tokens --top_p $top_p \
      --output_dir $output_dir --model_save_name $model_save_name

# use --debug if needed