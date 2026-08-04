'''
This file is for generating responses for the AlpacaEval 2.0 prompts. 
The output is saved in a json file. Then we need to run alpca_eval to evaluate the responses.

The prompt template is for Tulu model. For other models, we need to change the prompt template.
For Tulu: https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/tulu-2-dpo-70b/configs.yaml
'''
import datasets
import torch
import json
from tqdm import tqdm
import argparse
import time
import os
from transformers import AutoTokenizer

from model_arithmetic import ModelArithmetic, PromptedLLM

############# Argument parser #############
parser = argparse.ArgumentParser(description='Generate output for a model')
parser.add_argument('--model_base', type=str, default="allenai/tulu-2-7b")
parser.add_argument('--model_arm', type=str, default=None) # if None, only use the model_base for generation.
parser.add_argument('--alpha_arm', type=float, default=1) # model = model_base + alpha_arm * model_arm

# save outputs; the generated outputs will be saved in output_dir/model_save_name.json
parser.add_argument('--output_dir', type=str, default="alpaca_eval_generations/") 
parser.add_argument('--model_save_name', type=str, default="tulu2") # the name of the model in the alpaca benchmark. 

# generation arguments 
parser.add_argument('--max_new_tokens', type=int, default=7500) # default for all tulu: 7500 
parser.add_argument('--temperature', type=float, default=0.0) # default for all tulu: 0.0
parser.add_argument('--top_p', type=float, default=1.0) # default for all tulu: 1.0

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.model_arm is not None:
    if args.model_arm.lower() == 'none':
        args.model_arm = None


if args.debug:
    print('Debug mode is on, setting max_new_tokens to 128.')
    args.max_new_tokens = 128
    args.model_save_name = args.model_save_name + '_debug'

############# Model set up #############
assert 'tulu' in args.model_base, "Only tulu model is supported for now. Otherwise need to change the prompt template."

# NOTE: the following prompt template is only for tulu model (7,13,70B)
prompt_template = lambda system_prompt, input_string: f"<|user|>\n{input_string}\n<|assistant|>\n" 

# base model
tokenizer = AutoTokenizer.from_pretrained(args.model_base)
M_base = PromptedLLM(system_prompt="Not used for tulu", prompt_template=prompt_template, model=args.model_base, tokenizer=tokenizer) 

# arm model
if args.model_arm is None or args.alpha_arm == 0:
    print('\n\n********************************************')
    print(f'Only using the base model: {args.model_base}\n\n')
    formula = 1 * M_base
    temperature = args.temperature
else:
    assert args.alpha_arm > 0, "alpha_arm must be positive"
    print('\n\n********************************************')
    print(f'ARM guided Decoding: base model: {args.model_base} and the arm model: {args.model_arm} with alpha: {args.alpha_arm}\n\n')
    M_arm = PromptedLLM(system_prompt="Not used for tulu", prompt_template=prompt_template, model=args.model_arm, tokenizer=tokenizer) 
    formula = M_base + args.alpha_arm * M_arm
    temperature = args.temperature / (1 + args.alpha_arm)

M = ModelArithmetic(formula, needs_input_tokens_lm_eval=False, lm_eval_task=None, dtype=torch.bfloat16) 

generation_save_path = os.path.join(args.output_dir, args.model_save_name) + '.json'
print(f'Generated responses will be saved to {generation_save_path}')

############# Generate output on AE #############
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
generate = lambda prompt: M.generate_text(prompt, max_new_tokens=args.max_new_tokens, batch_size=None, temperature=temperature, top_p=args.top_p, top_k=0, do_speculation=False)[0]
output_set = []

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Initialize or load existing output data
if os.path.exists(generation_save_path):
    with open(generation_save_path, 'r', encoding='utf-8') as f:
        output_set = json.load(f)
    start_index = len(output_set)  # Resume from where it left off
    print(f'Output file {generation_save_path} exists\nStarting from {start_index}-th prompt in the eval_set\n\n')
else:
    output_set = []
    start_index = 0

# generate! 
start_time = time.time() 
for i in tqdm(range(start_index, len(eval_set))):
    output_set.append(
        {
            "instruction": eval_set[i]["instruction"],
            "output": generate(eval_set[i]["instruction"]).removesuffix(M.tokenizer.eos_token),
            "generator": args.model_save_name 
        }
    )
    if i % 5 == 1 or args.debug:
        print(f'''Generated {i} outputs.\n Instruction: {output_set[-1]["instruction"]}\n Output: {output_set[-1]["output"]}''')
        # Save intermediate results to avoid data loss
        with open(generation_save_path, 'w', encoding='utf-8') as f:
            json.dump(output_set, f, ensure_ascii=False, indent=4)
    
    if args.debug and i == 2:
        break

############# Save output #############
with open(generation_save_path, 'w', encoding='utf-8') as f:
    json.dump(output_set, f, ensure_ascii=False, indent=4)

print(f'script finished!\nSaving to {generation_save_path}\nTime:{(time.time()-start_time)/ 3600} hours for {len(output_set) - start_index} outputs')