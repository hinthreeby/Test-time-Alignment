'''
Adapted from args/collect_model_outs.py
'''

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM


##################### Arguments #####################
parser = argparse.ArgumentParser()
# Model
parser.add_argument("--base_model_pth", type=str)
parser.add_argument("--arm_pth", type=str)
parser.add_argument("--alpha", type=float, default=1.0, help='Final model = base_model + alpha * arm')
parser.add_argument("--temperature", type=float, default=0.5, help='Technically, we should use temperature = 1/(1+alpha)')
parser.add_argument("--max_new_tokens", type=int, default=128)

# Output file: output json file will be saved in out_folder/out_file.jsonl
parser.add_argument("--out_folder", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--resume", action="store_true", help="Resume from previous run")

# Dataset
parser.add_argument("--num_prompt", type=int, default=300, help="Number of prompts to generate response") 
parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum length of prompt. Prompt longer than this will be filtered out")
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--split", type=str, default="test")

args = parser.parse_args()

##################### output path #####################
out_path = Path(os.path.join(args.out_folder, args.out_file) + f".jsonl")
if out_path.exists() and (not args.resume):
    print("ERROR: out_path already exists! If resuming, use --resume flag.")
    exit(1)
if not out_path.exists() and args.resume:
    print("ERROR: Resuming but out_path DOESN'T exist!")
    exit(1)
print(f'Generation results will be saved in {out_path}')

##################### Load dataset #####################
print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
assert args.dataset == "Dahoas/full-hh-rlhf", "Only support Dahoas/full-hh-rlhf dataset for now"
test_ds = load_dataset(args.dataset, split=args.split)

# filer out those prompts longer than max_prompt_length
tokenizer_for_filter = AutoTokenizer.from_pretrained(args.base_model_pth)
test_ds.filter(lambda x: len(tokenizer_for_filter(x["prompt"])["input_ids"]) <= args.max_prompt_length)

if args.dataset == "Dahoas/full-hh-rlhf":
    # FOR HHRLHF
    test_ds = test_ds["prompt"]
elif args.dataset == "stanfordnlp/SHP":
    # FOR SHP (not used for now. keep it for future reference)
    unique_prompts = []
    seen_posts = set()
    for post_id, histr in zip(test_ds["post_id"], test_ds['history']):
        if post_id in seen_posts: continue
        model_prompt = " Human: " + histr + " Assistant: "
        unique_prompts.append(model_prompt)
        seen_posts.add(post_id)
    test_ds = unique_prompts

truncated_ds = test_ds[0:args.num_prompt] # will be used for generating responses

##################### Load models #####################
prompt_template_base = lambda system_prompt, input_string: f"{input_string}"
prompt_template_arm = lambda system_prompt, input_string: f"{input_string}"

M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_base, model=args.base_model_pth) 
M_arm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_arm, model=args.arm_pth) if args.alpha != 0 else None

if args.alpha == 0:
    print("alpha is 0, using the base model directly.")
    formula = M_base
else:
    formula = M_base + args.alpha * M_arm

M = ModelArithmetic(formula, needs_input_tokens_lm_eval=False, lm_eval_task=None) 
# generate function will remove eos token
generate = lambda prompt: M.generate_text(prompt, max_new_tokens=args.max_new_tokens, batch_size=None, temperature=args.temperature, top_p=1, top_k=0, do_speculation=False)[0].removesuffix(M.tokenizer.eos_token)

##################### Generate responses #####################
# Initialize or load existing output data
if args.resume:
    with open(out_path, 'r', encoding='utf-8') as f:
        output_set = json.load(f)
    start_index = len(output_set)  # Resume from where it left off
    print(f'Resuming from {start_index}-th prompt.\n')
else:
    output_set = []
    start_index = 0

start_time_script = time.time()
for i in tqdm(range(start_index, len(truncated_ds))):
    prompt = truncated_ds[i]

    start = time.time()
    response = generate(prompt)
    elapsed = time.time() -start

    output_set.append({
        "prompt": prompt, 
        "result": response,
        "response": prompt + response,
        "elapsed":elapsed,
        "method": os.path.join(args.out_folder, args.out_file)})

    if i % 10 == 1:
        print(f'''Generated {i} outputs.\n Prompt: {output_set[-1]["prompt"]}\n response: {output_set[-1]["result"]}''')
        # Save intermediate results to avoid data loss
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_set, f, ensure_ascii=False, indent=4)

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output_set, f, ensure_ascii=False, indent=4)
print(f'Generating responses finished!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/ 3600} hours for {len(output_set) - start_index} outputs')
