'''
The evaluation data is from the supplemetary material of the safe-RLHF paper: https://openreview.net/forum?id=TyFrPOKYXw
The template is for the Alpaca model in the safe-rlhf paper. Here we assume all the models (base and reward model) use the same template.
'''

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os
from collections import OrderedDict
import torch


# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_arithmetic import ModelArithmetic, PromptedLLM
from peft.utils.save_and_load import get_peft_model_state_dict

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generation of prompts using LLMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_base_name_or_path',
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--model_arm_helpfulness_name_or_path',
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--model_arm_harmlessness_name_or_path',
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--alpha_helpfulness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness',
    )
    model_parser.add_argument(
        '--alpha_harmlessness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness',
    )

    model_parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='The maximum sequence length of the generation.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )

    model_parser.add_argument(
        '--model_name_for_logging',
        type=str,
    )
    model_parser.add_argument(
        '--model_arm_helpfulness_name_for_logging',
        type=str,
    )
    model_parser.add_argument(
        '--model_arm_harmlessness_name_for_logging',
        type=str,
    )

    model_parser.add_argument(
        '--normalize_logit',
        type=str2bool,
        default=False,
        help='If True, the temperature argument to ModelArithmetic will be enforced to 1.0, and thus the logit weights of base LLM and ARM sum up to 1; If false, when using ARM decoding, the temperature will be 1/(1+alpha_1+alpha_2).',
    )

    # Method
    method_parser = parser.add_argument_group('method')
    method_parser.add_argument(
        '--rewarded_soups',
        type=str2bool,
        default=False,
        help='Using rewarded soups, a baseline method that interpolates the weights of multiple models.',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--datasets',
        type=str,
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--resume',
        type=str2bool,
        default=False,
    )

    args = parser.parse_args()
    return args

################ Model Arithmetic ################
def get_model_arithmetic(model_pth_base, model_pth_reward_1, model_pth_reward_2, alpha_1, alpha_2, args):
    '''
    Return the decoding policy and the temperature (input to generate) that corresponds to temperature=1 w.r.t. decoding policy

    NOTE
    1. The computation of temperature is only correct when beta = beta_1 = beta_2.
    2. We do not use any system prompt for now. 
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_pth_base)
    if 'llama'.lower() in model_pth_base.lower() and 'chat' not in model_pth_base.lower():
        print('\nUsing llama base model (not the chat version). Not using instruction template in llama chat.\n')
        prompt_template_base = lambda system_prompt, input_string: f"<s> {input_string}\n" # for Llama-2. Not using instruction template in llama chat.
    elif 'Llama-2-70B-Chat'.lower() in model_pth_base.lower():
        print('\nUsing Llama-2-70B-Chat as the base model.\n')
        prompt_template_base = lambda system_prompt, input_string: f"<s>[INST] {input_string} [/INST]\n" # Llama-2-70B. No system prompt according to AlpacaEval
    elif 'alpaca'.lower() in model_pth_base.lower() and '65B'.lower() in model_pth_base.lower():
        print('\nUsing Alpaca-65B as the base model.\n')
        prompt_template_base = lambda system_prompt, input_string: f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_string}\n\n### Response:\n"
    else:
        assert 'Llama'.lower() not in model_pth_base.lower()
        prompt_template_base = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string) 

    prompt_template_reward = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string) 

    # note: the reward model needs to use the tokenizer of the base model for the following reasons
    # when base = llama (32000 vocab size) and arm = alpaca (might have 32001 tokens), the model_arithmetic will truncate to the tokenizer vocab size (32000) to make it work. 
    M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_base, model=model_pth_base, tokenizer=tokenizer) 
    M_reward_1 = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_reward, model=model_pth_reward_1, tokenizer=tokenizer) if alpha_1 != 0 else None
    M_reward_2 = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_reward, model=model_pth_reward_2, tokenizer=tokenizer) if alpha_2 != 0 else None
 
    if alpha_1 != 0 and alpha_2 != 0:
        formula = M_base + alpha_1 * M_reward_1 + alpha_2 * M_reward_2
        print(f'\nModel Arithmetic: M_base + {alpha_1} * M_reward_1 + {alpha_2} * M_reward_2')
    elif alpha_1 == 0 and alpha_2 == 0:
        formula = M_base
        print(f'\nModel Arithmetic: M_base Only.')
    elif alpha_1 != 0 and alpha_2 == 0:
        formula = M_base + alpha_1 * M_reward_1
        print(f'\nModel Arithmetic: M_base + {alpha_1} * M_reward_1; M_reward_2 is not used')
    elif alpha_1 == 0 and alpha_2 != 0:
        formula = M_base + alpha_2 * M_reward_2
        print(f'\nModel Arithmetic: M_base + {alpha_2} * M_reward_2; M_reward_1 is not used')
    else:
        raise ValueError('Check alpha_1 and alpha_2')
    
    model = ModelArithmetic(formula, max_length = args.max_length) 
    temperature = 1 / (1 + alpha_1 + alpha_2)

    return model, tokenizer, temperature

################ Model Arithmetic Ends ################

################ Rewarded Soups ################
def get_rewarded_soups(peft_names: list, coefficients:list, args):
      '''
      peft_names: a list of Lora fine-tuned model paths (from the same pretrained model), customized to each preference dimension. 
      coefficients: a list of cofficients for each peft model. 

      rewarded soups average the lora weights of the peft models using the coefficients. 
      '''
      assert len(peft_names) == len(coefficients)
      assert 'Llama'.lower() not in peft_names[0].lower() # all models should be based on alpaca
      adapter_name = "default" # this is a argument of get_peft_model_state_dict function, which is "default" by default

      weights_averaged = OrderedDict()
      i = 0
      for peft_name, coefficient in zip(peft_names, coefficients): 
            if coefficient == 0.:
                  continue
            if peft_name is None:
                  print("Skipping none peft_name")
                  continue
            current_model = AutoModelForCausalLM.from_pretrained(peft_name, trust_remote_code=True, torch_dtype=torch.bfloat16, use_auth_token=True, device_map="auto") 
            current_weights = get_peft_model_state_dict(current_model, state_dict=None, adapter_name=adapter_name)
            for key in list(current_weights.keys()):
                  if i == 0:
                        weights_averaged[key] = coefficient * current_weights[key]
                  else:
                        weights_averaged[key] += coefficient * current_weights[key]
                  del current_weights[key]
            del current_model
            torch.cuda.empty_cache()
            i += 1

      # NOTE: we change the key names since "get_peft_model_state_dict" removes the adapter_name in the model key names
      new_weights_averaged = OrderedDict()
      for key, value in weights_averaged.items():
            assert 'lora' in key
            new_key = key.replace(".weight", f".{adapter_name}.weight")
            new_weights_averaged[new_key] = value
      del weights_averaged
      torch.cuda.empty_cache()
      wa = AutoModelForCausalLM.from_pretrained(peft_names[0], trust_remote_code=True, torch_dtype=torch.bfloat16, use_auth_token=True, device_map="auto")
      wa.load_state_dict(new_weights_averaged, strict=False) # NOTE: as a sanity check, we should manually set strict=True and see if there are any unexpected keys. If there are, we should manually inspect key names of new_weights_averaged.

      # convert to model_arithmetic
      prompt_template = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string)  # assume using alpaca-based model
      model = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template, model=wa) 
      formula = model
      model = ModelArithmetic(formula, max_length = args.max_length) 
      model.eval()

      tokenizer = AutoTokenizer.from_pretrained(peft_names[0])
      temperature = 1

      return model, tokenizer, temperature
################ Rewarded Soups Ends################

if __name__ == '__main__':

    args = parse_arguments()

    ##################### output path #####################
    out_path = Path(os.path.join(args.output_dir, 'generation') + f".json")
    if out_path.exists() and (not args.resume):
        print("ERROR: out_path already exists! If resuming, use --resume flag.")
        exit(1)
    if not out_path.exists() and args.resume:
        print(f"ERROR: Resuming but out_path DOESN'T exist! {out_path}")
        exit(1)
    print(f'Generation results will be saved in {out_path}')

    ##################### Load dataset #####################
    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)

    ##################### Load models #####################
    if args.rewarded_soups:
        print('\n\nUsing Rewarded Soups')
        model, tokenizer, temperature = get_rewarded_soups(peft_names=[args.model_arm_helpfulness_name_or_path, args.model_arm_harmlessness_name_or_path], 
                                                           coefficients=[args.alpha_helpfulness,args.alpha_harmlessness], args=args)
    else:
        model, tokenizer, temperature = get_model_arithmetic(model_pth_base=args.model_base_name_or_path, 
                                                            model_pth_reward_1=args.model_arm_helpfulness_name_or_path, 
                                                            model_pth_reward_2=args.model_arm_harmlessness_name_or_path, 
                                                            alpha_1=args.alpha_helpfulness, alpha_2=args.alpha_harmlessness, args=args)
    model.eval()
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0 in the model_arithmetic generation; The logit weights of base models and potential ARMs are normalized.\n')
        temperature = 1.0
    # generate function will remove eos token
    generate = lambda prompt: model.generate_text(prompt, max_new_tokens=args.max_new_tokens, batch_size=None, temperature=temperature, top_p=1, top_k=0, do_speculation=False)[0].removesuffix(tokenizer.eos_token)
    
    # get model name
    if args.alpha_helpfulness != 0 and args.alpha_harmlessness != 0:
        model_name = f'Base_{args.model_name_for_logging}_Helpfulness_{args.model_arm_helpfulness_name_for_logging}_Harmlessness_{args.model_arm_harmlessness_name_for_logging}_Alpha_{args.alpha_helpfulness}_{args.alpha_harmlessness}'
    elif args.alpha_helpfulness == 0 and args.alpha_harmlessness == 0:
        model_name = f'{args.model_name_for_logging}'
    elif args.alpha_helpfulness != 0 and args.alpha_harmlessness == 0:
        model_name = f'Base_{args.model_name_for_logging}_Helpfulness_{args.model_arm_helpfulness_name_for_logging}_Alpha_{args.alpha_helpfulness}'
    elif args.alpha_helpfulness == 0 and args.alpha_harmlessness != 0:
        model_name = f'Base_{args.model_name_for_logging}_Harmlessness_{args.model_arm_harmlessness_name_for_logging}_Alpha_{args.alpha_harmlessness}'
    else:
        raise ValueError('Check alpha_1 and alpha_2')
    
    if args.rewarded_soups:
        assert args.alpha_helpfulness != 0 and args.alpha_harmlessness != 0
        model_name = f'RewardedSoups_Helpfulness_{args.model_arm_helpfulness_name_for_logging}_Harmlessness_{args.model_arm_harmlessness_name_for_logging}_Alpha_{args.alpha_helpfulness}_{args.alpha_harmlessness}'

    if args.normalize_logit:
        model_name += '_NormalizedLogit'

    print(f'\nModel Name: {model_name}')

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
    for i in tqdm(range(start_index, len(data_evaluation))):
        prompt = data_evaluation[i]['prompt']
        # prompt_input = PROMPT_INPUT.format(input=prompt) # NOTE: the template is important
        prompt_input = prompt # the template is already used when initializing model arithmetic. 

        start = time.time()
        response = generate(prompt_input)
        elapsed = time.time() -start

        output_set.append({
            "uid": data_evaluation[i]['uid'],
            "prompt": prompt, 
            "response": response,
            "model": model_name,
            "elapsed":elapsed,
            })

        if i % 3 == 1:
            print(f'''Generated {i} outputs.\n Prompt: {output_set[-1]["prompt"]}\n response: {output_set[-1]["response"]}''')
            # Save intermediate results to avoid data loss
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output_set, f, ensure_ascii=False, indent=4)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)
    print(f'Generating responses finished!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/ 3600} hours for {len(output_set) - start_index} outputs')
