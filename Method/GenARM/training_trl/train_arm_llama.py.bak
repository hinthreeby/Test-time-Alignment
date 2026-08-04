'''
adapted from trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

Due to data processing steps (data format and model template), this script only supports training LLaMA-1 on the HH-RLHF dataset and training Alpaca on the PKU-SafeRLHF dataset.
For other dataset and models, we need to modify the data processing steps.
'''
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig
from trl.commands.cli_utils import TrlParser

from arm_trainer import ARMTrainer

@dataclass
class ARMConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.0,
        metadata={"help": "target reward margin gamma. The reward margin is reward_win - reward_lose - gamma, where \
                                                          reward_win (reward_lose) = beta * log pi_arm_win (lose)."},
    )
    length_normalization: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize the logprobs using the length of the response. length_normalization=True is not default for ARM and should only be used for testing purposes!"},
    )

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO/ARM training script.

    NOTE: other training arguments, such as learning rate and beta, should be set in the command line.
    They are included in DPOConfig, not here. ScriptArguments below are arguments that are not included in DPOConfig.
    """
    # used by DPO or ARM
    algorithm: Optional[str] = field(default="arm", metadata={"help": "algorithm to use [dpo, arm]"})
    model_name_or_path: Optional[str] = field(
        default="./sft/final_checkpoint",
        metadata={"help": "the location of the to-be-finetuned model name or path"},
    )

    # dataset
    preference_dataset: Optional[str] = field(
        default="HH_RLHF", metadata={"help": "the name of the preference dataset, which should be \
        in [\"HH_RLHF\", \"PKU_SafeRLHF_Harmlessness\", \"PKU_SafeRLHF_Helpfulness\"]. The specific HF dataset name is hard coded in preprocessing functions."},
    )

    # training
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    # model
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # others
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def get_HH_RLHF(
    split: str = "train",
    sanity_check: bool = False,
    num_proc=4,
) -> Dataset:
    """Load the HH-RLHF dataset "Dahoas/full-hh-rlhf" from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """ 
    dataset = load_dataset(
        "Dahoas/full-hh-rlhf",
        split=split,
        use_auth_token=True,
        num_proc=num_proc,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_PKU_SafeRLHF(
    dataset_name: str,
    sanity_check: bool = False,
    num_proc=4,
):
    """Load the PKU_SafeRLHF dataset convert it to the necessary format.

    The dataset is "PKU-Alignment/PKU-SafeRLHF-10K", which contains the first round of Safe RLHF training data (on Alpaca) with 10K instances
    It only has a train split, without a test split, so we "train_test_split". 

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Based on the dataset_name, decide whether to load the Safe or Helpful version of the dataset.
    """ 
    assert dataset_name in ["PKU_SafeRLHF_Harmlessness", "PKU_SafeRLHF_Helpfulness"], "dataset_name must be either PKU_SafeRLHF_Harmlessness or PKU_SafeRLHF_Helpfulness"
    version = "Safe" if dataset_name == "PKU_SafeRLHF_Harmlessness" else "Helpful"

    dataset = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF-10K",
        split="train",
        use_auth_token=True,
        num_proc=num_proc,
    )
    original_columns = dataset.column_names

    # split the data since originally there is no test split
    data_dict = dataset.train_test_split(test_size=0.05)
    train_dataset = data_dict["train"]; test_dataset = data_dict["test"]

    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))

    ##### chat template #####
    # this is from https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/configs/constants.py
    PROMPT_BEGIN_pku_safe_rlhf: str = 'BEGINNING OF CONVERSATION: '
    PROMPT_USER_pku_safe_rlhf: str = 'USER: {input} '
    PROMPT_ASSISTANT_pku_safe_rlhf: str = 'ASSISTANT:'  # should not have a space at the end

    def format_prompt_pku_safe_rlhf(
        input: str,  
        eos_token: str,
    ) -> str:
        assert isinstance(input, str), f'Unsupported type of `input`: {type(input)}. Expected: str.' 

        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

        if len(input) % 2 != 1:
            raise ValueError(
                'The length of `input` must be odd, while `input` must end at the user question.',
            )

        buffer = [PROMPT_BEGIN_pku_safe_rlhf]
        for i, line in enumerate(input):
            if i % 2 == 0:
                # User input
                buffer.extend((PROMPT_USER_pku_safe_rlhf.format(input=line), PROMPT_ASSISTANT_pku_safe_rlhf))
            else:
                # Assistant response
                buffer.extend((line, eos_token))

        return ''.join(buffer)
    ##### chat template ends #####


    def return_prompt_and_responses(sample,version) -> Dict[str, str]:
        assert version in ["Safe", "Helpful"], "version must be either Safe or Helpful"
        if version == "Safe":
            return {
                "prompt": format_prompt_pku_safe_rlhf(input=sample["prompt"], eos_token='Not used'), 
                "chosen": sample[f"response_{sample['safer_response_id']}"],
                "rejected": sample[f"response_{int(1 - sample['safer_response_id'])}"],
            }
        else:
            return {
                "prompt": format_prompt_pku_safe_rlhf(input=sample["prompt"], eos_token='Not used'),
                "chosen": sample[f"response_{sample['better_response_id']}"],
                "rejected": sample[f"response_{int(1 - sample['better_response_id'])}"],
            }
        
    return_prompt_and_responses_with_version = lambda x: return_prompt_and_responses(x,version)

    # need to set batched=False because return_prompt_and_responses_with_version can only be applied to a single sample
    return train_dataset.map(
        return_prompt_and_responses_with_version,
        batched=False, 
        num_proc=num_proc,
        remove_columns=original_columns,
    ), test_dataset.map(
        return_prompt_and_responses_with_version,
        batched=False,
        num_proc=num_proc,
        remove_columns=original_columns,
    )



if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, ARMConfig))
    script_args, training_args = parser.parse_args_and_config()

    assert script_args.preference_dataset in ["HH_RLHF", "PKU_SafeRLHF_Harmlessness", "PKU_SafeRLHF_Helpfulness"], f"Invalid preference dataset: {script_args.preference_dataset}"
    print(f'\nPreference dataset: {script_args.preference_dataset}\n')

    if script_args.algorithm == "arm":
        print('\n**********************************')
        print(f'Using ARM Trainer. gamma = {training_args.gamma}, length_normalization = {training_args.length_normalization}')
        print('**********************************')

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False} # this is necessary due to https://github.com/huggingface/trl/issues/480

    assert script_args.algorithm in ["dpo", "arm"], "algorithm must be either dpo or arm"
    if training_args.gamma > 0:
        assert training_args.loss_type == "sigmoid", "gamma (the target_reward_margin) is only supported for sigmoid loss."

    set_seed(training_args.seed)

    # 1. Load the preference dataset
    if script_args.preference_dataset == "HH_RLHF":
        train_dataset = get_HH_RLHF(split="train", sanity_check=script_args.sanity_check) 
        eval_dataset = get_HH_RLHF(split="test", sanity_check=True)
    elif script_args.preference_dataset in ["PKU_SafeRLHF_Harmlessness", "PKU_SafeRLHF_Helpfulness"]:
        train_dataset, eval_dataset = get_PKU_SafeRLHF(dataset_name=script_args.preference_dataset, sanity_check=script_args.sanity_check)
    else:
        raise ValueError(f"Invalid preference dataset: {script_args.preference_dataset}")

    print(f'\nBefore filtering. Train data size: {train_dataset.num_rows}, Test data size: {eval_dataset.num_rows}\n')

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    print(f'After filtering. Train data size: {train_dataset.num_rows}, Test data size: {eval_dataset.num_rows}\n')

    # 2. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    print(f'\n model_name_or_path: {script_args.model_name_or_path}\n')
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. initialize training arguments (done in DPOconfig) and peft config
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. initialize the DPO/ARM trainer
    if script_args.algorithm == "arm":
        print('\n**********************************')
        print('Using ARM Trainer')
        print('**********************************\n')
        trainer = ARMTrainer(
            model=model,
            ref_model=None, # if not using peft, pass the model to bypass DPO Trainer check for ref model but is NOT actually used; if using peft (which is the case here), just pass None
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        print('\n**********************************')
        print('Using DPO Trainer')
        print('**********************************\n')
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    # 5. train
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # 6. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
