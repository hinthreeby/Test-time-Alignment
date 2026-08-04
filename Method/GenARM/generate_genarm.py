import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from model_arithmetic import ModelArithmetic, PromptedLLM


BASE_DIR = Path(__file__).resolve().parents[2]

BASE_MODEL_PATH = BASE_DIR / "models" / "tulu-2-7b"
ARM_MODEL_PATH = BASE_DIR / "models" / "AutoregressiveRM-tulu2-7b"

INPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "rad_benchmark"
    / "negative_prompts.jsonl"
)

OUTPUT_FILE = BASE_DIR / "results" / "genarm.json"

NUM_PROMPTS = 100

MAX_NEW_TOKENS = 128

ALPHA = 2.0
TEMPERATURE = 1.0

TOP_P = 1.0
TOP_K = 0

SEED = 42
SAVE_EVERY = 1

def save_results(results: list[dict]) -> None:
    """Lưu kết quả ra JSON."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_dataset() -> list[dict]:
    """Đọc file JSONL của RAD."""
    samples = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON không hợp lệ tại dòng {line_number}: {error}"
                ) from error

    if NUM_PROMPTS is not None:
        samples = samples[:NUM_PROMPTS]

    return samples


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(BASE_MODEL_PATH),
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load cả hai model ở 4-bit để giảm VRAM.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


print("Loading Tulu-2 base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL_PATH),
    quantization_config=quantization_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

base_model.eval()


print("Loading autoregressive reward model...")

arm_model = AutoModelForCausalLM.from_pretrained(
    str(ARM_MODEL_PATH),
    quantization_config=quantization_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

arm_model.eval()


def prompt_template(system_prompt: str, input_string: str) -> str:
    return (
        f"<|user|>\n"
        f"{input_string}\n"
        f"<|assistant|>\n"
    )


M_base = PromptedLLM(
    system_prompt="Not used for Tulu-2",
    prompt_template=prompt_template,
    model=base_model,
    tokenizer=tokenizer,
    run_eager=True,
)

M_arm = PromptedLLM(
    system_prompt="Not used for Tulu-2",
    prompt_template=prompt_template,
    model=arm_model,
    tokenizer=tokenizer,
    run_eager=True,
)


# GenARM formula:
# Base LM + alpha × Autoregressive Reward Model
formula = M_base + ALPHA * M_arm

genarm = ModelArithmetic(
    formula,
    needs_input_tokens_lm_eval=False,
    lm_eval_task=None,
    dtype=torch.float16,
)

dataset = load_dataset()

print(f"Loaded {len(dataset)} prompts.")
print(f"Output: {OUTPUT_FILE}")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

results = []

progress_bar = tqdm(
    dataset,
    total=len(dataset),
    desc="GenARM decoding",
    unit="prompt",
)

for index, sample in enumerate(progress_bar):
    prompt = sample["prompt"]["text"]

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    try:
        outputs = genarm.generate_text(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,

            # README của GenARM dùng temperature / (1 + alpha).
            temperature=TEMPERATURE / (1.0 + ALPHA),

            top_p=TOP_P,
            top_k=TOP_K,
            do_speculation=False,
        )

        response = outputs[0]

        if tokenizer.eos_token:
            response = response.removesuffix(tokenizer.eos_token)

        response = response.strip()
        status = "success"
        error_message = None

    except Exception as error:
        response = None
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"

    torch.cuda.synchronize()
    latency = time.perf_counter() - start_time

    result = {
        "id": index,
        "md5_hash": sample.get("md5_hash"),
        "prompt": prompt,
        "reference": sample.get("continuation", {}).get("text"),
        "response": response,
        "num_positive": sample.get("num_positive"),

        "method": "GenARM",
        "base_model": str(BASE_MODEL_PATH),
        "reward_model": str(ARM_MODEL_PATH),

        "alpha": ALPHA,
        "temperature": TEMPERATURE,
        "effective_temperature": TEMPERATURE / (1.0 + ALPHA),
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,

        "latency": round(latency, 4),
        "status": status,
        "error": error_message,
    }

    results.append(result)

    progress_bar.set_postfix(
        latency=f"{latency:.2f}s",
        status=status,
    )

    if len(results) % SAVE_EVERY == 0:
        save_results(results)


save_results(results)

print(f"\nFinished. Saved {len(results)} samples to:")
print(OUTPUT_FILE)