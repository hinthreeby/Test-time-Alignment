import torch
from argsearch import ARGS

LLM_PATH = "../models/gpt2-large"
RM_PATH = "../models/gpt2-large-helpful-rm"

searcher = ARGS(
    llm_path=LLM_PATH,
    rm_path=RM_PATH,
    llm_dev="cuda:0",
    rm_dev="cuda:0",
    torch_dtype=torch.float16,
)

prompt = "The best way to learn programming is"

output_tokens = searcher.generate(
    prompt=prompt,
    topk=5,
    weight=1.0,
    max_new_token=30,
    method="greedy",
    debug=False,
)

output_text = searcher.tokens_to_text(output_tokens)[0]
print(output_text)