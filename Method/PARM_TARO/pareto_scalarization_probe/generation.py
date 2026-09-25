"""Resume-safe greedy generation from the frozen epoch-1 PBLoRA checkpoint."""
from __future__ import annotations
import gc, json, time
from typing import Any
import torch
from PARM_TARO.recovery.low_vram_suite.common import get_gpu_status, load_tulu_pblora_4bit, require_free_vram, set_requested_alpha
from .config import ADAPTER, DENSE_OUT, MAX_NEW_TOKENS, SEED
from .io import atomic_json, atomic_jsonl, response_key, sha256_file
STATE = DENSE_OUT / "generation_state"

def generate_one(model: Any, tokenizer: Any, prompt: str, alpha: list[float], max_new_tokens: int) -> dict[str, Any]:
    set_requested_alpha(model, alpha); encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.cuda() for name, value in encoded.items()}; torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    token_ids = output[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    return {"response": tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "selected_token_ids": token_ids, "generation_length": len(token_ids), "latency_seconds": elapsed,
            "latency_per_token": elapsed / max(1, len(token_ids))}

def run(cases: list[dict[str, Any]], *, resume: bool, min_free_mib: int, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    STATE.mkdir(parents=True, exist_ok=True)
    if any(STATE.glob("*.json")) and not resume: raise RuntimeError("Generation progress exists; pass --resume")
    missing = [row for row in cases if not (resume and (STATE / f"{row['case_id']}.json").is_file())]
    before = sha256_file(ADAPTER / "adapter_model.safetensors"); memory = {"before": get_gpu_status()}; model = tokenizer = None
    if missing:
        require_free_vram(min_free_mib); model, base_view, tokenizer = load_tulu_pblora_4bit(); memory["loaded"] = get_gpu_status()
        try:
            for index, case in enumerate(missing, 1):
                result = generate_one(model, tokenizer, case["prompt"], case["requested_alpha"], max_new_tokens)
                row = {**case, **result, "seed": SEED, "max_new_tokens": max_new_tokens}; row["response_key"] = response_key(row["prompt"], row["response"])
                atomic_json(STATE / f"{case['case_id']}.json", row); print(f"[{index}/{len(missing)}] {case['case_id']}", flush=True)
        finally:
            del model, base_view, tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); memory["after_release"] = get_gpu_status()
    after = sha256_file(ADAPTER / "adapter_model.safetensors")
    if before != after: raise RuntimeError("Frozen PBLoRA checkpoint changed")
    rows = [json.loads((STATE / f"{case['case_id']}.json").read_text()) for case in cases]; atomic_jsonl(DENSE_OUT / "generations.jsonl", rows)
    summary = {"status": "COMPLETE", "cases": len(rows), "new_generations": len(missing), "greedy": True, "adapter_sha256": after, "memory": memory}
    atomic_json(DENSE_OUT / "generation_summary.json", summary); return summary
