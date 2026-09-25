"""No-step gradient conflict audit on existing PBLoRA parameters only."""
from __future__ import annotations
import gc, json, math
from statistics import fmean, median
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime
from PARM_TARO.recovery.low_vram_suite.common import get_gpu_status, require_free_vram, set_requested_alpha
from PARM_TARO.training.data import MultiObjectiveExample, SequenceTooLongError, tokenize_response
from .config import ADAPTER, BASE, CANCELLATION_ALPHAS, GRADIENT_BATCH_SIZE, GRADIENT_OUT, NUM_GRADIENT_BATCHES, SEED, TRAIN
from .io import atomic_csv, atomic_json, sha256_file

def deterministic_batches(num_batches: int = NUM_GRADIENT_BATCHES, batch_size: int = GRADIENT_BATCH_SIZE) -> list[list[dict]]:
    rows = json.loads(TRAIN.read_text()); ranked = sorted(rows, key=lambda r: __import__("hashlib").sha256(f"{SEED}\0{r['sample_id']}".encode()).hexdigest())
    return [ranked[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

def load_trainable() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    vendored = activate_vendored_parm_runtime(); tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    backbone = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                    quantization_config=quant, device_map={"": 0})
    model = vendored.peft.PeftModel.from_pretrained(backbone, ADAPTER, is_trainable=True).eval(); model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("pblora_" not in n for n, _ in trainable): raise RuntimeError("Trainable set is not PBLoRA-only")
    return model, tokenizer

def _example(row: dict) -> MultiObjectiveExample:
    return MultiObjectiveExample(str(row["sample_id"]), int(row["source_index"]), str(row["prompt"]),
        (str(row["response_0"]), str(row["response_1"])), int(row["better_response_id"]), int(row["safer_response_id"]))

def _response_logp(model: Any, tokenized: Any) -> torch.Tensor:
    ids = tokenized.input_ids.cuda(); output = model(input_ids=ids, attention_mask=tokenized.attention_mask.cuda(),
        position_ids=tokenized.position_ids.cuda(), use_cache=False, return_dict=True)
    selected = output.logits[0].index_select(0, tokenized.logit_positions.cuda()).float()
    return torch.log_softmax(selected, -1).gather(-1, tokenized.gold_token_ids.cuda().unsqueeze(-1)).sum()

def separate_losses(model: Any, tokenizer: Any, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    set_requested_alpha(model, [0.5, 0.5]); safe_losses, help_losses = [], []
    for row in batch:
        example = _example(row)
        tokenized = [tokenize_response(tokenizer, example, i, max_length=512, max_continuation_tokens=511, include_eos_target=True) for i in (0, 1)]
        logps = torch.stack([_response_logp(model, item) for item in tokenized])
        safe = int(row["safer_response_id"]); help_id = int(row["better_response_id"])
        safe_losses.append(-F.logsigmoid(0.5 * (logps[safe] - logps[1-safe])))
        help_losses.append(-F.logsigmoid(0.5 * (logps[help_id] - logps[1-help_id])))
    return torch.stack(help_losses).mean(), torch.stack(safe_losses).mean()

def _flat(grads: tuple[torch.Tensor | None, ...], params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1) for g, p in zip(grads, params)])

def run(*, resume: bool, min_free_mib: int, num_batches: int = NUM_GRADIENT_BATCHES) -> dict:
    state = GRADIENT_OUT / "batch_state"; state.mkdir(parents=True, exist_ok=True)
    if any(state.glob("*.json")) and not resume: raise RuntimeError("Gradient progress exists; pass --resume")
    require_free_vram(min_free_mib); before = sha256_file(ADAPTER / "adapter_model.safetensors"); model, tokenizer = load_trainable()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]; params = [p for _, p in trainable]; memory = {"loaded": get_gpu_status()}
    try:
        for batch_index, batch in enumerate(deterministic_batches(num_batches), 1):
            path = state / f"batch_{batch_index:03d}.json"
            if resume and path.is_file(): continue
            try: help_loss, safe_loss = separate_losses(model, tokenizer, batch)
            except SequenceTooLongError as error: raise RuntimeError(f"Frozen audit batch is too long: {error}") from error
            gh = _flat(torch.autograd.grad(help_loss, params, retain_graph=True, allow_unused=True), params)
            gs = _flat(torch.autograd.grad(safe_loss, params, allow_unused=True), params)
            nh, ns = float(torch.linalg.vector_norm(gh)), float(torch.linalg.vector_norm(gs)); dot = float(torch.dot(gh, gs))
            cosine = dot / (nh*ns) if nh > 0 and ns > 0 else 0.0
            values = {str(alpha): float(torch.linalg.vector_norm(alpha*gh+(1-alpha)*gs) / max(1e-30, alpha*nh+(1-alpha)*ns)) for alpha in CANCELLATION_ALPHAS}
            atomic_json(path, {"batch": batch_index, "sample_ids": [r["sample_id"] for r in batch], "loss_help": float(help_loss.detach()),
                "loss_safe": float(safe_loss.detach()), "cosine_similarity": cosine, "norm_help": nh, "norm_safe": ns, "cancellation": values})
            del help_loss, safe_loss, gh, gs; model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); print(f"[{batch_index}/{num_batches}] gradient batch", flush=True)
    finally:
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); memory["after_release"] = get_gpu_status()
    if sha256_file(ADAPTER / "adapter_model.safetensors") != before: raise RuntimeError("Checkpoint changed during no-step audit")
    rows = [json.loads((state / f"batch_{i:03d}.json").read_text()) for i in range(1, num_batches+1)]
    flat_rows = [{k: v for k, v in row.items() if k != "cancellation"} | {f"cancellation_{a}": row["cancellation"][str(a)] for a in CANCELLATION_ALPHAS} for row in rows]
    atomic_csv(GRADIENT_OUT / "gradient_conflict.csv", flat_rows)
    def stats(values: list[float]) -> dict: return {"mean": fmean(values), "median": median(values), "std": float(np.std(values)),
        "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75))}
    cosines = [r["cosine_similarity"] for r in rows]
    summary = {"status": "COMPLETE", "batches": num_batches, "batch_size": GRADIENT_BATCH_SIZE, "optimizer_steps": 0,
        "preference_used_for_conditioning": [0.5, 0.5], "trainable_parameter_count": sum(p.numel() for _, p in trainable),
        "all_trainable_parameters_pblora": all("pblora_" in n for n, _ in trainable), "cosine_similarity": stats(cosines),
        "fraction_cosine_below_zero": sum(x < 0 for x in cosines)/len(cosines), "norm_help": stats([r["norm_help"] for r in rows]),
        "norm_safe": stats([r["norm_safe"] for r in rows]), "cancellation_by_alpha": {str(a): stats([r["cancellation"][str(a)] for r in rows]) for a in CANCELLATION_ALPHAS},
        "adapter_sha256_before_after": before, "memory": memory}
    atomic_json(GRADIENT_OUT / "gradient_summary.json", summary); return summary
