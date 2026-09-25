"""Sequential, resume-safe Beaver scoring for counterfactual rollouts."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from PARM_TARO.recovery.low_vram_suite.common import (
    activate_safe_rlhf_score_runtime,
    get_gpu_status,
    require_free_vram,
    resolve_safe_rlhf_source,
)

from .config import COST, OUT, REWARD, SCORE_STATE
from .io import atomic_csv, atomic_json, read_jsonl


def _load(model_path: Path, resolution: Mapping[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoTokenizer, BitsAndBytesConfig

    auto_model=activate_safe_rlhf_score_runtime(Path(resolution["path"]))
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=auto_model.from_pretrained(model_path,local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.float16,quantization_config=quantization,device_map={"":0}).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model,tokenizer


def _build_scoring_summary(
    *,
    kind: str,
    model_path: Path,
    rollout_count: int,
    scores: Mapping[str, float],
    memory: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Build bookkeeping metadata without touching scorer weights.

    Provenance keys other than ``path`` are optional because older recovered
    Safe-RLHF trees may not carry Git metadata. Missing values stay ``None``;
    they are never inferred or fabricated.
    """
    source = resolution.get("path")
    return {
        "status": "PASS",
        "kind": kind,
        "model_path": str(model_path),
        "rollouts": rollout_count,
        "unique_responses": len(scores),
        "finite": all(math.isfinite(float(value)) for value in scores.values()),
        "memory": dict(memory),
        "score_semantics": "higher reward is more helpful; higher cost is more harmful",
        "safe_rlhf_source": str(source) if source is not None else None,
        "safe_rlhf_commit": resolution.get("git_commit"),
        "safe_rlhf_provenance": resolution.get("provenance_status"),
    }


def run_scoring(kind: str, *, resume: bool, min_free_mib: int=6000) -> dict[str,Any]:
    if kind not in {"reward","cost"}: raise ValueError(kind)
    resolution=resolve_safe_rlhf_source()
    if not resolution.get("ready") or not resolution.get("path"):
        raise RuntimeError(f"Safe-RLHF runtime unavailable: {resolution}")
    rollouts=read_jsonl(OUT/"counterfactual_rollouts.jsonl")
    if not rollouts: raise RuntimeError("No counterfactual rollouts")
    unique={row["response_key"]:{"prompt":row["prompt"],"response":row["response"]} for row in rollouts}
    state_dir=SCORE_STATE/kind; existing={path.stem for path in state_dir.glob("*.json")} if state_dir.is_dir() else set()
    if existing and not resume: raise RuntimeError(f"Score progress exists; use --resume: {state_dir}")
    missing=[key for key in sorted(unique) if key not in existing]
    model_path=REWARD if kind=="reward" else COST
    memory={"before":get_gpu_status()}; model=tokenizer=None
    if missing:
        require_free_vram(min_free_mib); model,tokenizer=_load(model_path,resolution); memory["loaded"]=get_gpu_status()
        try:
            with torch.inference_mode():
                for index,key in enumerate(missing,1):
                    row=unique[key]; text=f"BEGINNING OF CONVERSATION: USER: {row['prompt']} ASSISTANT:{row['response']}"
                    encoded=tokenizer(text,return_tensors="pt",truncation=True,max_length=512); encoded={name:value.cuda() for name,value in encoded.items()}
                    output=model(**encoded); end=output["end_scores"] if isinstance(output,dict) else output.end_scores; value=float(end.detach().float().reshape(-1)[0].cpu())
                    if not torch.isfinite(torch.tensor(value)): raise ValueError(f"Non-finite {kind} score")
                    atomic_json(state_dir/f"{key}.json",{"response_key":key,"score":value}); del encoded,output,end
                    print(f"[{index}/{len(missing)}] {kind} {key[:12]}",flush=True)
        finally:
            if model is not None: del model,tokenizer
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); memory["after_release"]=get_gpu_status()
    scores={key:float(json.loads((state_dir/f"{key}.json").read_text())["score"]) for key in unique}
    rows=[{"rollout_id":row["rollout_id"],"state_id":row["state_id"],"response_key":row["response_key"],f"{kind}_score":scores[row["response_key"]]} for row in rollouts]
    atomic_csv(OUT/f"{kind}_scores.csv",rows)
    summary=_build_scoring_summary(kind=kind,model_path=model_path,rollout_count=len(rows),scores=scores,memory=memory,resolution=resolution)
    atomic_json(OUT/f"{kind}_scoring_summary.json",summary); return summary
