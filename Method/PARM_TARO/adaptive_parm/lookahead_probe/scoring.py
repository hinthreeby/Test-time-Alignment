"""Resume-safe internal sequence scoring with one frozen physical backbone."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv, atomic_json, read_jsonl, sha256_file
from PARM_TARO.recovery.low_vram_suite.common import (
    get_gpu_status, load_tulu_pblora_4bit, require_free_vram, set_requested_alpha,
)

from .config import ADAPTER, CACHE, HORIZONS, OUT, SOURCE


def load_source_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rollouts = read_jsonl(SOURCE / "counterfactual_rollouts.jsonl")
    with (SOURCE / "state_features.csv").open(newline="", encoding="utf-8") as handle:
        states = {row["state_id"]: row for row in csv.DictReader(handle)}
    with (SOURCE / "state_utility.csv").open(newline="", encoding="utf-8") as handle:
        utility = {row["rollout_id"]: row for row in csv.DictReader(handle)}
    return rollouts, states, utility


def continuation_tokens(rollout: Mapping[str, Any], state: Mapping[str, Any]) -> list[int]:
    """Return only tokens at/after the saved state; never any unseen future input."""
    response = [int(value) for value in rollout["response_token_ids"]]
    generated_before_state = int(state["generated_token_count"])
    if generated_before_state < 0 or generated_before_state >= len(response):
        raise ValueError(f"Invalid rollout boundary for {rollout['rollout_id']}")
    continuation = response[generated_before_state:]
    horizon = int(rollout["rollout_horizon"])
    if not continuation or len(continuation) > horizon:
        raise ValueError(f"Invalid continuation length for {rollout['rollout_id']}: {len(continuation)}")
    if continuation[0] != int(rollout["candidate_next_token_id"]):
        raise ValueError(f"Candidate token/boundary mismatch: {rollout['rollout_id']}")
    return continuation


def branch_key(prefix: Sequence[int], continuation: Sequence[int], alpha: Sequence[float]) -> str:
    payload = json.dumps({"prefix": list(prefix), "continuation": list(continuation), "alpha": list(alpha)}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_jobs(rollouts: Sequence[Mapping[str, Any]], states: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    jobs: dict[str, dict[str, Any]] = {}; rollout_to_key = {}
    for rollout in rollouts:
        state = states[str(rollout["state_id"])]
        prefix = [int(value) for value in json.loads(state["prefix_token_ids"])]
        continuation = continuation_tokens(rollout, state)
        alpha = [float(rollout["alpha_helpfulness"]), float(rollout["alpha_harmlessness"])]
        key = branch_key(prefix, continuation, alpha)
        jobs.setdefault(key, {"branch_key": key, "prefix_token_ids": prefix, "continuation_token_ids": continuation, "alpha": alpha})
        rollout_to_key[str(rollout["rollout_id"])] = key
    return jobs, rollout_to_key


def score_sequence(model: Any, prefix: Sequence[int], continuation: Sequence[int]) -> np.ndarray:
    import torch
    if not prefix or not continuation: raise ValueError("Both prefix and continuation are required")
    tokens = list(prefix) + list(continuation)
    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    attention = torch.ones_like(ids); position = attention.cumsum(-1) - 1
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=attention, position_ids=position, use_cache=False, return_dict=True)
        start = len(prefix) - 1; stop = start + len(continuation)
        logits = output.logits[0, start:stop].float()
        targets = ids[0, len(prefix):len(prefix) + len(continuation)]
        selected = torch.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1).detach().cpu().numpy().astype(np.float32, copy=True)
    del ids, attention, position, output, logits, targets
    if len(selected) != len(continuation) or not np.isfinite(selected).all():
        raise FloatingPointError("Invalid internal sequence scores")
    return selected


def cache_valid(path: Path, key: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["branch_key"] == key and len(value["base_token_logprobs"]) == len(value["parm_token_logprobs"]) > 0 and np.isfinite(value["base_token_logprobs"]).all() and np.isfinite(value["parm_token_logprobs"]).all()
    except Exception:
        return False


def horizon_means(values: Sequence[float]) -> dict[int, tuple[int, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all(): raise ValueError("Invalid token score vector")
    return {horizon: (min(horizon, len(array)), float(array[:horizon].mean())) for horizon in HORIZONS}


def combine_scores(rollouts: Sequence[Mapping[str, Any]], utility: Mapping[str, Mapping[str, Any]], rollout_to_key: Mapping[str, str]) -> list[dict[str, Any]]:
    rows=[]
    for rollout in rollouts:
        rid=str(rollout["rollout_id"]); key=rollout_to_key[rid]
        cached=json.loads((CACHE/f"{key}.json").read_text(encoding="utf-8")); base=horizon_means(cached["base_token_logprobs"]); parm=horizon_means(cached["parm_token_logprobs"])
        for horizon in HORIZONS:
            effective=base[horizon][0]
            rows.append({"rollout_id":rid,"state_id":rollout["state_id"],"case_id":rollout["case_id"],"sample_id":rollout["sample_id"],"alpha_helpfulness":float(rollout["alpha_helpfulness"]),"alpha_harmlessness":float(rollout["alpha_harmlessness"]),"weight":float(rollout["weight"]),"is_actionable":not bool(rollout["all_weights_same_next_token"]),"horizon":horizon,"effective_horizon":effective,"parm_score":parm[horizon][1],"base_score":base[horizon][1],"ratio_score":parm[horizon][1]-base[horizon][1],"mip":float(utility[rid]["mip"]),"branch_key":key})
    return rows


def extract(*, resume: bool, min_free_mib: int) -> dict[str, Any]:
    import torch
    rollouts,states,utility=load_source_records();jobs,mapping=build_jobs(rollouts,states);CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    missing=[job for key,job in jobs.items() if not (resume and cache_valid(CACHE/f"{key}.json",key))]
    adapter_before=sha256_file(ADAPTER/"adapter_model.safetensors");memory={"before":get_gpu_status()};model=base_view=tokenizer=None
    if missing:
        require_free_vram(min_free_mib);torch.cuda.reset_peak_memory_stats();model,base_view,tokenizer=load_tulu_pblora_4bit();memory["after_load"]={"allocated_mib":torch.cuda.memory_allocated()/1024**2,"reserved_mib":torch.cuda.memory_reserved()/1024**2,"nvidia_smi":get_gpu_status()}
        try:
            for index,job in enumerate(missing,1):
                set_requested_alpha(model,job["alpha"])
                base=score_sequence(base_view,job["prefix_token_ids"],job["continuation_token_ids"])
                parm=score_sequence(model,job["prefix_token_ids"],job["continuation_token_ids"])
                atomic_json(CACHE/f"{job['branch_key']}.json",{"branch_key":job["branch_key"],"alpha":job["alpha"],"num_tokens":len(parm),"base_token_logprobs":base.tolist(),"parm_token_logprobs":parm.tolist()})
                print(f"[{index}/{len(missing)}] scored branch {job['branch_key'][:12]}",flush=True)
        finally:
            memory["peak_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
            if model is not None: del model,base_view,tokenizer
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();memory["after_release"]={"allocated_mib":torch.cuda.memory_allocated()/1024**2,"reserved_mib":torch.cuda.memory_reserved()/1024**2,"nvidia_smi":get_gpu_status()}
    adapter_after=sha256_file(ADAPTER/"adapter_model.safetensors")
    if adapter_before != adapter_after: raise RuntimeError("Frozen PBLoRA checkpoint changed")
    rows=combine_scores(rollouts,utility,mapping);atomic_csv(OUT/"internal_scores.csv",rows)
    summary={"status":"PASS","rollouts":len(rollouts),"unique_branches":len(jobs),"newly_scored":len(missing),"rows":len(rows),"horizons":list(HORIZONS),"no_generation":True,"no_beaver_models":True,"prefix_only":True,"pblora_frozen":True,"adapter_sha256":adapter_after,"memory":memory};atomic_json(OUT/"extraction_summary.json",summary);return summary
