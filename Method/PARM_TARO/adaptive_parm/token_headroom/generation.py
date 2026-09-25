"""Autoregressive reference trajectories and one-step counterfactual rollouts."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from PARM_TARO.adaptive_parm.fusion import adaptive_parm_logits
from PARM_TARO.adaptive_parm.prompt_router.feature_extraction import compute_prompt_features
from PARM_TARO.recovery.low_vram_suite.common import (
    get_gpu_status,
    load_tulu_pblora_4bit,
    require_free_vram,
    set_requested_alpha,
)

from .config import (
    ADAPTER,
    GENERATION_STATE,
    LEAKAGE_STATE,
    LEAKAGE_TOLERANCE,
    MAX_REFERENCE_TOKENS,
    OUT,
    REFERENCE_WEIGHT,
    ROLLOUT_HORIZON,
    STATE_FRACTIONS,
    TOP_K,
    WEIGHTS,
)
from .io import atomic_csv, atomic_json, atomic_jsonl, read_jsonl, response_key, sha256_file


def select_state_positions(length: int) -> list[int]:
    """Select up to five unique relative positions on a valid trajectory."""
    if length <= 0: return []
    return sorted({min(length - 1, max(0, round((length - 1) * fraction))) for fraction in STATE_FRACTIONS})


def _full_logprobs(model: Any, token_ids: Sequence[int], *, output_index: int = -1) -> torch.Tensor:
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device="cuda")
    attention = torch.ones_like(ids)
    position = attention.cumsum(-1) - 1
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=attention, position_ids=position, use_cache=False)
        result = torch.log_softmax(output.logits[0, output_index].float(), -1).cpu().clone()
    del ids, attention, position, output
    return result


def _generate_guide(model: Any, start_ids: Sequence[int], max_new_tokens: int, eos_id: int | None) -> list[int]:
    if max_new_tokens <= 0: return []
    full = torch.tensor([list(start_ids)], dtype=torch.long, device="cuda")
    attention = torch.ones_like(full)
    position = attention.cumsum(-1) - 1
    generated: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=full, attention_mask=attention, position_ids=position, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1].float()
        del output
        for step in range(max_new_tokens):
            token = logits.argmax(-1, keepdim=True)
            token_id = int(token.item()); generated.append(token_id)
            if eos_id is not None and token_id == eos_id: break
            if step + 1 == max_new_tokens: break
            attention = torch.cat((attention, torch.ones((1, 1), dtype=attention.dtype, device=attention.device)), -1)
            next_position = torch.full((1, 1), attention.shape[1] - 1, dtype=torch.long, device=attention.device)
            output = model(input_ids=token, attention_mask=attention, position_ids=next_position, past_key_values=past, use_cache=True)
            past = output.past_key_values; logits = output.logits[:, -1].float(); del output
    del full, attention, position, past, logits
    return generated


def _leakage_probe(model: Any, base_view: Any, prefix: list[int], future: list[int]) -> dict[str, Any]:
    if not future: future = [2, 2, 2]
    alternative = list(reversed(future))
    if alternative == future: alternative = [2 if token != 2 else 1 for token in future]
    index = len(prefix) - 1
    base_reference = _full_logprobs(base_view, prefix)
    guide_reference = _full_logprobs(model, prefix)
    base_a = _full_logprobs(base_view, prefix + future, output_index=index)
    base_b = _full_logprobs(base_view, prefix + alternative, output_index=index)
    guide_a = _full_logprobs(model, prefix + future, output_index=index)
    guide_b = _full_logprobs(model, prefix + alternative, output_index=index)
    errors = {
        "base_prefix_vs_future_a": float((base_reference - base_a).abs().max()),
        "base_prefix_vs_future_b": float((base_reference - base_b).abs().max()),
        "guide_prefix_vs_future_a": float((guide_reference - guide_a).abs().max()),
        "guide_prefix_vs_future_b": float((guide_reference - guide_b).abs().max()),
    }
    maximum = max(errors.values())
    return {"status": "PASS" if maximum <= LEAKAGE_TOLERANCE else "FAIL", "tolerance": LEAKAGE_TOLERANCE, "max_abs_logprob_diff": maximum, "comparisons": errors, "future_tokens_not_used_by_operational_path": True}


def _case_payload(case: dict[str, Any], model: Any, base_view: Any, tokenizer: Any, prompt_index: int, alpha_index: int) -> dict[str, Any]:
    alpha = [float(value) for value in case["requested_alpha"]]
    set_requested_alpha(model, alpha)
    encoded = tokenizer(case["prompt"], return_tensors="pt", add_special_tokens=False)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if not prompt_ids: raise ValueError(f"Empty prompt encoding: {case['case_id']}")
    reference = _generate_guide(model, prompt_ids, MAX_REFERENCE_TOKENS, tokenizer.eos_token_id)
    positions = select_state_positions(len(reference))
    if not positions: raise RuntimeError(f"Reference trajectory is empty: {case['case_id']}")

    audit = None
    if prompt_index % 5 == alpha_index:
        audit_position = positions[len(positions) // 2]
        prefix = prompt_ids + reference[:audit_position]
        future = reference[audit_position : audit_position + 4]
        audit = {"case_id": case["case_id"], "sample_id": case["sample_id"], "alpha": alpha, "state_position": audit_position, **_leakage_probe(model, base_view, prefix, future)}
        atomic_json(LEAKAGE_STATE / f"{case['sample_id']}.json", audit)
        if audit["status"] != "PASS": raise RuntimeError(f"LEAKAGE_AUDIT_FAIL: {audit}")

    states=[]; rollouts=[]
    for state_index, position in enumerate(positions):
        prefix_response = reference[:position]
        full_prefix = prompt_ids + prefix_response
        base = _full_logprobs(base_view, full_prefix)
        guide = _full_logprobs(model, full_prefix)
        features = compute_prompt_features(base, guide, prompt_token_length=len(prompt_ids), top_k=TOP_K)
        state_id = f"{case['case_id']}_s{state_index:02d}"
        action_tokens = {weight: int(adaptive_parm_logits(base, guide, weight).argmax()) for weight in WEIGHTS}
        all_same = len(set(action_tokens.values())) == 1
        generated_by_action: dict[int, list[int]] = {}
        for token in sorted(set(action_tokens.values())):
            remainder = [] if token == tokenizer.eos_token_id else _generate_guide(model, full_prefix + [token], ROLLOUT_HORIZON - 1, tokenizer.eos_token_id)
            generated_by_action[token] = [token] + remainder
        states.append({
            "state_id": state_id, "case_id": case["case_id"], "sample_id": case["sample_id"], "prompt": case["prompt"],
            "alpha_helpfulness": alpha[0], "alpha_harmlessness": alpha[1], "reference_position": position,
            "normalized_generation_position": position / max(1, len(reference) - 1), "generated_token_count": position,
            "prompt_token_count": len(prompt_ids), "prefix_token_ids": json.dumps(full_prefix), "reference_length": len(reference),
            **{key:value for key,value in features.items() if key != "prompt_token_length"},
            "all_weights_same_next_token": all_same,
        })
        for weight in WEIGHTS:
            token = action_tokens[weight]; continuation = generated_by_action[token]
            response_ids = prefix_response + continuation
            response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            rollouts.append({
                "rollout_id": f"{state_id}_w{weight:.2f}", "state_id": state_id, "case_id": case["case_id"], "sample_id": case["sample_id"],
                "prompt": case["prompt"], "alpha_helpfulness": alpha[0], "alpha_harmlessness": alpha[1], "weight": weight,
                "reference_weight": REFERENCE_WEIGHT, "action_span": 1, "rollout_horizon": ROLLOUT_HORIZON,
                "candidate_next_token_id": token, "all_weights_same_next_token": all_same,
                "response": response, "response_token_ids": response_ids, "response_key": response_key(case["prompt"], response),
            })
        del base, guide
    return {"case":case,"reference_token_ids":reference,"reference_text":tokenizer.decode(reference,skip_special_tokens=True,clean_up_tokenization_spaces=False),"states":states,"rollouts":rollouts,"leakage_audit":audit}


def combine_generation_outputs(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    payloads=[]
    for case in manifest:
        path=GENERATION_STATE/f"{case['case_id']}.json"
        if not path.is_file(): raise RuntimeError(f"Missing generation state: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    states=[row for payload in payloads for row in payload["states"]]
    rollouts=[row for payload in payloads for row in payload["rollouts"]]
    atomic_csv(OUT/"state_features.csv",states); atomic_jsonl(OUT/"counterfactual_rollouts.jsonl",rollouts)
    audits=[json.loads(path.read_text()) for path in sorted(LEAKAGE_STATE.glob("*.json"))]
    leakage={"status":"PASS" if len(audits)==10 and all(row["status"]=="PASS" for row in audits) else "FAIL","num_sampled_states":len(audits),"tolerance":LEAKAGE_TOLERANCE,"max_abs_logprob_diff":max((row["max_abs_logprob_diff"] for row in audits),default=None),"audits":audits}
    atomic_json(OUT/"leakage_audit.json",leakage)
    if leakage["status"]!="PASS": raise RuntimeError(f"Leakage audit incomplete/failed: {leakage}")
    return {"cases":len(payloads),"states":len(states),"rollouts":len(rollouts),"unique_responses":len({row['response_key'] for row in rollouts}),"leakage":leakage}


def run_generation(manifest: list[dict[str, Any]], *, resume: bool, min_free_mib: int) -> dict[str, Any]:
    existing={path.stem for path in GENERATION_STATE.glob("*.json")} if GENERATION_STATE.is_dir() else set()
    if existing and not resume: raise RuntimeError(f"Generation progress exists; use --resume: {GENERATION_STATE}")
    missing=[case for case in manifest if case["case_id"] not in existing]
    adapter_before=sha256_file(ADAPTER/"adapter_model.safetensors")
    if missing:
        require_free_vram(min_free_mib)
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable in this environment")
        model=base_view=tokenizer=None
        try:
            model,base_view,tokenizer=load_tulu_pblora_4bit()
            prompt_order={sample:index for index,sample in enumerate(dict.fromkeys(case["sample_id"] for case in manifest))}
            alpha_order={tuple(alpha):index for index,alpha in enumerate(((1.,0.),(.75,.25),(.5,.5),(.25,.75),(0.,1.)))}
            for index,case in enumerate(missing,1):
                payload=_case_payload(case,model,base_view,tokenizer,prompt_order[case["sample_id"]],alpha_order[tuple(case["requested_alpha"])])
                atomic_json(GENERATION_STATE/f"{case['case_id']}.json",payload); print(f"[{index}/{len(missing)}] generated {case['case_id']}",flush=True)
        finally:
            if model is not None: del model,base_view,tokenizer
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    adapter_after=sha256_file(ADAPTER/"adapter_model.safetensors")
    if adapter_before!=adapter_after: raise RuntimeError("Frozen PBLoRA checkpoint changed")
    summary=combine_generation_outputs(manifest)
    summary.update({"status":"PASS","adapter_sha256":adapter_after,"pblora_frozen":True,"gpu_after":get_gpu_status()})
    atomic_json(OUT/"generation_summary.json",summary); return summary
