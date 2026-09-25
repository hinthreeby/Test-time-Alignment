"""Deterministic fresh validation manifest, disjoint from prior diagnostics."""
from __future__ import annotations
import hashlib,json
from typing import Any
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_json,atomic_jsonl,read_jsonl
from .config import ALPHAS,DIAGNOSTIC_MANIFEST,MANIFEST_PROMPTS,OUT,VALIDATION

def build_manifest()->list[dict[str,Any]]:
    validation=json.loads(VALIDATION.read_text(encoding="utf-8"));excluded={str(row["sample_id"]) for row in read_jsonl(DIAGNOSTIC_MANIFEST)}
    fresh=[row for row in validation if str(row["sample_id"]) not in excluded][:MANIFEST_PROMPTS]
    if len(fresh)!=MANIFEST_PROMPTS or len({row["sample_id"] for row in fresh})!=MANIFEST_PROMPTS:raise ValueError("Cannot construct 200 unique fresh validation prompts")
    rows=[]
    for prompt_index,row in enumerate(fresh):
        for alpha_index,alpha in enumerate(ALPHAS):
            rows.append({"case_id":f"dg_p{prompt_index:03d}_a{alpha_index}","sample_id":str(row["sample_id"]),"validation_index":validation.index(row),"prompt":str(row["prompt"]),"prompt_sha256":hashlib.sha256(str(row["prompt"]).encode()).hexdigest(),"requested_alpha":list(alpha),"alpha_order":["helpfulness","harmlessness"]})
    path=OUT/"manifest.jsonl"
    if path.exists() and read_jsonl(path)!=rows:raise RuntimeError("Frozen disagreement-gate manifest differs from deterministic reconstruction")
    atomic_jsonl(path,rows);atomic_json(OUT/"manifest_summary.json",{"status":"PASS","source":str(VALIDATION),"split":"validation","selection":"first 200 validation prompts after excluding every Feasibility60 diagnostic prompt","excluded_diagnostic_prompts":len(excluded),"unique_prompts":MANIFEST_PROMPTS,"alphas":[list(x) for x in ALPHAS],"cases":len(rows),"test_set_used":False})
    return rows

def select_cases(rows:list[dict[str,Any]],num_prompts:int)->list[dict[str,Any]]:
    if not 1<=num_prompts<=MANIFEST_PROMPTS:raise ValueError("num-prompts must be in [1,200]")
    selected=[];seen=[]
    for row in rows:
        if row["sample_id"] not in seen:seen.append(row["sample_id"])
        if seen.index(row["sample_id"])<num_prompts:selected.append(row)
    return selected

