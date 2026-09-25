"""Causal frozen feature extraction; never consumes responses or scorer data."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from PARM_TARO.recovery.low_vram_suite.common import (
    get_gpu_status, load_tulu_pblora_4bit, require_free_vram,
    set_requested_alpha, sha256_file,
)

from .config import ADAPTER, CACHE, OUT, SOURCE, TOP_K_VALUES, WEIGHTS

INFERENCE_SCALAR_COLUMNS=("alpha_helpfulness","alpha_harmlessness","normalized_generation_position","generated_token_count","prompt_token_count","base_entropy","base_top1_probability","base_top1_top2_margin","base_topk_mass","parm_entropy","parm_top1_probability","parm_top1_top2_margin","parm_topk_mass","top1_disagreement","topk_overlap","js_divergence","symmetric_kl","logprob_correlation","candidate_rank_disagreement")
ACTION_MODEL_COLUMNS=("top1_probability","entropy","top1_top2_margin","top1_differs_from_w1","w1_top_token_rank","w1_top_token_probability_change","top1_logprob_margin_change_vs_w1")


def atomic_npz(path:Path,**arrays:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".npz",dir=path.parent);os.close(fd)
    try:
        np.savez_compressed(tmp,**arrays)
        with open(tmp,"rb") as handle:os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def prefix_hash(token_ids:Sequence[int])->str:
    return hashlib.sha256(json.dumps(list(token_ids),separators=(",",":")).encode()).hexdigest()


def validate_prefix_record(row:Mapping[str,Any])->list[int]:
    forbidden=("response","future","continuation","reward","cost","mip","utility")
    if any(any(term in key.lower() for term in forbidden) for key in row if key!="prompt"):
        raise ValueError("State feature record contains a future/scorer field")
    ids=json.loads(row["prefix_token_ids"])
    if not isinstance(ids,list) or not ids or not all(isinstance(value,int) for value in ids):raise ValueError("Invalid causal prefix_token_ids")
    return ids


def hidden_and_logprobs(model:Any,token_ids:Sequence[int])->tuple[np.ndarray,np.ndarray]:
    ids=torch.tensor([list(token_ids)],dtype=torch.long,device="cuda");attention=torch.ones_like(ids);position=attention.cumsum(-1)-1
    with torch.inference_mode():
        output=model(input_ids=ids,attention_mask=attention,position_ids=position,use_cache=False,output_hidden_states=True,return_dict=True)
        logprobs=torch.log_softmax(output.logits[0,-1].float(),-1).cpu().numpy().astype(np.float32,copy=True)
        hidden=output.hidden_states[-1][0,-1].float().cpu().numpy().astype(np.float32,copy=True)
    del ids,attention,position,output
    if not np.isfinite(logprobs).all() or not np.isfinite(hidden).all():raise ValueError("Non-finite frozen features")
    return logprobs,hidden


def logit_geometry(base:np.ndarray,parm:np.ndarray,k:int)->np.ndarray:
    """Identity-free union top-k geometry sorted by local probability support."""
    if base.shape!=parm.shape or base.ndim!=1:raise ValueError("Expected matching vocabulary vectors")
    base_ids=np.argpartition(base,-k)[-k:];parm_ids=np.argpartition(parm,-k)[-k:];union=np.unique(np.concatenate((base_ids,parm_ids)))
    union=union[np.argsort(-np.maximum(base[union],parm[union]),kind="stable")]
    target=2*k;result=np.zeros((target,6),dtype=np.float32);result[:,3:5]=1.0
    base_order=np.empty(base.size,dtype=np.int32);base_order[np.argsort(-base,kind="stable")]=np.arange(base.size)
    parm_order=np.empty(parm.size,dtype=np.int32);parm_order[np.argsort(-parm,kind="stable")]=np.arange(parm.size)
    for index,token in enumerate(union[:target]):
        result[index]=(base[token],parm[token],parm[token]-base[token],min(base_order[token],k+1)/(k+1),min(parm_order[token],k+1)/(k+1),1.0)
    return result.reshape(-1)


def distribution_metrics(logprobs:np.ndarray)->tuple[int,float,float,float]:
    probs=np.exp(logprobs-logprobs.max());probs/=probs.sum();order=np.argsort(-probs,kind="stable");top=int(order[0]);entropy=float(-(probs*np.log(np.maximum(probs,1e-45))).sum());return top,float(probs[top]),float(probs[top]-probs[order[1]]),entropy


def action_feature_rows(state_id:str,base:np.ndarray,parm:np.ndarray)->list[dict[str,Any]]:
    reference=(1.0-1.0)*base+parm;ref_top,ref_prob,ref_margin,_=distribution_metrics(reference);ref_order=np.argsort(-reference,kind="stable");ref_log_margin=float(reference[ref_order[0]]-reference[ref_order[1]])
    rows=[]
    for weight in WEIGHTS:
        fused=(1-weight)*base+weight*parm;top,prob,margin,entropy=distribution_metrics(fused);order=np.argsort(-fused,kind="stable");rank=int(np.flatnonzero(order==ref_top)[0]);ref_token_prob=float(np.exp(fused[ref_top]-np.logaddexp.reduce(fused)));log_margin=float(fused[order[0]]-fused[order[1]])
        rows.append({"state_id":state_id,"weight":weight,"top1_token_id_audit_only":top,"top1_probability":prob,"entropy":entropy,"top1_top2_margin":margin,"top1_differs_from_w1":float(top!=ref_top),"w1_top_token_rank":rank,"w1_top_token_probability_change":ref_token_prob-ref_prob,"top1_logprob_margin_change_vs_w1":log_margin-ref_log_margin})
    return rows


def cache_valid(path:Path,state_id:str,prefix_sha:str)->bool:
    try:
        with np.load(path,allow_pickle=False) as data:
            return str(data["state_id"].item())==state_id and str(data["prefix_sha256"].item())==prefix_sha and all(np.isfinite(data[name]).all() for name in ("base_logprobs","parm_logprobs","base_hidden","parm_hidden"))
    except Exception:return False


def extract(*,resume:bool,min_free_mib:int=6000)->dict[str,Any]:
    import csv
    with (SOURCE/"state_features.csv").open(newline="",encoding="utf-8") as handle:states=list(csv.DictReader(handle))
    if len(states)!=250:raise ValueError("Expected exactly 250 frozen states")
    manifest=[]
    for row in states:
        ids=validate_prefix_record(row);manifest.append({"state_id":row["state_id"],"prompt_id":row["sample_id"],"alpha":[float(row["alpha_helpfulness"]),float(row["alpha_harmlessness"])],"prefix_token_count":len(ids),"prefix_sha256":prefix_hash(ids),"source":"raw prompt + already generated prefix only","cache_file":str(CACHE/f"{row['state_id']}.npz")})
    from PARM_TARO.adaptive_parm.token_headroom.io import atomic_jsonl,atomic_json,atomic_csv
    atomic_jsonl(OUT/"feature_manifest.jsonl",manifest);missing=[]
    for row,item in zip(states,manifest):
        path=Path(item["cache_file"])
        if not (resume and path.is_file() and cache_valid(path,item["state_id"],item["prefix_sha256"])):missing.append((row,item))
    adapter_before=sha256_file(ADAPTER/"adapter_model.safetensors");memory={"before":get_gpu_status()};model=base_view=tokenizer=None
    if missing:
        require_free_vram(min_free_mib);torch.cuda.reset_peak_memory_stats();model,base_view,tokenizer=load_tulu_pblora_4bit();memory["loaded"]={"nvidia_smi":get_gpu_status(),"allocated_mib":torch.cuda.memory_allocated()/1024**2,"reserved_mib":torch.cuda.memory_reserved()/1024**2}
        try:
            for index,(row,item) in enumerate(missing,1):
                ids=json.loads(row["prefix_token_ids"]);set_requested_alpha(model,item["alpha"]);base_lp,base_h=hidden_and_logprobs(base_view,ids);parm_lp,parm_h=hidden_and_logprobs(model,ids)
                atomic_npz(Path(item["cache_file"]),state_id=np.asarray(item["state_id"]),prefix_sha256=np.asarray(item["prefix_sha256"]),base_logprobs=base_lp,parm_logprobs=parm_lp,base_hidden=base_h,parm_hidden=parm_h)
                print(f"[{index}/{len(missing)}] cached {item['state_id']}",flush=True)
        finally:
            memory["peak_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
            if model is not None:del model,base_view,tokenizer
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();memory["after_release"]={"nvidia_smi":get_gpu_status(),"allocated_mib":torch.cuda.memory_allocated()/1024**2,"reserved_mib":torch.cuda.memory_reserved()/1024**2}
    scalar=[];geometry={k:[] for k in TOP_K_VALUES};base_hidden=[];parm_hidden=[];actions=[]
    for row,item in zip(states,manifest):
        with np.load(item["cache_file"],allow_pickle=False) as data:base=data["base_logprobs"].copy();parm=data["parm_logprobs"].copy();bh=data["base_hidden"].copy();ph=data["parm_hidden"].copy()
        scalar.append({"state_id":row["state_id"],"prompt_id":row["sample_id"],**{name:float(row[name]) for name in INFERENCE_SCALAR_COLUMNS}})
        for k in TOP_K_VALUES:geometry[k].append(logit_geometry(base,parm,k))
        base_hidden.append(bh);parm_hidden.append(ph);actions.extend(action_feature_rows(row["state_id"],base,parm))
    state_ids=np.asarray([row["state_id"] for row in states]);bh=np.stack(base_hidden);ph=np.stack(parm_hidden);delta=ph-bh;cosine=np.sum(bh*ph,axis=1)/(np.linalg.norm(bh,axis=1)*np.linalg.norm(ph,axis=1)+1e-12)
    atomic_csv(OUT/"scalar_features.csv",scalar);atomic_npz(OUT/"logit_geometry_features.npz",state_ids=state_ids,**{f"k{k}":np.stack(values) for k,values in geometry.items()});atomic_npz(OUT/"hidden_features.npz",state_ids=state_ids,h_base=bh,h_parm=ph,h_delta=delta,h_abs_delta=np.abs(delta),cosine=cosine);atomic_csv(OUT/"action_features.csv",actions)
    adapter_after=sha256_file(ADAPTER/"adapter_model.safetensors")
    if adapter_before!=adapter_after:raise RuntimeError("Frozen PBLoRA checkpoint changed during extraction")
    summary={"status":"PASS","states":len(states),"cached":len(states),"newly_extracted":len(missing),"prefix_only":True,"future_fields_used":False,"scorer_fields_used":False,"adapter_sha256":adapter_after,"pblora_frozen":True,"memory":memory};atomic_json(OUT/"feature_extraction_summary.json",summary);return summary
