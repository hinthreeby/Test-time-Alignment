"""Sequential Beaver scoring with response-level resume caches."""
from __future__ import annotations
import gc,json,math
from pathlib import Path
from typing import Any
import torch
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv,atomic_json,read_jsonl
from PARM_TARO.adaptive_parm.token_headroom.scoring import _load
from PARM_TARO.recovery.low_vram_suite.common import get_gpu_status,require_free_vram,resolve_safe_rlhf_source
from .config import COST,OUT,REWARD,SCORE_STATE

def run(kind:str,*,resume:bool,min_free_mib:int)->dict[str,Any]:
    if kind not in ("reward","cost"):raise ValueError(kind)
    resolution=resolve_safe_rlhf_source()
    if not resolution.get("ready"):raise RuntimeError(f"Safe-RLHF runtime unavailable: {resolution}")
    generations=read_jsonl(OUT/"generations.jsonl");unique={row["response_key"]:{"prompt":row["prompt"],"response":row["response"]} for row in generations};state=SCORE_STATE/kind;state.mkdir(parents=True,exist_ok=True)
    existing={path.stem for path in state.glob("*.json")};missing=[key for key in sorted(unique) if key not in existing]
    if existing and not resume:raise RuntimeError(f"{kind} progress exists; use --resume")
    path=REWARD if kind=="reward" else COST;memory={"before":get_gpu_status()};model=tokenizer=None
    if missing:
        require_free_vram(min_free_mib);model,tokenizer=_load(path,resolution);memory["loaded"]=get_gpu_status()
        try:
            with torch.inference_mode():
                for index,key in enumerate(missing,1):
                    item=unique[key];text=f"BEGINNING OF CONVERSATION: USER: {item['prompt']} ASSISTANT:{item['response']}";encoded=tokenizer(text,return_tensors="pt",truncation=True,max_length=512);encoded={name:value.cuda() for name,value in encoded.items()};output=model(**encoded);end=output["end_scores"] if isinstance(output,dict) else output.end_scores;value=float(end.detach().float().reshape(-1)[0].cpu())
                    if not math.isfinite(value):raise FloatingPointError(f"Non-finite {kind} score")
                    atomic_json(state/f"{key}.json",{"response_key":key,"score":value});del encoded,output,end;print(f"[{index}/{len(missing)}] {kind} {key[:12]}",flush=True)
        finally:
            if model is not None:del model,tokenizer
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();memory["after_release"]=get_gpu_status()
    scores={key:float(json.loads((state/f"{key}.json").read_text())["score"]) for key in unique};rows=[{"case_id":row["case_id"],"method":row["method"],"response_key":row["response_key"],f"{kind}_score":scores[row["response_key"]]} for row in generations];atomic_csv(OUT/f"{kind}_scores.csv",rows)
    summary={"status":"PASS","kind":kind,"records":len(rows),"unique_responses":len(unique),"finite":all(math.isfinite(v) for v in scores.values()),"model_path":str(path),"new_scores":len(missing),"memory":memory,"safe_rlhf_source":resolution.get("path"),"safe_rlhf_commit":resolution.get("git_commit"),"score_semantics":"higher reward is more helpful; higher cost is more harmful"};atomic_json(OUT/f"{kind}_scoring_summary.json",summary);return summary

