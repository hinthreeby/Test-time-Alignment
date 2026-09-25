"""Resume-safe greedy end-to-end generation for four frozen policies."""
from __future__ import annotations
import gc,hashlib,time
from typing import Any,Sequence
import torch
from PARM_TARO.adaptive_parm.fusion import adaptive_parm_logits
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_json,atomic_jsonl,read_jsonl,response_key,sha256_file
from PARM_TARO.recovery.low_vram_suite.common import get_gpu_status,load_tulu_pblora_4bit,require_free_vram,set_requested_alpha
from .config import ADAPTER,GENERATION_STATE,MAX_NEW_TOKENS,METHODS,OUT,SEED

def _next(model:Any,inputs:torch.Tensor,attention:torch.Tensor,position:torch.Tensor,past:Any)->tuple[torch.Tensor,Any]:
    with torch.inference_mode():
        output=model(input_ids=inputs,attention_mask=attention,position_ids=position,past_key_values=past,use_cache=True,return_dict=True);logp=torch.log_softmax(output.logits[:,-1].float(),-1);new_past=output.past_key_values
    del output
    return logp,new_past

def disagreement_weight_and_token(base:Any,guide:Any)->tuple[float,Any]:
    """Apply the frozen hard gate; returned token is necessarily Base top-1."""
    disagreement=bool((base.argmax(-1)!=guide.argmax(-1)).item());weight=0. if disagreement else 1.
    return weight,(base if disagreement else guide).argmax(-1,keepdim=True)

def generate_method(model:Any,base_view:Any,tokenizer:Any,prompt:str,alpha:Sequence[float],method:str,max_new_tokens:int=MAX_NEW_TOKENS)->dict[str,Any]:
    if method not in METHODS:raise ValueError(method)
    set_requested_alpha(model,alpha);encoded=tokenizer(prompt,return_tensors="pt",add_special_tokens=False);initial=encoded["input_ids"].cuda();attention=encoded.get("attention_mask",torch.ones_like(initial)).cuda();position=attention.cumsum(-1)-1
    base_input=guide_input=initial;base_past=guide_past=None;tokens=[];weights=[];disagreements=0;torch.cuda.synchronize();started=time.perf_counter()
    for _ in range(max_new_tokens):
        base=guide=None
        if method in ("base","midpoint","disagreement_gate"):base,base_past=_next(base_view,base_input,attention,position,base_past)
        if method in ("parm_fixed","midpoint","disagreement_gate"):guide,guide_past=_next(model,guide_input,attention,position,guide_past)
        if method=="base":weight=0.;token=base.argmax(-1,keepdim=True)
        elif method=="parm_fixed":weight=1.;token=guide.argmax(-1,keepdim=True)
        elif method=="midpoint":weight=.5;token=adaptive_parm_logits(base,guide,weight).argmax(-1,keepdim=True)
        else:
            disagreement=bool((base.argmax(-1)!=guide.argmax(-1)).item());disagreements+=int(disagreement);weight,token=disagreement_weight_and_token(base,guide)
        token_id=int(token.item());tokens.append(token_id);weights.append(weight)
        if token_id==tokenizer.eos_token_id:break
        attention=torch.cat((attention,torch.ones((1,1),dtype=attention.dtype,device=attention.device)),-1);position=torch.full((1,1),attention.shape[1]-1,dtype=torch.long,device=attention.device);base_input=guide_input=token
        del base,guide
    torch.cuda.synchronize();elapsed=time.perf_counter()-started;text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False);count=len(tokens);interventions=sum(value==0. for value in weights) if method=="disagreement_gate" else 0
    return {"response":text,"selected_token_ids":tokens,"generation_length":count,"latency_seconds":elapsed,"latency_per_token":elapsed/max(1,count),"disagreement_states":disagreements if method=="disagreement_gate" else None,"intervention_count":interventions,"intervention_rate":interventions/max(1,count),"top1_agreement_rate":1-disagreements/max(1,count) if method=="disagreement_gate" else None,"weight_history":weights,"weight_counts":{"0.0":sum(x==0. for x in weights),"0.5":sum(x==.5 for x in weights),"1.0":sum(x==1. for x in weights)},"weight_sequence_sha256":hashlib.sha256(str(weights).encode()).hexdigest(),"eos":bool(tokens and tokens[-1]==tokenizer.eos_token_id)}

def combine(cases:list[dict[str,Any]])->list[dict[str,Any]]:
    rows=[]
    for case in cases:
        for method in METHODS:
            path=GENERATION_STATE/f"{case['case_id']}__{method}.json"
            if not path.is_file():raise RuntimeError(f"Missing generation record: {path}")
            rows.append(__import__('json').loads(path.read_text(encoding='utf-8')))
    atomic_jsonl(OUT/"generations.jsonl",rows);return rows

def run(cases:list[dict[str,Any]],*,resume:bool,min_free_mib:int,max_new_tokens:int)->dict[str,Any]:
    GENERATION_STATE.mkdir(parents=True,exist_ok=True);jobs=[(case,method) for case in cases for method in METHODS]
    for case,method in jobs:
        path=GENERATION_STATE/f"{case['case_id']}__{method}.json"
        if path.is_file():
            row=__import__('json').loads(path.read_text(encoding='utf-8'))
            expected=(case['case_id'],case['prompt_sha256'],case['requested_alpha'],method,max_new_tokens,SEED);observed=(row.get('case_id'),row.get('prompt_sha256'),row.get('requested_alpha'),row.get('method'),row.get('max_new_tokens'),row.get('seed'))
            if observed!=expected:raise RuntimeError(f"Existing generation does not match frozen config: {path}")
    missing=[(case,method) for case,method in jobs if not (resume and (GENERATION_STATE/f"{case['case_id']}__{method}.json").is_file())]
    if any(GENERATION_STATE.glob('*.json')) and not resume:raise RuntimeError("Generation state exists; use --resume")
    before=sha256_file(ADAPTER/"adapter_model.safetensors");memory={"before":get_gpu_status()};model=base_view=tokenizer=None
    if missing:
        require_free_vram(min_free_mib);torch.cuda.reset_peak_memory_stats();model,base_view,tokenizer=load_tulu_pblora_4bit();memory["loaded"]=get_gpu_status()
        try:
            for index,(case,method) in enumerate(missing,1):
                generated=generate_method(model,base_view,tokenizer,case["prompt"],case["requested_alpha"],method,max_new_tokens)
                row={**case,"method":method,"seed":SEED,"max_new_tokens":max_new_tokens,"guide_alpha":case["requested_alpha"],**generated};row["response_key"]=response_key(row["prompt"],row["response"]);atomic_json(GENERATION_STATE/f"{case['case_id']}__{method}.json",row);print(f"[{index}/{len(missing)}] {case['case_id']} {method}",flush=True)
        finally:
            memory["peak_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
            if model is not None:del model,base_view,tokenizer
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();memory["after_release"]=get_gpu_status()
    after=sha256_file(ADAPTER/"adapter_model.safetensors")
    if before!=after:raise RuntimeError("Frozen PBLoRA checkpoint changed")
    rows=combine(cases);summary={"status":"COMPLETE","cases":len(cases),"methods":list(METHODS),"generations":len(rows),"unique_responses":len({r['response_key'] for r in rows}),"new_generations":len(missing),"max_new_tokens":max_new_tokens,"greedy":True,"canonical_fusion":"(1-w)*a+w*b","adapter_sha256":after,"memory":memory};atomic_json(OUT/"generation_summary.json",summary);return summary
