"""Full 60-case teacher-forced lambda diagnostic with resumable CPU cache."""
import argparse, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import *

GRID=(0.0,.001,.01,.05,.1,.25,.5,.75,1.0,1.5,2.0,4.0,8.0,16.0)
CACHE_DIR=OUT/"03_lambda60/cache"; CACHE=CACHE_DIR/"cached_cases.pt"

def metadata():
    tokenizer_files=[BASE/name for name in ("tokenizer.json","tokenizer.model","tokenizer_config.json","special_tokens_map.json") if (BASE/name).is_file()]
    revision=BASE/".tta_artifact_revision.json"
    return {"schema":1,"model_config_sha256":sha256_file(BASE/"config.json"),"model_revision":json.loads(revision.read_text()).get("revision") if revision.is_file() else "UNRECORDED","tokenizer_hashes":{p.name:sha256_file(p) for p in tokenizer_files},"adapter_sha256":sha256_file(ADAPTER/"adapter_model.safetensors"),"manifest_sha256":sha256_file(MANIFEST),"validation_sha256":sha256_file(VALIDATION),"fusion":{"a":"log_softmax(base_logits)","b":"log_softmax(PBLoRA_logits)"},"alpha_order":list(ALPHA_ORDER),"max_length":512,"max_continuation_tokens":128,"dtype":"float32_cpu_clone"}

def load_cache(torch):
    expected=metadata()
    if not CACHE.is_file(): return {"metadata":expected,"cases":[]}
    value=torch.load(CACHE,map_location="cpu",weights_only=False)
    if value.get("metadata")!=expected: raise RuntimeError("03 cache provenance mismatch; move it aside instead of mixing artifacts")
    return value

def build_cache(torch,cases,model,base_view,tokenizer):
    payload=load_cache(torch); complete={x["case_id"] for x in payload["cases"]}; base_lookup={}
    for cached in payload["cases"]:
        for response in cached["responses"]: base_lookup[(cached["sample_id"],response["response_index"])]=response["a"]
    for index,case in enumerate(cases,1):
        if case["case_id"] in complete: print(f"[{index}/60] {case['case_id']} resume skip",flush=True); continue
        set_requested_alpha(model,case["requested_alpha"]); pair=tokenize_response_pair(tokenizer,case); responses=[]
        for response_index,item in enumerate(pair):
            key=(case["sample_id"],response_index); a=base_lookup.get(key)
            if a is None: a=selected_logprobs(base_view,item); base_lookup[key]=a
            b=selected_logprobs(model,item)
            responses.append({"response_index":response_index,"a":a.detach().float().cpu().clone(),"b":b.detach().float().cpu().clone(),"gold":item.gold_token_ids.detach().long().cpu().clone(),"truncated":bool(item.truncated)})
            del b
        payload["cases"].append({"case_id":case["case_id"],"sample_id":case["sample_id"],"prompt":case["prompt"],"alpha":list(case["requested_alpha"]),"weights":response_weights(case),"responses":responses}); atomic_torch_save(CACHE,payload); complete.add(case["case_id"]); print(f"[{index}/60] cached {case['case_id']}",flush=True)
    if len(payload["cases"])!=60: raise RuntimeError(f"Expected 60 cached cases, got {len(payload['cases'])}")
    return payload

def case_metrics(torch,case,fusion,lam):
    scalar=torch.tensor(float(lam),requires_grad=True); values=[]
    for response in case["responses"]:
        a=response["a"]; b=response["b"]
        logp=current_fusion(a,b,scalar) if fusion=="current" else normalized_fusion(a,b,scalar)
        gold=response["gold"]; nll=continuation_mean_nll(logp,gold); p=logp.exp(); entropy=-(p*logp).sum(-1).mean(); maxp=p.max(-1).values.mean(); raw=a+scalar*b if fusion=="current" else (a+scalar*b)/(1+scalar); std=raw.std(-1,unbiased=False).mean(); values.append((nll,entropy,maxp,std))
    weights=case["weights"].float(); aggregate=[sum(weights[i]*values[i][j] for i in (0,1)) for j in range(4)]; gradient=torch.autograd.grad(aggregate[0],scalar)[0]
    return {"nll":float(aggregate[0]),"gradient":float(gradient),"entropy":float(aggregate[1]),"max_probability":float(aggregate[2]),"logit_std":float(aggregate[3])}

def guide_metrics(torch,case):
    vals=[]
    for response in case["responses"]:
        logp=response["b"]; p=logp.exp(); vals.append((continuation_mean_nll(logp,response["gold"]),-(p*logp).sum(-1).mean(),p.max(-1).values.mean(),logp.std(-1,unbiased=False).mean()))
    w=case["weights"].float(); return {"nll":float(sum(w[i]*vals[i][0] for i in (0,1))),"gradient":"","entropy":float(sum(w[i]*vals[i][1] for i in (0,1))),"max_probability":float(sum(w[i]*vals[i][2] for i in (0,1))),"logit_std":float(sum(w[i]*vals[i][3] for i in (0,1)))}

def mean(items,key): return sum(float(x[key]) for x in items)/len(items)
def best(rows,scope): return min(rows,key=lambda r:(r.get("mean_nll",r.get("nll")),999 if r["lambda"]=="guide_only" else float(r["lambda"])))

def analyze(torch,payload):
    rows=[]; parity={"author":0.,"paper":0.}
    for case in payload["cases"]:
        for fusion in ("current","normalized"):
            for lam in GRID: rows.append({"case_id":case["case_id"],"sample_id":case["sample_id"],"alpha":alpha_key(case["alpha"]),"fusion":fusion,"lambda":lam,**case_metrics(torch,case,fusion,lam)})
        rows.append({"case_id":case["case_id"],"sample_id":case["sample_id"],"alpha":alpha_key(case["alpha"]),"fusion":"normalized","lambda":"guide_only",**guide_metrics(torch,case)})
        for response in case["responses"]:
            a,b=response["a"],response["b"]; parity["author"]=max(parity["author"],float((normalized_fusion(a,b,1.)-author_fusion(a,b)).abs().max())); parity["paper"]=max(parity["paper"],float((current_fusion(a,b,1.)-paper_beta1_fusion(a,b)).abs().max()))
    atomic_csv_write(OUT/"03_per_case_lambda.csv",rows,("case_id","sample_id","alpha","fusion","lambda","nll","gradient","entropy","max_probability","logit_std"))
    gradient=[]; numeric=[r for r in rows if r["lambda"]!="guide_only"]
    for keys in (("fusion","lambda"),("fusion","lambda","alpha"),("fusion","lambda","sample_id")):
        groups=defaultdict(list)
        for row in numeric: groups[tuple(row[k] for k in keys)].append(row)
        for values,group in groups.items():
            grads=[r["gradient"] for r in group]; item={k:v for k,v in zip(keys,values)}; item.update({"alpha":item.get("alpha","ALL"),"sample_id":item.get("sample_id","ALL"),"n":len(group),"mean_gradient":statistics.mean(grads),"median_gradient":statistics.median(grads),"fraction_positive":sum(g>0 for g in grads)/len(grads)}); gradient.append(item)
    atomic_csv_write(OUT/"03_gradient_summary.csv",gradient,("fusion","lambda","alpha","sample_id","n","mean_gradient","median_gradient","fraction_positive"))
    scale=[]
    for fusion in ("current","normalized"):
        labels=list(GRID)+(["guide_only"] if fusion=="normalized" else [])
        for label in labels:
            group=[r for r in rows if r["fusion"]==fusion and r["lambda"]==label]; scale.append({"fusion":fusion,"lambda":label,"mean_nll":mean(group,"nll"),"mean_entropy":mean(group,"entropy"),"mean_max_probability":mean(group,"max_probability"),"mean_logit_std":mean(group,"logit_std")})
    atomic_csv_write(OUT/"03_fusion_scale.csv",scale,scale[0].keys())
    global_rows=[]; alpha_rows=[]; prompt_rows=[]; oracle={}
    for fusion in ("current","normalized"):
        labels=list(GRID)+(["guide_only"] if fusion=="normalized" else [])
        candidates=[]
        for label in labels:
            group=[r for r in rows if r["fusion"]==fusion and r["lambda"]==label]; candidates.append({"fusion":fusion,"lambda":label,"mean_nll":mean(group,"nll")})
        chosen=best(candidates,"global"); global_rows.extend({**x,"is_best":x is chosen} for x in candidates)
        for alpha in map(alpha_key,ALPHAS):
            opts=[]
            for label in labels:
                group=[r for r in rows if r["fusion"]==fusion and r["alpha"]==alpha and r["lambda"]==label]; opts.append({"fusion":fusion,"alpha":alpha,"lambda":label,"mean_nll":mean(group,"nll")})
            alpha_rows.append(best(opts,"alpha"))
        prompt_ids=sorted(set(r["sample_id"] for r in rows))
        for prompt in prompt_ids:
            opts=[]
            for label in labels:
                group=[r for r in rows if r["fusion"]==fusion and r["sample_id"]==prompt and r["lambda"]==label]; opts.append({"fusion":fusion,"sample_id":prompt,"lambda":label,"mean_nll":mean(group,"nll")})
            prompt_rows.append({**best(opts,"prompt"),"alpha":"ALL"})
            for alpha in map(alpha_key,ALPHAS):
                alpha_opts=[]
                for label in labels:
                    group=[r for r in rows if r["fusion"]==fusion and r["sample_id"]==prompt and r["alpha"]==alpha and r["lambda"]==label]; alpha_opts.append({"fusion":fusion,"sample_id":prompt,"alpha":alpha,"lambda":label,"mean_nll":mean(group,"nll")})
                prompt_rows.append(best(alpha_opts,"prompt_alpha"))
        case_best=[]
        for case_id in sorted(set(r["case_id"] for r in rows)):
            case_best.append(best([r for r in rows if r["fusion"]==fusion and r["case_id"]==case_id],"case"))
        oracle_nll=mean(case_best,"nll"); gain=chosen["mean_nll"]-oracle_nll; dist=Counter(str(r["lambda"]) for r in case_best)
        oracle[fusion]={"global":chosen,"oracle_nll":oracle_nll,"absolute_gain":gain,"relative_gain_pct":100*gain/chosen["mean_nll"],"lambda_distribution":dict(dist),"fraction_differs_global":sum(r["lambda"]!=chosen["lambda"] for r in case_best)/60,"num_cases":60,"num_unique_prompts":12,"alphas_per_prompt":5}
    atomic_csv_write(OUT/"03_global_optima.csv",global_rows,("fusion","lambda","mean_nll","is_best")); atomic_csv_write(OUT/"03_per_alpha_optima.csv",alpha_rows,("fusion","alpha","lambda","mean_nll")); atomic_csv_write(OUT/"03_per_prompt_optima.csv",prompt_rows,("fusion","sample_id","alpha","lambda","mean_nll")); atomic_json_dump(OUT/"03_oracle.json",oracle)
    summary={"status":"COMPLETE","num_cases":60,"num_unique_prompts":12,"alphas_per_prompt":5,"grid":list(GRID)+["guide_only(normalized only)"],"parity_max_abs_logprob_diff":parity,"oracle":oracle,"old_cache_reused":False,"old_cache_reason":"prior cache lacks all required model revision/tokenizer/checkpoint provenance and covers only 20 cases"}; atomic_json_dump(OUT/"03_summary.json",summary)
    atomic_text(OUT/"03_report.md",f"# Lambda60 diagnostic\n\n- 60 correlated cases = 12 unique validation prompts × 5 alpha.\n- Current best: {oracle['current']['global']['lambda']}; oracle gain {oracle['current']['relative_gain_pct']:.3f}%.\n- Normalized best: {oracle['normalized']['global']['lambda']}; oracle gain {oracle['normalized']['relative_gain_pct']:.3f}%.\n- Author parity: {parity['author']:.3g}; paper parity: {parity['paper']:.3g}.\n\nTeacher-forced NLL is diagnostic only. `guide_only` is computed directly and is not a current-fusion infinite-lambda surrogate.\n")
    return summary

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--min-free-mib",type=int,default=5500); args=parser.parse_args(); cases=load_manifest(); manifest=validate_manifest(cases)
    if args.dry_run:
        print(json.dumps(dry_run_payload("03_lambda60",grid=list(GRID)+["guide_only"],cache=str(CACHE),old_cache_reusable=False,old_cache_reason="insufficient exact provenance/coverage"),indent=2)); return 0
    import torch
    existing=load_cache(torch); expected={c["case_id"] for c in cases}; got={c["case_id"] for c in existing["cases"]}; complete=got==expected and len(got)==60
    memory={}; model_loaded=False
    if complete: payload=existing; print("Complete CPU cache found; skipping GPU model load")
    else:
        require_free_vram(args.min_free_mib)
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats(); memory["before"]=cuda_memory_snapshot(); model,base_view,tokenizer=load_tulu_pblora_4bit(); memory["loaded"]=cuda_memory_snapshot(); model_loaded=True
        try: payload=build_cache(torch,cases,model,base_view,tokenizer); memory["peak"]=cuda_memory_snapshot()
        finally: del model,base_view,tokenizer; memory["after_release"]=release_model()
    summary=analyze(torch,payload); summary["memory"]=memory; summary["model_loaded_this_run"]=model_loaded; atomic_json_dump(OUT/"03_summary.json",summary); print("LAMBDA60 DIAGNOSTIC COMPLETE"); return 0
if __name__=="__main__": raise SystemExit(main())
