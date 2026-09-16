"""Resumable full validation-only Feasibility-60 generation."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import *
OUTPUT=OUT/"08_generations.jsonl"

def parse_grid(value):
    result=[]
    for item in value.split(","):
        item=item.strip(); result.append("guide_only" if item=="guide_only" else float(item))
    return result
def informed_default_grid():
    values={0.,.1,.5,1.,2.,"guide_only"}; source=OUT/"04_headroom_summary.json"
    if source.is_file():
        try:
            summary=json.loads(source.read_text())["normalized"]
            candidates=[summary["global_fixed"]["lambda"]]+[item["lambda"] for item in summary.get("per_alpha",{}).values()]
            for value in candidates: values.add("guide_only" if value=="guide_only" else float(value))
        except (KeyError,TypeError,ValueError,json.JSONDecodeError): pass
    return sorted((x for x in values if x!="guide_only"),key=float)+["guide_only"]
def specs(grid,fusion):
    output=[]
    if fusion in ("all","normalized"):
        for value in grid:
            if value==0.: output.append(("base","base",0.,["normalized_lambda_0"])); continue
            if value=="guide_only": output.append(("normalized_guide_only","guide_only",None,[])); continue
            aliases=["author"] if value==1. else []; output.append((f"normalized_{value:g}","normalized",value,aliases))
    if fusion in ("all","current"):
        output.extend((("current_0.010731","current",.010731,[]),("current_1","current",1.,["paper_beta1"])))
    seen=set(); unique=[]
    for item in output:
        key=(item[1],item[2])
        if key not in seen: unique.append(item); seen.add(key)
    return unique
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); p.add_argument("--limit",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--max-new-tokens",type=int,default=64); p.add_argument("--fusion",choices=("all","normalized","current"),default="all"); p.add_argument("--lambda-grid",help="comma grid; default uses 04 optima plus 0,.1,.5,1,2,guide_only"); p.add_argument("--min-free-mib",type=int,default=5500); args=p.parse_args(); cases=load_manifest(); cases=cases[:args.limit] if args.limit else cases; grid=parse_grid(args.lambda_grid) if args.lambda_grid else informed_default_grid(); methods=specs(grid,args.fusion)
    if args.dry_run: print(json.dumps(dry_run_payload("08_feasibility60_generate",cases=len(cases),unique_generations=len(cases)*len(methods),methods=[{"method":x[0],"fusion":x[1],"lambda":x[2],"equivalent_method":x[3]} for x in methods],resume_key="case_id+fusion+lambda+seed"),indent=2)); return 0
    if OUTPUT.exists() and not args.resume: raise SystemExit("08_generations.jsonl exists; use --resume")
    existing=read_jsonl(OUTPUT); done={generation_key(r) for r in existing}; require_free_vram(args.min_free_mib); import torch
    torch.cuda.reset_peak_memory_stats(); memory={"before":cuda_memory_snapshot()}; model,base_view,tokenizer=load_tulu_pblora_4bit(); memory["loaded"]=cuda_memory_snapshot()
    try:
        for case in cases:
            for method,fusion,lam,aliases in methods:
                key=generation_key({"case_id":case["case_id"],"fusion":fusion,"lambda":lam,"seed":SEED})
                if key in done: continue
                generated=generate_fixed(model,base_view,tokenizer,case["prompt"],case["requested_alpha"],fusion=fusion,lambda_value=lam,max_new_tokens=args.max_new_tokens,seed=SEED); row={"case_id":case["case_id"],"sample_id":case["sample_id"],"prompt_id":case["sample_id"],"requested_alpha":case["requested_alpha"],"guide_alpha":case["requested_alpha"],"method":method,"fusion":fusion,"lambda":lam,"equivalent_method":aliases,"seed":SEED,"prompt":case["prompt"],**generated}; append_jsonl_atomic(OUTPUT,row); done.add(key); print(f"[{len(done)}] {case['case_id']} {method}",flush=True)
    finally:
        memory["peak"]=cuda_memory_snapshot(); del model,base_view,tokenizer; memory["after_release"]=release_model()
    summary={"status":"COMPLETE","records":len(read_jsonl(OUTPUT)),"requested_cases":len(cases),"methods":len(methods),"memory":memory,"resume_key":"case_id+fusion+lambda+seed","duplicate_equivalences_avoided":True}; atomic_json_dump(OUT/"08_generation_summary.json",summary); print("FEASIBILITY60 GENERATION COMPLETE"); return 0
if __name__=="__main__": raise SystemExit(main())
