"""Ten-case evaluator-free fixed-lambda generation pipeline smoke test."""
import argparse, json, math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import *

METHODS=(("base","base",0.),("normalized_0.1","normalized",.1),("normalized_0.5","normalized",.5),("normalized_1","normalized",1.),("normalized_2","normalized",2.),("guide_only","guide_only",None),("current_0.010731","current",.010731),("current_1","current",1.))
OUTPUT=OUT/"06_generations.jsonl"
def repeat4(ids):
    grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    return 0. if not grams else 1-len(set(grams))/len(grams)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); p.add_argument("--min-free-mib",type=int,default=5500); p.add_argument("--max-new-tokens",type=int,default=64); p.add_argument("--resume",action="store_true"); args=p.parse_args(); cases=select_prompt_subset(load_manifest(),2)
    if args.dry_run: print(json.dumps(dry_run_payload("06_generation_smoke",num_cases=len(cases),methods=[x[0] for x in METHODS],planned_generations=len(cases)*len(METHODS),max_new_tokens=args.max_new_tokens,resume_key="case_id+fusion+lambda+seed"),indent=2)); return 0
    require_free_vram(args.min_free_mib); existing=read_jsonl(OUTPUT) if args.resume else []
    if OUTPUT.exists() and not args.resume: raise SystemExit(f"{OUTPUT} exists; pass --resume to preserve it")
    done={generation_key(r) for r in existing}; import torch
    torch.cuda.reset_peak_memory_stats(); memory={"before":cuda_memory_snapshot()}; model,base_view,tokenizer=load_tulu_pblora_4bit(); memory["loaded"]=cuda_memory_snapshot()
    try:
        count=0
        for case in cases:
            for method,fusion,lam in METHODS:
                key=generation_key({"case_id":case["case_id"],"fusion":fusion,"lambda":lam,"seed":SEED})
                if key in done: continue
                result=generate_fixed(model,base_view,tokenizer,case["prompt"],case["requested_alpha"],fusion=fusion,lambda_value=lam,max_new_tokens=args.max_new_tokens,seed=SEED); row={"case_id":case["case_id"],"sample_id":case["sample_id"],"prompt_id":case["sample_id"],"requested_alpha":case["requested_alpha"],"guide_alpha":case["requested_alpha"],"method":method,"fusion":fusion,"lambda":lam,"seed":SEED,"prompt":case["prompt"],**result}; append_jsonl_atomic(OUTPUT,row); done.add(key); count+=1; print(f"generated {count}: {case['case_id']} {method}",flush=True)
    finally:
        memory["peak"]=cuda_memory_snapshot(); del model,base_view,tokenizer; memory["after_release"]=release_model()
    rows=read_jsonl(OUTPUT); required=len(cases)*len(METHODS); valid= len(rows)==required and all(r["length"]>0 and math.isfinite(r["base_conditional_perplexity"]) for r in rows)
    summary={"status":"PASS" if valid else "FAIL","expected":required,"records":len(rows),"mean_length":sum(r["length"] for r in rows)/len(rows),"eos_rate":sum(r["eos"] for r in rows)/len(rows),"mean_repeated_4gram_rate":sum(repeat4(r["token_ids"]) for r in rows)/len(rows),"unique_response_rate":len(set(r["response"] for r in rows))/len(rows),"memory":memory,"alignment_claim":False}; atomic_json_dump(OUT/"06_generation_summary.json",summary); atomic_text(OUT/"06_generation_smoke.md",f"# Generation smoke\n\n**GENERATION PIPELINE {summary['status']}**\n\n- Records: {len(rows)}/{required}\n- Mean length: {summary['mean_length']:.2f}\n- EOS rate: {summary['eos_rate']:.3f}\n- GPU release: {memory['after_release']['status']}\n\nThis validates the pipeline, not alignment quality.\n"); print(f"GENERATION PIPELINE {summary['status']}"); return 0 if valid else 2
if __name__=="__main__": raise SystemExit(main())
