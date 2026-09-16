"""Compare teacher-forced and short greedy-prefix lambda-gradient signals."""
import argparse, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import PARM_TARO.recovery.low_vram_suite.common as c
from PARM_TARO.recovery.low_vram_suite.common import *

GRID=(.01,.1,.5,1.,2.); CACHE=OUT/"03_lambda60/cache/cached_cases.pt"; STATE_DIR=OUT/"05_exposure_state"
def token_gradient(torch,a,b,gold,lam,fusion="normalized"):
    scalar=torch.tensor(lam,requires_grad=True); logp=normalized_fusion(a,b,scalar) if fusion=="normalized" else current_fusion(a,b,scalar); loss=-logp[int(gold)]; grad=torch.autograd.grad(loss,scalar)[0]; p=logp.exp(); js=.5*((a.exp()*(a-torch.logaddexp(a,b)-math.log(.5))).sum()+(b.exp()*(b-torch.logaddexp(a,b)-math.log(.5))).sum()); return float(loss),float(grad),float(js)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); p.add_argument("--min-free-mib",type=int,default=5500); p.add_argument("--max-new-tokens",type=int,default=32); p.add_argument("--resume",action="store_true"); args=p.parse_args(); cases=select_prompt_subset(load_manifest(),2)
    if args.dry_run: print(json.dumps(dry_run_payload("05_exposure_shift",num_cases=len(cases),cache_required=str(CACHE),cache_exists=CACHE.is_file(),lambdas=GRID,max_new_tokens=args.max_new_tokens),indent=2)); return 0
    if not CACHE.is_file(): raise SystemExit("Run 03_lambda60_diagnostic.py first; teacher-forced cache is required")
    import torch
    cached=torch.load(CACHE,map_location="cpu",weights_only=False); by_case={x["case_id"]:x for x in cached["cases"]}; rows=[]
    for case in cases:
        item=by_case[case["case_id"]]
        for response in item["responses"]:
            weight=float(item["weights"][response["response_index"]])
            for pos,(a,b,gold) in enumerate(zip(response["a"],response["b"],response["gold"])):
                for lam in GRID:
                    loss,grad,js=token_gradient(torch,a,b,gold,lam); rows.append({"case_id":case["case_id"],"sample_id":case["sample_id"],"alpha":alpha_key(case["requested_alpha"]),"prefix_type":"teacher_forced","response_index":response["response_index"],"position":pos,"lambda":lam,"loss":loss,"gradient":grad,"objective_weight":weight,"weighted_gradient":weight*grad,"base_guide_js":js})
    existing_files=list(STATE_DIR.glob("*.json")) if STATE_DIR.is_dir() else []
    if existing_files and not args.resume: raise SystemExit(f"{STATE_DIR} contains progress; use --resume")
    done={path.stem for path in existing_files}; missing=[case for case in cases if case["case_id"] not in done]; memory={}; model=base_view=tokenizer=None
    if missing:
      require_free_vram(args.min_free_mib); torch.cuda.reset_peak_memory_stats(); memory={"before":cuda_memory_snapshot()}; model,base_view,tokenizer=load_tulu_pblora_4bit(); memory["loaded"]=cuda_memory_snapshot()
    try:
        for index,case in enumerate(missing,1):
            case_rows=[]
            set_requested_alpha(model,case["requested_alpha"]); encoded=tokenizer(case["prompt"],return_tensors="pt",add_special_tokens=False); ids=encoded["input_ids"].cuda(); attention=encoded.get("attention_mask",torch.ones_like(ids)).cuda(); position=attention.cumsum(-1)-1; base_input=guide_input=ids; base_past=guide_past=None
            for step in range(args.max_new_tokens):
                a,base_past=c._next_logprobs(base_view,base_input,attention,position,base_past); b,guide_past=c._next_logprobs(model,guide_input,attention,position,guide_past); selected=normalized_fusion(a,b,1.).argmax(-1,keepdim=True); token=int(selected.item()); ac=a[0].detach().float().cpu().clone(); bc=b[0].detach().float().cpu().clone()
                for lam in GRID:
                    loss,grad,js=token_gradient(torch,ac,bc,token,lam); case_rows.append({"case_id":case["case_id"],"sample_id":case["sample_id"],"alpha":alpha_key(case["requested_alpha"]),"prefix_type":"autoregressive_selected_token_proxy","response_index":"","position":step,"lambda":lam,"loss":loss,"gradient":grad,"objective_weight":1.,"weighted_gradient":grad,"base_guide_js":js})
                del a,b,ac,bc
                if token==tokenizer.eos_token_id: break
                base_input=guide_input=selected; attention=torch.cat((attention,torch.ones((1,1),dtype=attention.dtype,device=attention.device)),-1); position=torch.full((1,1),attention.shape[1]-1,dtype=torch.long,device=attention.device)
            del ids,attention,position,base_input,guide_input,base_past,guide_past
            atomic_json_dump(STATE_DIR/f"{case['case_id']}.json",case_rows); print(f"[{index}/{len(missing)}] {case['case_id']}",flush=True)
    finally:
        if missing: memory["peak"]=cuda_memory_snapshot(); del model,base_view,tokenizer; memory["after_release"]=release_model()
    for case in cases:
        path=STATE_DIR/f"{case['case_id']}.json"
        if not path.is_file(): raise RuntimeError(f"Missing completed exposure case: {path}")
        rows.extend(json.loads(path.read_text()))
    atomic_csv_write(OUT/"05_exposure_shift_tokens.csv",rows,rows[0].keys()); summary_rows=[]
    for lam in GRID:
        for kind in ("teacher_forced","autoregressive_selected_token_proxy"):
            group=[r for r in rows if r["lambda"]==lam and r["prefix_type"]==kind]; case_gradients=[]
            for case in cases:
                selected=[r for r in group if r["case_id"]==case["case_id"]]
                if kind=="teacher_forced":
                    # Exact dual-response objective: mean within each response,
                    # then alpha-derived response weighting (already in rows).
                    value=sum(statistics.mean(r["weighted_gradient"] for r in selected if r["response_index"]==idx) for idx in (0,1))
                else: value=statistics.mean(r["gradient"] for r in selected)
                case_gradients.append(value)
            summary_rows.append({"lambda":lam,"prefix_type":kind,"mean_gradient":statistics.mean(case_gradients),"fraction_positive":sum(g>0 for g in case_gradients)/len(case_gradients),"mean_base_guide_js":statistics.mean(r["base_guide_js"] for r in group)})
    agreements=[]
    for case in cases:
        for lam in GRID:
            t=[r for r in rows if r["case_id"]==case["case_id"] and r["lambda"]==lam and r["prefix_type"]=="teacher_forced"]; a=[r["gradient"] for r in rows if r["case_id"]==case["case_id"] and r["lambda"]==lam and r["prefix_type"].startswith("autoregressive")]; teacher=sum(statistics.mean(r["weighted_gradient"] for r in t if r["response_index"]==idx) for idx in (0,1)); agreements.append((teacher>0)==(statistics.mean(a)>0))
    summary={"status":"COMPLETE","num_cases":len(cases),"num_unique_prompts":2,"max_new_tokens":args.max_new_tokens,"summary":summary_rows,"gradient_sign_agreement":sum(agreements)/len(agreements),"memory":memory,"autoregressive_loss_label":"selected-token diagnostic proxy, not ground truth"}; atomic_json_dump(OUT/"05_exposure_shift_summary.json",summary); atomic_text(OUT/"05_exposure_shift.md",f"# Exposure shift\n\n- Cases: {len(cases)} (2 unique prompts × 5 alpha).\n- Gradient-sign agreement: {summary['gradient_sign_agreement']:.3f}.\n- Autoregressive loss uses the selected trajectory token and is not ground-truth loss.\n- GPU release: {memory['after_release']['status']}.\n"); print("EXPOSURE SHIFT COMPLETE"); return 0
if __name__=="__main__": raise SystemExit(main())
