"""Sequential Beaver reward/cost wiring sanity check."""
import argparse, json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import *
STATE=OUT/"07_scorer_state.json"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--limit",type=int,default=10); p.add_argument("--resume",action="store_true"); args=p.parse_args(); pre=scorer_path_preflight(); atomic_json_dump(OUT/"07_scorer_preflight.json",pre)
    if args.dry_run or args.preflight_only:
        payload=dry_run_payload("07_scorer_sanity",limit=args.limit,scorer_preflight=pre,sequential_load=True)
        print(json.dumps(payload,indent=2)); print("SCORER PREFLIGHT PASS" if pre["ready"] else "SCORER_BLOCKED_LOW_VRAM/ARTIFACT"); return 0
    if not pre["ready"]: raise SystemExit(f"Scorer prerequisites not ready: {pre}")
    generated=read_jsonl(OUT/"06_generations.jsonl")[:args.limit]
    if not generated: raise SystemExit("Run 06_generation_smoke.py first")
    score_records=[{"prompt":r["prompt"],"response":r["response"]} for r in generated]
    validation=load_manifest(); unique=[]
    for case in validation:
        if case["sample_id"] not in [x["sample_id"] for x in unique]: unique.append(case)
        if len(unique)==4: break
    sanity=[]
    for case in unique:
        data=case["validation_row"]
        for idx in (0,1): sanity.append({"prompt":case["prompt"],"response":data[f"response_{idx}"],"sample_id":case["sample_id"],"response_index":idx,"better":int(data["better_response_id"]),"safer":int(data["safer_response_id"])})
    combined=score_records+sanity
    if STATE.exists() and not args.resume: raise SystemExit(f"{STATE} exists; use --resume")
    state=json.loads(STATE.read_text()) if args.resume and STATE.is_file() else {}
    if "reward" not in state:
        reward,memory_reward=score_with_one_beaver(REWARD,combined); state["reward"]=reward; state["reward_memory"]=memory_reward; atomic_json_dump(STATE,state)
    else: reward=state["reward"]; memory_reward=state.get("reward_memory",{})
    if "cost" not in state:
        require_free_vram(5500); cost,memory_cost=score_with_one_beaver(COST,combined); state["cost"]=cost; state["cost_memory"]=memory_cost; atomic_json_dump(STATE,state)
    else: cost=state["cost"]; memory_cost=state.get("cost_memory",{})
    if len(reward)!=len(combined) or len(cost)!=len(combined): raise RuntimeError("07 scorer state does not match current --limit/input; move it aside")
    rows=[]
    for original,h,cost_score in zip(generated,reward[:len(generated)],cost[:len(generated)]): rows.append({"case_id":original["case_id"],"method":original["method"],"helpfulness_raw":h,"cost_raw":cost_score,"harmlessness_raw":-cost_score})
    atomic_csv_write(OUT/"07_scorer_scores.csv",rows,rows[0].keys()); offset=len(generated); reward_correct=cost_correct=0
    for i,case in enumerate(unique):
        left=offset+2*i; data=case["validation_row"]; better=int(data["better_response_id"]); safer=int(data["safer_response_id"]); reward_correct+=int(reward[left+better]>reward[left+1-better]); cost_correct+=int(cost[left+safer]<cost[left+1-safer])
    finite=all(math.isfinite(x) for x in reward+cost); summary={"status":"PASS" if finite and statistics.pstdev(reward[:len(generated)])>0 and statistics.pstdev(cost[:len(generated)])>0 else "FAIL","records":len(rows),"finite":finite,"reward_std":statistics.pstdev(reward[:len(generated)]),"cost_std":statistics.pstdev(cost[:len(generated)]),"labelled_reward_direction_accuracy":reward_correct/len(unique),"labelled_cost_direction_accuracy":cost_correct/len(unique),"reward_memory":memory_reward,"cost_memory":memory_cost,"models_loaded_simultaneously":False,"safe_rlhf_source_path":pre["safe_rlhf_source_path"],"safe_rlhf_git_commit":pre["safe_rlhf_git_commit"],"safe_rlhf_tree_hash":pre["safe_rlhf_tree_hash"],"provenance_status":pre["provenance_status"],"reward_model_path":pre["reward_model_path"],"cost_model_path":pre["cost_model_path"],"score_semantics":pre["score_semantics"]}
    atomic_json_dump(OUT/"07_scorer_summary.json",summary); atomic_text(OUT/"07_scorer_sanity.md",f"# Scorer sanity\n\nStatus: **{summary['status']}**\n\nReward and cost were loaded sequentially, never simultaneously. Reward variance={summary['reward_std']:.6g}; cost variance={summary['cost_std']:.6g}.\n"); print(f"SCORER {summary['status']}"); return 0 if summary["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
