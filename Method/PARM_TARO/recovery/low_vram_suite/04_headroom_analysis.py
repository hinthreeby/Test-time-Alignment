"""Unique-prompt-aware CPU analysis of Lambda60 NLL headroom."""
import argparse, csv, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import OUT, atomic_csv_write, atomic_json_dump, atomic_text, dry_run_payload

SOURCE=OUT/"03_per_case_lambda.csv"
def read_rows():
    with SOURCE.open() as f:
        rows=list(csv.DictReader(f))
    for r in rows: r["nll"]=float(r["nll"]); r["lambda_value"]=float("inf") if r["lambda"]=="guide_only" else float(r["lambda"])
    return rows
def choose(rows): return min(rows,key=lambda r:(r["mean_nll"],r["lambda_value"]))
def bootstrap_prompt_ci(prompt_losses,reps=2000):
    rng=random.Random(2026); ids=sorted(prompt_losses); values=[]
    for _ in range(reps):
        sample=[rng.choice(ids) for __ in ids]; values.append(sum(prompt_losses[x] for x in sample)/len(sample))
    values.sort(); return [values[int(.025*(reps-1))],values[int(.975*(reps-1))]]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    if args.dry_run: print(json.dumps(dry_run_payload("04_headroom",source=str(SOURCE),source_exists=SOURCE.is_file()),indent=2)); return 0
    if not SOURCE.is_file(): raise SystemExit("Run 03_lambda60_diagnostic.py first")
    rows=read_rows(); output={}; by_prompt_rows=[]
    for fusion in ("current","normalized"):
        frows=[r for r in rows if r["fusion"]==fusion]; labels=sorted(set(r["lambda"] for r in frows),key=lambda x:float("inf") if x=="guide_only" else float(x)); prompts=sorted(set(r["sample_id"] for r in frows)); alphas=sorted(set(r["alpha"] for r in frows))
        globals=[]
        for label in labels: group=[r for r in frows if r["lambda"]==label]; globals.append({"lambda":label,"lambda_value":float("inf") if label=="guide_only" else float(label),"mean_nll":statistics.mean(r["nll"] for r in group)})
        global_best=choose(globals)
        per_alpha={}; per_prompt={}; per_case={}; prompt_multi={}
        for alpha in alphas:
            opts=[]
            for label in labels: group=[r for r in frows if r["alpha"]==alpha and r["lambda"]==label]; opts.append({"lambda":label,"lambda_value":float("inf") if label=="guide_only" else float(label),"mean_nll":statistics.mean(r["nll"] for r in group)})
            per_alpha[alpha]=choose(opts)
        for prompt in prompts:
            opts=[]
            for label in labels: group=[r for r in frows if r["sample_id"]==prompt and r["lambda"]==label]; opts.append({"lambda":label,"lambda_value":float("inf") if label=="guide_only" else float(label),"mean_nll":statistics.mean(r["nll"] for r in group)})
            per_prompt[prompt]=choose(opts)
            alpha_choices=[]
            for alpha in alphas:
                group=[r for r in frows if r["sample_id"]==prompt and r["alpha"]==alpha]; alpha_choices.append(min(group,key=lambda r:(r["nll"],r["lambda_value"]))["lambda"])
            prompt_multi[prompt]=len(set(alpha_choices))>1
            by_prompt_rows.append({"fusion":fusion,"sample_id":prompt,"best_prompt_lambda":per_prompt[prompt]["lambda"],"num_distinct_alpha_optima":len(set(alpha_choices)),"optimal_changes_with_alpha":prompt_multi[prompt]})
        for case_id in sorted(set(r["case_id"] for r in frows)):
            per_case[case_id]=min([r for r in frows if r["case_id"]==case_id],key=lambda r:(r["nll"],r["lambda_value"]))
        global_nll=global_best["mean_nll"]; per_alpha_nll=statistics.mean(per_alpha[r["alpha"]]["mean_nll"] for r in frows); per_prompt_nll=statistics.mean(per_prompt[p]["mean_nll"] for p in prompts); oracle_nll=statistics.mean(r["nll"] for r in per_case.values())
        prompt_gains={prompt:statistics.mean(r["nll"] for r in frows if r["sample_id"]==prompt and r["lambda"]==global_best["lambda"])-statistics.mean(per_case[r["case_id"]]["nll"] for r in frows if r["sample_id"]==prompt and r["lambda"]==global_best["lambda"]) for prompt in prompts}
        dist=Counter(r["lambda"] for r in per_case.values()); zero=dist.get("0.0",0)/60; upper=(dist.get("16.0",0)+dist.get("guide_only",0))/60
        gain_pct=100*(global_nll-oracle_nll)/global_nll
        classification="HEADROOM CLEAR" if gain_pct>=1 and sum(prompt_multi.values())>=3 else ("HEADROOM ABSENT" if gain_pct<.1 and not any(prompt_multi.values()) else ("HEADROOM WEAK" if gain_pct<1 else "HEADROOM AMBIGUOUS"))
        output[fusion]={"classification":classification,"num_cases":60,"num_unique_prompts":12,"global_fixed":{"lambda":global_best["lambda"],"nll":global_nll},"per_alpha_nll":per_alpha_nll,"per_prompt_nll":per_prompt_nll,"per_case_oracle_nll":oracle_nll,"oracle_absolute_gain":global_nll-oracle_nll,"oracle_relative_gain_pct":gain_pct,"oracle_gain_bootstrap_95pct_ci_over_prompts":bootstrap_prompt_ci(prompt_gains),"fraction_cases_oracle_differs_global":sum(r["lambda"]!=global_best["lambda"] for r in per_case.values())/60,"fraction_prompts_multiple_alpha_optima":sum(prompt_multi.values())/12,"selected_distribution":dict(dist),"boundary_flags":{"OPTIMUM_AT_ZERO_BOUNDARY":zero>=.25,"OPTIMUM_AT_UPPER_BOUNDARY":upper>=.25,"zero_fraction":zero,"upper_fraction":upper},"per_alpha":per_alpha}
    atomic_json_dump(OUT/"04_headroom_summary.json",output); atomic_csv_write(OUT/"04_headroom_by_prompt.csv",by_prompt_rows,by_prompt_rows[0].keys()); atomic_text(OUT/"04_headroom_report.md",f"# Headroom analysis\n\nPrompt is the bootstrap/independence unit: 12 prompts × 5 correlated alpha cases.\n\n- Current: **{output['current']['classification']}**, oracle gain {output['current']['oracle_relative_gain_pct']:.3f}%.\n- Normalized: **{output['normalized']['classification']}**, oracle gain {output['normalized']['oracle_relative_gain_pct']:.3f}%.\n\nTeacher-forced NLL remains diagnostic and cannot establish final router GO/NO-GO.\n"); print(output["normalized"]["classification"]); return 0
if __name__=="__main__": raise SystemExit(main())
