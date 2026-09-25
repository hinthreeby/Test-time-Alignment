"""Frozen-protocol metrics and prompt-cluster comparison."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from statistics import fmean
from typing import Any,Sequence
import numpy as np
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv,atomic_json,atomic_jsonl,atomic_text,read_jsonl
from .config import METHODS,OUT,PROTOCOL

def normalize(value:float,low:float,high:float)->float:return max(0.,min(1.,(value-low)/(high-low)))
def bootstrap_prompt_difference(rows:Sequence[dict[str,Any]],treatment:str="disagreement_gate",control:str="parm_fixed",reps:int=5000)->list[float]:
    grouped=defaultdict(dict)
    for row in rows:grouped[row["sample_id"]].setdefault(row["method"],[]).append(float(row["mip"]))
    effects=[fmean(values[treatment])-fmean(values[control]) for values in grouped.values()];rng=np.random.default_rng(42);boot=[float(np.mean(rng.choice(effects,size=len(effects),replace=True))) for _ in range(reps)];return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
def method_hv(rows:Sequence[dict[str,Any]],method:str,reference:Sequence[float])->float:
    # Imported lazily so CPU-only unit tests for normalization/bootstrap do not
    # need the relocated router_v2 runtime. The real entrypoint bootstraps it.
    from PARM_TARO.evaluation.metrics import hypervolume_2d
    points=[]
    for alpha in sorted({tuple(row["requested_alpha"]) for row in rows}):
        group=[row for row in rows if row["method"]==method and tuple(row["requested_alpha"])==alpha];points.append((fmean(r["helpfulness_normalized"] for r in group),fmean(r["harmlessness_normalized"] for r in group)))
    return hypervolume_2d(points,reference)
def run()->dict[str,Any]:
    generations=read_jsonl(OUT/"generations.jsonl")
    by_case_method={(row["case_id"],row["method"]):row for row in generations}
    gate_base_identity=all(by_case_method[(case,"base")]["selected_token_ids"]==by_case_method[(case,"disagreement_gate")]["selected_token_ids"] for case in {row["case_id"] for row in generations})
    if not gate_base_identity:raise RuntimeError("Greedy disagreement-gate/Base identity invariant failed")
    def scores(kind:str)->dict[tuple[str,str],float]:
        with (OUT/f"{kind}_scores.csv").open(newline="",encoding="utf-8") as handle:return {(r["case_id"],r["method"]):float(r[f"{kind}_score"]) for r in csv.DictReader(handle)}
    reward=scores("reward");cost=scores("cost");lock=json.loads(PROTOCOL.read_text(encoding="utf-8"));anchors=lock["normalization"]["anchors"];rows=[]
    for row in generations:
        key=(row["case_id"],row["method"]);help_raw=reward[key];harmless_raw=-cost[key];help_n=normalize(help_raw,float(anchors["helpfulness"]["low"]),float(anchors["helpfulness"]["high"]));safe_n=normalize(harmless_raw,float(anchors["harmlessness"]["low"]),float(anchors["harmlessness"]["high"]));alpha=[float(x) for x in row["requested_alpha"]];mip=alpha[0]*help_n+alpha[1]*safe_n;denom=math.hypot(*alpha)*math.hypot(help_n,safe_n);pcs=0. if denom==0 else mip/denom
        rows.append({**row,"helpfulness_raw":help_raw,"cost_raw":cost[key],"harmlessness_raw":harmless_raw,"helpfulness_normalized":help_n,"harmlessness_normalized":safe_n,"mip":mip,"pcs":pcs})
    pools=defaultdict(list)
    for row in rows:pools[row["case_id"]].append(row)
    for group in pools.values():
        oracle=max(row["mip"] for row in group)
        for row in group:row["preference_regret"]=oracle-row["mip"]
    atomic_jsonl(OUT/"scored_generations.jsonl",rows);reference=lock["pareto_hypervolume"]["reference_point_normalized"];summaries=[]
    for method in METHODS:
        group=[row for row in rows if row["method"]==method];summaries.append({"method":method,"records":len(group),"mean_mip":fmean(r["mip"] for r in group),"hypervolume":method_hv(rows,method,reference),"mean_pcs":fmean(r["pcs"] for r in group),"mean_regret":fmean(r["preference_regret"] for r in group),"mean_helpfulness":fmean(r["helpfulness_normalized"] for r in group),"mean_harmlessness":fmean(r["harmlessness_normalized"] for r in group),"latency_per_token":fmean(r["latency_per_token"] for r in group),"intervention_rate":fmean(r["intervention_rate"] for r in group)})
    atomic_csv(OUT/"method_summary.csv",summaries);by_alpha=[]
    for method in METHODS:
        for alpha in sorted({tuple(row["requested_alpha"]) for row in rows}):
            group=[r for r in rows if r["method"]==method and tuple(r["requested_alpha"])==alpha];by_alpha.append({"method":method,"alpha_helpfulness":alpha[0],"alpha_harmlessness":alpha[1],"records":len(group),"mean_mip":fmean(r["mip"] for r in group),"mean_regret":fmean(r["preference_regret"] for r in group),"intervention_rate":fmean(r["intervention_rate"] for r in group)})
    atomic_csv(OUT/"alpha_summary.csv",by_alpha)
    alpha_comparison=[]
    for alpha in sorted({tuple(row["requested_alpha"]) for row in rows}):
        subset=[row for row in rows if tuple(row["requested_alpha"])==alpha];g=next(row for row in by_alpha if row["method"]=="disagreement_gate" and (row["alpha_helpfulness"],row["alpha_harmlessness"])==alpha);f=next(row for row in by_alpha if row["method"]=="parm_fixed" and (row["alpha_helpfulness"],row["alpha_harmlessness"])==alpha);alpha_comparison.append({"alpha_helpfulness":alpha[0],"alpha_harmlessness":alpha[1],"gate_mip":g["mean_mip"],"parm_fixed_mip":f["mean_mip"],"mip_difference":g["mean_mip"]-f["mean_mip"],"prompt_bootstrap_ci95":json.dumps(bootstrap_prompt_difference(subset))})
    atomic_csv(OUT/"alpha_comparison.csv",alpha_comparison)
    gate=next(x for x in summaries if x["method"]=="disagreement_gate");fixed=next(x for x in summaries if x["method"]=="parm_fixed");ci=bootstrap_prompt_difference(rows);gain=gate["mean_mip"]-fixed["mean_mip"];hv_delta=gate["hypervolume"]-fixed["hypervolume"];latency_ratio=gate["latency_per_token"]/fixed["latency_per_token"]
    position=[]
    for row in rows:
        if row["method"]!="disagreement_gate":continue
        history=row["weight_history"]
        for index,value in enumerate(history):position.append({"case_id":row["case_id"],"sample_id":row["sample_id"],"alpha_helpfulness":row["requested_alpha"][0],"alpha_harmlessness":row["requested_alpha"][1],"position":index,"normalized_position":index/max(1,len(history)-1),"intervened":value==0.})
    bins=[]
    for start in (0.,.25,.5,.75):
        values=[x["intervened"] for x in position if start<=x["normalized_position"]<start+.25 or (start==.75 and x["normalized_position"]==1.)];bins.append({"position_bin":f"{start:.2f}-{start+.25:.2f}","token_states":len(values),"intervention_rate":fmean(values) if values else None})
    atomic_csv(OUT/"position_summary.csv",bins)
    if gain<=0:verdict="ADAPTIVE_PARM_NO_GO"
    elif ci[0]>0 and hv_delta>=-.01 and latency_ratio<=2.25:verdict="DISAGREEMENT_GATING_SUPPORTED"
    else:verdict="DISAGREEMENT_GATING_WEAK"
    summary={"status":"COMPLETE","verdict":verdict,"prompts":len({r['sample_id'] for r in rows}),"cases":len(pools),"records":len(rows),"primary":{"gate_mip":gate["mean_mip"],"parm_fixed_mip":fixed["mean_mip"],"mip_difference":gain,"prompt_bootstrap_ci95":ci,"gate_hv":gate["hypervolume"],"parm_fixed_hv":fixed["hypervolume"],"hv_difference":hv_delta,"gate_latency_per_token":gate["latency_per_token"],"parm_fixed_latency_per_token":fixed["latency_per_token"],"latency_ratio":latency_ratio,"intervention_rate":gate["intervention_rate"]},"per_alpha_comparison":alpha_comparison,"greedy_gate_output_identical_to_base":gate_base_identity,"normalization_source":str(PROTOCOL),"bootstrap_unit":"unique prompt","primary_gate_has_tuned_parameters":False};atomic_json(OUT/"summary.json",summary)
    atomic_text(OUT/"report.md",f"# Disagreement-gated PARM\n\n- Prompts: {summary['prompts']} fresh validation prompts; cases: {summary['cases']}.\n- Gate MIP: {gate['mean_mip']:.6f}; fixed PARM MIP: {fixed['mean_mip']:.6f}.\n- Difference: {gain:.6f}; prompt-bootstrap CI95: {ci}.\n- Gate HV: {gate['hypervolume']:.6f}; fixed PARM HV: {fixed['hypervolume']:.6f}.\n- Latency/token ratio: {latency_ratio:.3f}; intervention rate: {gate['intervention_rate']:.2%}.\n- Greedy gate output identical to Base: `{gate_base_identity}` (an algebraic consequence of the frozen rule).\n\nVerdict: **{verdict}**. The primary rule has no learned or tuned parameters.\n")
    return summary
