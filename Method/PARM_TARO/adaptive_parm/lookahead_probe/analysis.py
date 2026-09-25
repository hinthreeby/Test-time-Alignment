"""CPU-only direct-score ranking, policy, and compute analysis."""

from __future__ import annotations

import csv,json
from collections import Counter
from itertools import combinations
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv,atomic_json,atomic_text
from .config import HORIZONS,OUT,RICH,SCORE_FAMILIES,UTILITY_EPSILON,WEIGHTS


def informative_pair_accuracy(utility:Sequence[float],score:Sequence[float],epsilon:float=UTILITY_EPSILON)->tuple[int,float|None]:
    correct=[]
    for i,j in combinations(range(len(utility)),2):
        delta=float(utility[i]-utility[j])
        if abs(delta)<=epsilon:continue
        prediction=float(score[i]-score[j]);correct.append(0.5 if abs(prediction)<=1e-12 else float(np.sign(prediction)==np.sign(delta)))
    return len(correct),float(np.mean(correct)) if correct else None


def defined_spearman(utility:Sequence[float],score:Sequence[float])->float|None:
    if np.ptp(utility)<=UTILITY_EPSILON or np.ptp(score)<=1e-12:return None
    value=float(spearmanr(utility,score).statistic)
    return value if np.isfinite(value) else None


def select_weight(weights:Sequence[float],scores:Sequence[float])->int:
    maximum=max(scores);ties=[i for i,value in enumerate(scores) if abs(value-maximum)<=1e-12]
    for index in ties:
        if abs(weights[index]-1.0)<=1e-12:return index
    return ties[0]


def grouped(rows:list[dict[str,Any]],subset:str)->dict[str,list[dict[str,Any]]]:
    chosen=[row for row in rows if subset=="all" or row["is_actionable"]]
    result={}
    for row in chosen:result.setdefault(row["state_id"],[]).append(row)
    return result


def evaluate(rows:list[dict[str,Any]],family:str,horizon:int,subset:str)->tuple[dict[str,Any],list[dict[str,Any]]]:
    states=grouped([row for row in rows if row["horizon"]==horizon],subset);pairs=[];correct=[];spearman=[];top1=[];top2=[];policy=[];fixed=[];oracle=[];decisions=[]
    for state_id,items in states.items():
        items=sorted(items,key=lambda row:row["weight"]);weights=[row["weight"] for row in items];utility=[row["mip"] for row in items];scores=[row[f"{family}_score"] for row in items]
        count,accuracy=informative_pair_accuracy(utility,scores);pairs.append(count)
        if accuracy is not None:correct.append((count,accuracy))
        rho=defined_spearman(utility,scores)
        if rho is not None:spearman.append(rho)
        selected=select_weight(weights,scores);ranked=sorted(range(len(items)),key=lambda i:(scores[i],weights[i]==1.0),reverse=True);maximum=max(utility);oracle_set={i for i,value in enumerate(utility) if maximum-value<=UTILITY_EPSILON};ref=weights.index(1.0)
        selected_u=utility[selected];fixed_u=utility[ref];policy.append(selected_u);fixed.append(fixed_u);oracle.append(maximum);top1.append(selected in oracle_set);top2.append(bool(set(ranked[:2])&oracle_set));delta=selected_u-fixed_u
        decisions.append({"state_id":state_id,"sample_id":items[0]["sample_id"],"subset":subset,"score_family":family,"horizon":horizon,"selected_weight":weights[selected],"selected_utility":selected_u,"fixed_w1_utility":fixed_u,"oracle_utility":maximum,"true_gain":delta,"beneficial":delta>UTILITY_EPSILON,"harmful":delta < -UTILITY_EPSILON})
    fixed_mean=float(np.mean(fixed));oracle_mean=float(np.mean(oracle));policy_mean=float(np.mean(policy));gap=oracle_mean-fixed_mean;interventions=[row for row in decisions if row["selected_weight"]!=1.0]
    summary={"subset":subset,"score_family":family,"horizon":horizon,"states":len(states),"informative_pair_count":sum(pairs),"pairwise_accuracy":sum(count*value for count,value in correct)/sum(count for count,_ in correct) if correct else None,"defined_spearman_states":len(spearman),"undefined_spearman_fraction":1-len(spearman)/len(states) if states else None,"mean_defined_spearman":float(np.mean(spearman)) if spearman else None,"top1_oracle_set_accuracy":float(np.mean(top1)),"top2_oracle_set_coverage":float(np.mean(top2)),"policy_utility":policy_mean,"fixed_w1_utility":fixed_mean,"oracle_utility":oracle_mean,"gain_over_fixed":policy_mean-fixed_mean,"oracle_gap_capture":(policy_mean-fixed_mean)/gap if gap>0 else None,"intervention_rate":len(interventions)/len(states),"beneficial_decision_fraction":sum(row["beneficial"] for row in interventions)/len(interventions) if interventions else None,"harmful_decision_fraction":sum(row["harmful"] for row in interventions)/len(interventions) if interventions else None,"selected_weight_distribution":json.dumps(dict(sorted(Counter(str(row["selected_weight"]) for row in decisions).items())))}
    return summary,decisions


def compute_estimates(actionable_fraction:float)->list[dict[str,Any]]:
    rows=[]
    for label,count in (("W3",3),("W5",5)):
        for horizon in (4,8,16):
            extra=actionable_fraction*(count-1)*horizon
            rows.append({"candidate_set":label,"candidate_count":count,"horizon":horizon,"actionable_fraction":actionable_fraction,"additional_forward_token_units_per_generated_token":extra,"estimated_relative_forward_token_multiplier":1+extra,"note":"token-work estimate only; not wall-clock latency"})
    return rows


def analyze()->dict[str,Any]:
    with (OUT/"internal_scores.csv").open(newline="",encoding="utf-8") as handle:
        raw=list(csv.DictReader(handle))
    rows=[]
    for row in raw:
        rows.append({**row,"weight":float(row["weight"]),"horizon":int(row["horizon"]),"mip":float(row["mip"]),"parm_score":float(row["parm_score"]),"base_score":float(row["base_score"]),"ratio_score":float(row["ratio_score"]),"is_actionable":row["is_actionable"].lower()=="true"})
    summaries=[];predictions=[]
    for subset in ("actionable","all"):
        for family in SCORE_FAMILIES:
            for horizon in HORIZONS:
                summary,decision=evaluate(rows,family,horizon,subset);summaries.append(summary);predictions.extend(decision)
    atomic_csv(OUT/"horizon_summary.csv",summaries)
    atomic_csv(OUT/"pairwise_summary.csv",[{key:value for key,value in row.items() if key in ("subset","score_family","horizon","states","informative_pair_count","pairwise_accuracy","defined_spearman_states","undefined_spearman_fraction","mean_defined_spearman","top1_oracle_set_accuracy","top2_oracle_set_coverage")} for row in summaries])
    policy_rows=[{key:value for key,value in row.items() if key not in ("informative_pair_count","pairwise_accuracy","defined_spearman_states","undefined_spearman_fraction","mean_defined_spearman","top1_oracle_set_accuracy","top2_oracle_set_coverage")} for row in summaries]
    rich=json.loads((RICH/"summary.json").read_text(encoding="utf-8"))["best_family"]
    policy_rows.extend([{"subset":"all","score_family":"local_logit_k64","horizon":"local","states":rich["states"],"policy_utility":rich["policy_mip"],"fixed_w1_utility":rich["fixed_w1_mip"],"oracle_utility":rich["oracle_mip"],"gain_over_fixed":rich["gain_over_fixed"],"oracle_gap_capture":rich["oracle_gap_capture"],"intervention_rate":rich["intervention_rate"],"beneficial_decision_fraction":rich["beneficial_intervention_rate"],"harmful_decision_fraction":rich["harmful_intervention_rate"],"selected_weight_distribution":rich["selected_weight_distribution"]}])
    atomic_csv(OUT/"policy_summary.csv",policy_rows)
    actionable_fraction=len({r["state_id"] for r in rows if r["is_actionable"]})/len({r["state_id"] for r in rows});estimates=compute_estimates(actionable_fraction);atomic_csv(OUT/"compute_estimate.csv",estimates)
    primary=[row for row in summaries if row["subset"]=="actionable"]
    best=max(primary,key=lambda row:row["policy_utility"])
    supported=best["pairwise_accuracy"] is not None and best["pairwise_accuracy"]>.5 and best["mean_defined_spearman"] is not None and best["mean_defined_spearman"]>0 and best["gain_over_fixed"]>0 and best["oracle_gap_capture"]>0
    strong=supported and best["pairwise_accuracy"]>=.6 and best["oracle_gap_capture"]>=.2
    verdict="LOOKAHEAD_PARM_SUPPORTED" if strong else ("LOOKAHEAD_PARM_WEAK" if supported else "ADAPTIVE_PARM_NO_GO")
    result={"status":"COMPLETE","verdict":verdict,"primary_subset":"64 actionable states","best_direct_score":best,"horizon_improvement":{"h1_pairwise":max(row["pairwise_accuracy"] or 0 for row in primary if row["horizon"]==1),"best_long_horizon_pairwise":max(row["pairwise_accuracy"] or 0 for row in primary if row["horizon"] in (8,16,32))},"no_learned_parameters":True,"no_generation":True,"no_rescoring_with_beaver":True};atomic_json(OUT/"summary.json",result)
    report=f"""# Selective Lookahead PARM probe\n\n## Facts\n\n- Status: `{result['status']}`\n- Primary subset: 64 actionable states; secondary subset: all 250 states.\n- Best internal score/horizon: `{best['score_family']}`, H={best['horizon']}.\n- Pairwise accuracy: {best['pairwise_accuracy']:.6f}; defined-state Spearman: {best['mean_defined_spearman']:.6f}.\n- Policy utility: {best['policy_utility']:.6f}; fixed w=1: {best['fixed_w1_utility']:.6f}.\n- Gain: {best['gain_over_fixed']:.6f}; oracle-gap capture: {best['oracle_gap_capture']:.2%}.\n- Harmful decision fraction: {best['harmful_decision_fraction']}.\n\n## Interpretation\n\nVerdict: **{verdict}**. Longer horizons are evaluated as direct, parameter-free internal branch scores, not learned predictors.\n\n## Limitations\n\nThis reuses 10 prompts and existing deterministic one-step counterfactual rollouts. Internal likelihood is not an external alignment evaluator, and token-work estimates are not latency measurements.\n""";atomic_text(OUT/"report.md",report);return result
