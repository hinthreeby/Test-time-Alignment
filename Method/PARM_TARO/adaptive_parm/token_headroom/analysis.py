"""Headroom, prompt-cluster bootstrap, and simple structure probes."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from PARM_TARO.adaptive_parm.prompt_router.train import _pairwise_accuracy, utility_design

from .config import OUT, PHASE09, WEIGHTS
from .io import atomic_csv, atomic_json, atomic_text, read_jsonl


STATE_FEATURES=(
    "alpha_helpfulness","alpha_harmlessness","normalized_generation_position","generated_token_count","prompt_token_count",
    "base_entropy","base_top1_probability","base_top1_top2_margin","base_topk_mass",
    "parm_entropy","parm_top1_probability","parm_top1_top2_margin","parm_topk_mass",
    "top1_disagreement","topk_overlap","js_divergence","symmetric_kl","logprob_correlation","candidate_rank_disagreement",
)


def _csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))


def _normalize(value: float, low: float, high: float) -> float:
    return max(0.0,min(1.0,(value-low)/(high-low)))


def _state_vector(row: dict[str,str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in STATE_FEATURES],dtype=np.float64)


def _position_label(state_id: str) -> str:
    return state_id.rsplit("_s",1)[1]


def _structure_probe(states: dict[str,dict[str,str]], utilities: dict[str,list[dict[str,Any]]], oracles: dict[str,dict[str,Any]]) -> dict[str,Any]:
    prompts=sorted({row["sample_id"] for row in states.values()}); oof=[]; all_true=[]; all_pred=[]; ranks=[]; pairs=[]
    for fold,held in enumerate(prompts):
        train_states=[state for state,row in states.items() if row["sample_id"]!=held]; test_states=[state for state,row in states.items() if row["sample_id"]==held]
        x=[]; w=[]; y=[]
        for state in train_states:
            vector=_state_vector(states[state])
            for row in utilities[state]: x.append(vector); w.append(row["weight"]); y.append(row["mip"])
        model=Pipeline((("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))))
        model.fit(utility_design(np.stack(x),np.asarray(w)),np.asarray(y))
        for state in test_states:
            group=sorted(utilities[state],key=lambda row:row["weight"]); vector=_state_vector(states[state]); weights=np.asarray([row["weight"] for row in group]); true=np.asarray([row["mip"] for row in group]); predicted=model.predict(utility_design(np.repeat(vector[None,:],len(group),axis=0),weights)); order=np.argsort(-predicted,kind="stable"); chosen=group[int(order[0])]; oracle=oracles[state]
            correlation=spearmanr(true,predicted).statistic; ranks.append(0.0 if not math.isfinite(float(correlation)) else float(correlation)); pairs.append(_pairwise_accuracy(true,predicted)); all_true.extend(true); all_pred.extend(predicted)
            oof.append({"state_id":state,"selected_weight":chosen["weight"],"selected_mip":chosen["mip"],"oracle_weight":oracle["oracle_weight"],"oracle_correct":chosen["weight"]==oracle["oracle_weight"],"top2_correct":oracle["oracle_weight"] in set(weights[order[:2]].tolist())})
    fixed=fmean(next(row["mip"] for row in utilities[state] if row["weight"]==1.0) for state in utilities); oracle_mean=fmean(row["oracle_mip"] for row in oracles.values()); policy=fmean(row["selected_mip"] for row in oof)
    labels=np.asarray([WEIGHTS.index(oracles[state]["oracle_weight"]) for state in sorted(states)]); matrix=np.stack([_state_vector(states[state]) for state in sorted(states)]); mi=mutual_info_classif(matrix,labels,discrete_features=False,random_state=42)
    return {
        "split":"leave-one-unique-prompt-out","num_folds":len(prompts),"prompt_leakage":False,"model":"StandardScaler + Ridge(alpha=10), action interactions",
        "utility_mse":float(np.mean((np.asarray(all_true)-np.asarray(all_pred))**2)),"utility_mae":float(np.mean(np.abs(np.asarray(all_true)-np.asarray(all_pred)))),
        "spearman":fmean(ranks),"pairwise_ranking_accuracy":fmean(pairs),"oracle_action_accuracy":fmean(float(row["oracle_correct"]) for row in oof),"top2_action_accuracy":fmean(float(row["top2_correct"]) for row in oof),
        "offline_policy_mip":policy,"fixed_w1_mip":fixed,"oracle_mip":oracle_mean,"oracle_gap_capture":(policy-fixed)/(oracle_mean-fixed) if oracle_mean>fixed else None,
        "selected_weight_distribution":dict(Counter(str(row["selected_weight"]) for row in oof)),"mutual_information_by_feature":dict(zip(STATE_FEATURES,(float(value) for value in mi))),
        "prompt_level_comparison":{"spearman":0.006788626103585345,"pairwise_ranking_accuracy":0.5025809484890339},
    }


def run_analysis() -> dict[str,Any]:
    leakage=json.loads((OUT/"leakage_audit.json").read_text())
    if leakage["status"]!="PASS":
        summary={"status":"INVALID","verdict":"TOKEN_LEVEL_HEADROOM_INVALID","reason":"leakage audit failed","leakage":leakage}; atomic_json(OUT/"headroom_summary.json",summary); return summary
    rollouts=read_jsonl(OUT/"counterfactual_rollouts.jsonl"); rewards={row["rollout_id"]:float(row["reward_score"]) for row in _csv(OUT/"reward_scores.csv")}; costs={row["rollout_id"]:float(row["cost_score"]) for row in _csv(OUT/"cost_scores.csv")}; state_rows=_csv(OUT/"state_features.csv"); states={row["state_id"]:row for row in state_rows}
    if set(rewards)!=set(costs) or set(rewards)!={row["rollout_id"] for row in rollouts}: raise ValueError("Rollout/score keys differ")
    metrics=json.loads(PHASE09.read_text()); anchors=metrics["normalization"]["anchors"]; utility_rows=[]
    for row in rollouts:
        reward=rewards[row["rollout_id"]]; cost=costs[row["rollout_id"]]; harmless=-cost; help_n=_normalize(reward,float(anchors["helpfulness"]["low"]),float(anchors["helpfulness"]["high"])); safe_n=_normalize(harmless,float(anchors["harmlessness"]["low"]),float(anchors["harmlessness"]["high"])); ah=float(row["alpha_helpfulness"]); ass=float(row["alpha_harmlessness"]); mip=ah*help_n+ass*safe_n; denom=math.hypot(ah,ass)*math.hypot(help_n,safe_n); pcs=0.0 if denom==0 else mip/denom
        utility_rows.append({"rollout_id":row["rollout_id"],"state_id":row["state_id"],"case_id":row["case_id"],"sample_id":row["sample_id"],"alpha_helpfulness":ah,"alpha_harmlessness":ass,"weight":float(row["weight"]),"candidate_next_token_id":row["candidate_next_token_id"],"all_weights_same_next_token":row["all_weights_same_next_token"],"helpfulness_raw":reward,"harmlessness_raw":harmless,"normalized_helpfulness":help_n,"normalized_harmlessness":safe_n,"mip":mip,"pcs":pcs})
    atomic_csv(OUT/"state_utility.csv",utility_rows)
    groups=defaultdict(list)
    for row in utility_rows: groups[row["state_id"]].append(row)
    oracle_rows=[]
    for state,group in sorted(groups.items()):
        maximum=max(row["mip"] for row in group); tied=sorted((row for row in group if math.isclose(row["mip"],maximum,abs_tol=1e-12,rel_tol=0)),key=lambda row:row["weight"]); oracle=tied[0]; fixed=next(row for row in group if row["weight"]==1.0); spread=maximum-min(row["mip"] for row in group)
        oracle_rows.append({"state_id":state,"case_id":oracle["case_id"],"sample_id":oracle["sample_id"],"alpha_helpfulness":oracle["alpha_helpfulness"],"alpha_harmlessness":oracle["alpha_harmlessness"],"position_bin":_position_label(state),"oracle_weight":oracle["weight"],"oracle_tied_weights":json.dumps([row["weight"] for row in tied]),"oracle_mip":maximum,"fixed_w1_mip":fixed["mip"],"oracle_gain":maximum-fixed["mip"],"utility_spread":spread,"all_weights_same_next_token":oracle["all_weights_same_next_token"]})
    atomic_csv(OUT/"oracle_actions.csv",oracle_rows); oracles={row["state_id"]:row for row in oracle_rows}
    dist_rows=[]
    scopes=[("overall","ALL",oracle_rows)]
    for alpha in sorted({(row["alpha_helpfulness"],row["alpha_harmlessness"]) for row in oracle_rows},reverse=True): scopes.append(("alpha",f"{alpha[0]:.2f},{alpha[1]:.2f}",[row for row in oracle_rows if (row["alpha_helpfulness"],row["alpha_harmlessness"])==alpha]))
    for position in sorted({row["position_bin"] for row in oracle_rows}): scopes.append(("position",position,[row for row in oracle_rows if row["position_bin"]==position]))
    for kind,label,rows in scopes:
        counts=Counter(row["oracle_weight"] for row in rows)
        for weight in WEIGHTS: dist_rows.append({"scope":kind,"scope_value":label,"weight":weight,"count":counts[weight],"fraction":counts[weight]/len(rows)})
    atomic_csv(OUT/"weight_distribution.csv",dist_rows)
    position_rows=[]
    for position in sorted({row["position_bin"] for row in oracle_rows}):
        rows=[row for row in oracle_rows if row["position_bin"]==position]; position_rows.append({"position_bin":position,"n":len(rows),"mean_oracle_mip":fmean(row["oracle_mip"] for row in rows),"mean_fixed_w1_mip":fmean(row["fixed_w1_mip"] for row in rows),"mean_oracle_gain":fmean(row["oracle_gain"] for row in rows),"fraction_oracle_not_w1":fmean(float(row["oracle_weight"]!=1.0) for row in rows),"mean_utility_spread":fmean(row["utility_spread"] for row in rows),"fraction_all_same_next_token":fmean(float(row["all_weights_same_next_token"]) for row in rows)})
    atomic_csv(OUT/"position_analysis.csv",position_rows)
    utility_groups={state:sorted(rows,key=lambda row:row["weight"]) for state,rows in groups.items()}; structure=_structure_probe(states,utility_groups,oracles); atomic_json(OUT/"structure_probe.json",structure)
    oracle_mean=fmean(row["oracle_mip"] for row in oracle_rows); fixed_mean=fmean(row["fixed_w1_mip"] for row in oracle_rows); gain=oracle_mean-fixed_mean; relative=gain/fixed_mean if fixed_mean else None; nonref=fmean(float(row["oracle_weight"]!=1.0) for row in oracle_rows); actionability=1.0-fmean(float(row["all_weights_same_next_token"]) for row in oracle_rows); distribution=Counter(str(row["oracle_weight"]) for row in oracle_rows)
    prompt_groups=defaultdict(list)
    for row in oracle_rows: prompt_groups[row["sample_id"]].append(row)
    prompt_agg=[{"sample_id":key,"oracle_mip":fmean(row["oracle_mip"] for row in rows),"fixed_w1_mip":fmean(row["fixed_w1_mip"] for row in rows)} for key,rows in sorted(prompt_groups.items())]
    rng=np.random.default_rng(42); boot=[]
    for _ in range(2000):
        sampled=rng.choice(len(prompt_agg),len(prompt_agg),replace=True); boot.append(fmean(prompt_agg[index]["oracle_mip"]-prompt_agg[index]["fixed_w1_mip"] for index in sampled))
    clear=(relative is not None and relative>=.02 and nonref>=.20 and len(distribution)>=2 and actionability>=.10 and structure["spearman"]>=.10 and structure["pairwise_ranking_accuracy"]>=.55 and structure["oracle_gap_capture"]>0)
    weak=(relative is not None and relative>=.005 and nonref>=.10 and len(distribution)>=2)
    verdict="TOKEN_LEVEL_HEADROOM_CLEAR" if clear else ("TOKEN_LEVEL_HEADROOM_WEAK" if weak else "TOKEN_LEVEL_HEADROOM_ABSENT")
    summary={"status":"COMPLETE","verdict":verdict,"num_prompts":len(prompt_groups),"num_cases":len({row['case_id'] for row in oracle_rows}),"num_states":len(oracle_rows),"weights":list(WEIGHTS),"mean_oracle_state_utility":oracle_mean,"mean_fixed_w1_utility":fixed_mean,"absolute_oracle_gain":gain,"relative_oracle_headroom":relative,"fraction_states_oracle_not_w1":nonref,"oracle_weight_distribution":dict(distribution),"mean_utility_spread":fmean(row["utility_spread"] for row in oracle_rows),"fraction_all_weights_same_next_token":1.0-actionability,"actionable_state_fraction":actionability,"prompt_bootstrap_gain_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"normalization":metrics["normalization"],"structure_probe":structure,"decision_thresholds":{"clear":"relative>=2%, nonref>=20%, >=2 weights, actionable>=10%, Spearman>=.10, pairwise>=.55, learned gap capture>0","weak":"relative>=0.5%, nonref>=10%, >=2 weights"}}
    atomic_json(OUT/"headroom_summary.json",summary); atomic_text(OUT/"report.md",render_report(summary)); return summary


def render_report(s: dict[str,Any]) -> str:
    p=s["structure_probe"]
    return f"""# Token-level Adaptive PARM headroom

Verdict: **{s['verdict']}**

- Prompts/cases/states: {s['num_prompts']} / {s['num_cases']} / {s['num_states']}
- Mean fixed `w=1` utility: `{s['mean_fixed_w1_utility']:.9f}`
- Mean per-state oracle utility: `{s['mean_oracle_state_utility']:.9f}`
- Absolute gain: `{s['absolute_oracle_gain']:.9f}`
- Relative headroom: `{s['relative_oracle_headroom']:.2%}`
- States preferring `w != 1`: `{s['fraction_states_oracle_not_w1']:.2%}`
- States where all weights choose the same next token: `{s['fraction_all_weights_same_next_token']:.2%}`
- Prompt-cluster bootstrap gain CI95: `{s['prompt_bootstrap_gain_ci95']}`

## Structure probe

- Spearman: `{p['spearman']:.6f}` (failed prompt-level baseline `0.006789`)
- Pairwise ranking accuracy: `{p['pairwise_ranking_accuracy']:.6f}` (prompt-level `0.502581`)
- Oracle action accuracy: `{p['oracle_action_accuracy']:.2%}`
- Top-2 action accuracy: `{p['top2_action_accuracy']:.2%}`
- Learned offline policy MIP: `{p['offline_policy_mip']:.9f}`
- Learned oracle-gap capture: `{p['oracle_gap_capture']:.2%}`

All bootstrap samples resample unique prompts, not token states. Reward/cost are
labels only. No router or PBLoRA parameter was trained during state collection.
"""
