"""Offline tie-aware and selective-policy analysis for token headroom."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / ".git").exists() and (_candidate / "PARM_TARO").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Cannot locate project root")

from PARM_TARO.adaptive_parm.token_headroom.analysis import STATE_FEATURES
from PARM_TARO.adaptive_parm.token_headroom.config import OUT, WEIGHTS
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv, atomic_json, atomic_text

TIE_OUT = OUT / "tie_aware"
EPSILONS = (1e-8, 1e-6, 1e-4, 1e-3)
PRIMARY_EPSILON = 1e-6
DELTA_GRID = (0.0, 1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def oracle_set(utilities: Mapping[float, float], epsilon: float) -> tuple[float, ...]:
    if epsilon < 0 or not utilities:
        raise ValueError("epsilon must be non-negative and utilities non-empty")
    maximum = max(float(value) for value in utilities.values())
    return tuple(sorted(float(weight) for weight, value in utilities.items() if maximum - float(value) <= epsilon))


def informative_pairs(true: Sequence[float], epsilon: float) -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(true)) for j in range(i + 1, len(true)) if abs(float(true[i]) - float(true[j])) > epsilon]


def pairwise_accuracy(true: Sequence[float], predicted: Sequence[float], epsilon: float) -> tuple[int, int, float | None]:
    pairs = informative_pairs(true, epsilon)
    correct = sum((float(true[i]) - float(true[j])) * (float(predicted[i]) - float(predicted[j])) > 0 for i, j in pairs)
    return correct, len(pairs), (correct / len(pairs) if pairs else None)


def defined_spearman(true: Sequence[float], predicted: Sequence[float], epsilon: float) -> float | None:
    if max(true) - min(true) <= epsilon or max(predicted) - min(predicted) <= 1e-15:
        return None
    value = float(spearmanr(true, predicted).statistic)
    return value if math.isfinite(value) else None


def grouped_splits(prompt_ids: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    values = np.asarray(prompt_ids)
    result = []
    for held in sorted(set(prompt_ids)):
        train = np.flatnonzero(values != held); test = np.flatnonzero(values == held)
        if set(values[train]) & set(values[test]):
            raise AssertionError("Grouped split leaked a prompt")
        result.append((train, test, held))
    return result


def _feature(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in STATE_FEATURES], dtype=np.float64)


def _design(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights.reshape(-1, 1)
    return np.concatenate((features, w, w**2, features*w, features*(w**2)), axis=1)


def _ridge() -> Pipeline:
    return Pipeline((('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0))))


def _logistic() -> Pipeline:
    return Pipeline((('scale', StandardScaler()), ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000, random_state=42))))


def _state_data() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    states_raw = read_csv(OUT / "state_features.csv")
    utility_raw = read_csv(OUT / "state_utility.csv")
    states: dict[str, dict[str, Any]] = {row["state_id"]: dict(row) for row in states_raw}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in utility_raw:
        row = dict(raw)
        for key in ("weight", "mip", "alpha_helpfulness", "alpha_harmlessness"):
            row[key] = float(row[key])
        row["candidate_next_token_id"] = int(row["candidate_next_token_id"])
        groups[row["state_id"]].append(row)
    if len(states) != 250 or len(utility_raw) != 1250 or set(states) != set(groups):
        raise ValueError("Expected complete 250-state / 1250-action artifacts")
    for state, rows in groups.items():
        rows.sort(key=lambda row: row["weight"])
        if tuple(row["weight"] for row in rows) != WEIGHTS:
            raise ValueError(f"Incomplete weight grid: {state}")
    ordered = [states[key] for key in sorted(states)]
    return ordered, states, groups


def _bootstrap_gain(rows: Sequence[Mapping[str, Any]], *, iterations: int = 5000) -> list[float]:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(float(row["oracle_gain"]))
    prompt_gain = [fmean(values) for _, values in sorted(by_prompt.items())]
    rng = np.random.default_rng(42)
    values = [fmean(prompt_gain[index] for index in rng.integers(0, len(prompt_gain), len(prompt_gain))) for _ in range(iterations)]
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def _scope_summary(state_ids: Sequence[str], states: Mapping[str, Mapping[str, Any]], groups: Mapping[str, list[dict[str, Any]]], epsilon: float) -> dict[str, Any]:
    rows = []
    set_sizes = Counter(); member = Counter(); unique_weights = Counter()
    for state_id in state_ids:
        group = groups[state_id]; utility = {row["weight"]: row["mip"] for row in group}; tied = oracle_set(utility, epsilon)
        fixed = utility[1.0]; maximum = max(utility.values())
        rows.append({"prompt_id": states[state_id]["sample_id"], "oracle_gain": maximum-fixed})
        set_sizes[len(tied)] += 1
        for weight in tied: member[str(weight)] += 1
        if len(tied) == 1: unique_weights[str(tied[0])] += 1
    fixed_mean = fmean(next(row["mip"] for row in groups[state] if row["weight"] == 1.0) for state in state_ids)
    oracle_mean = fmean(max(row["mip"] for row in groups[state]) for state in state_ids)
    spreads = [max(row["mip"] for row in groups[state])-min(row["mip"] for row in groups[state]) for state in state_ids]
    return {
        "count": len(state_ids), "mean_fixed_w1_utility": fixed_mean, "mean_tie_aware_oracle_utility": oracle_mean,
        "absolute_gain": oracle_mean-fixed_mean, "relative_gain": (oracle_mean-fixed_mean)/fixed_mean if fixed_mean else None,
        "prompt_grouped_bootstrap_ci95": _bootstrap_gain(rows), "mean_utility_spread": fmean(spreads),
        "oracle_set_size_distribution": {str(k): v for k, v in sorted(set_sizes.items())},
        "unique_oracle_weight_distribution": dict(unique_weights),
        "fraction_w1_uniquely_optimal": unique_weights["1.0"]/len(state_ids),
        "fraction_w1_in_oracle_set": member["1.0"]/len(state_ids),
        "fraction_non_w1_uniquely_better": sum(v for k,v in unique_weights.items() if k != "1.0")/len(state_ids),
    }


def build_tie_and_advantage(states: list[dict[str, Any]], state_map: dict[str, dict[str, Any]], groups: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tie_rows=[]; advantage=[]; raw_w0_tie=0
    for state in states:
        state_id=state["state_id"]; group=groups[state_id]; utility={row["weight"]:row["mip"] for row in group}; exact=oracle_set(utility,1e-12)
        raw=min(weight for weight,value in utility.items() if math.isclose(value,max(utility.values()),abs_tol=1e-12,rel_tol=0))
        if raw==0.0 and len(exact)>1: raw_w0_tie += 1
        actionable=len({row["candidate_next_token_id"] for row in group})>=2
        primary=oracle_set(utility,PRIMARY_EPSILON); reference=utility[1.0]
        for epsilon in EPSILONS:
            tied=oracle_set(utility,epsilon)
            tie_rows.append({"state_id":state_id,"prompt_id":state["sample_id"],"epsilon":epsilon,"max_utility":max(utility.values()),"oracle_set":json.dumps(tied),"oracle_set_size":len(tied),"unique_oracle":len(tied)==1,"w1_in_oracle_set":1.0 in tied,"w0_in_oracle_set":0.0 in tied,"is_actionable":actionable})
        for row in group:
            advantage.append({
                "prompt_id":state["sample_id"],"state_id":state_id,"alpha":f"{float(state['alpha_helpfulness']):.2f},{float(state['alpha_harmlessness']):.2f}",
                "alpha_helpfulness":float(state["alpha_helpfulness"]),"alpha_harmlessness":float(state["alpha_harmlessness"]),
                "generation_position":float(state["normalized_generation_position"]),
                **{name:float(state[name]) for name in STATE_FEATURES if name not in {"alpha_helpfulness","alpha_harmlessness","normalized_generation_position"}},
                "weight":row["weight"],"utility":row["mip"],"reference_w1_utility":reference,"advantage":row["mip"]-reference,
                "next_token_id":row["candidate_next_token_id"],"is_actionable":actionable,"oracle_set_membership":row["weight"] in primary,"unique_oracle":len(primary)==1,
            })
    primary_ties=[row for row in tie_rows if row["epsilon"]==PRIMARY_EPSILON]
    tie_summary={"primary_epsilon":PRIMARY_EPSILON,"epsilon_summary":{},"raw_w0_oracle_labels":sum(1 for s in states if min(oracle_set({r['weight']:r['mip'] for r in groups[s['state_id']]},1e-12))==0.0),"raw_w0_labels_caused_only_by_exact_ties":raw_w0_tie}
    for epsilon in EPSILONS:
        rows=[row for row in tie_rows if row["epsilon"]==epsilon]; tie_summary["epsilon_summary"][str(epsilon)]={"states":len(rows),"unique_oracle":sum(row["unique_oracle"] for row in rows),"tied_oracle":sum(not row["unique_oracle"] for row in rows),"fraction_unique_oracle":fmean(float(row["unique_oracle"]) for row in rows),"fraction_tied_oracle":fmean(float(not row["unique_oracle"]) for row in rows),"fraction_w1_in_oracle_set":fmean(float(row["w1_in_oracle_set"]) for row in rows),"fraction_w0_in_oracle_set":fmean(float(row["w0_in_oracle_set"]) for row in rows)}
    return tie_rows,advantage,tie_summary


def actionable_analysis(states: list[dict[str,Any]], state_map: dict[str,dict[str,Any]], groups: dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    ids=[row["state_id"] for row in states]; actionable=[state for state in ids if len({r["candidate_next_token_id"] for r in groups[state]})>=2]; non=[state for state in ids if state not in set(actionable)]
    result={"primary_epsilon":PRIMARY_EPSILON,"all":_scope_summary(ids,state_map,groups,PRIMARY_EPSILON),"actionable":_scope_summary(actionable,state_map,groups,PRIMARY_EPSILON),"non_actionable":_scope_summary(non,state_map,groups,PRIMARY_EPSILON),"utility_actionable_by_epsilon":{}}
    for epsilon in EPSILONS:
        utility_ids=[state for state in ids if max(r["mip"] for r in groups[state])-min(r["mip"] for r in groups[state])>epsilon]
        result["utility_actionable_by_epsilon"][str(epsilon)]={"count":len(utility_ids),"fraction":len(utility_ids)/len(ids)}
    result["actionable_by_alpha"]={}
    for alpha in sorted({(float(s["alpha_helpfulness"]),float(s["alpha_harmlessness"])) for s in states},reverse=True):
        subset=[s for s in actionable if (float(state_map[s]["alpha_helpfulness"]),float(state_map[s]["alpha_harmlessness"]))==alpha]
        result["actionable_by_alpha"][f"{alpha[0]:.2f},{alpha[1]:.2f}"]=_scope_summary(subset,state_map,groups,PRIMARY_EPSILON)
    result["actionable_by_position"]={}
    for position in sorted({s.rsplit("_s",1)[1] for s in actionable}):
        subset=[s for s in actionable if s.rsplit("_s",1)[1]==position]
        result["actionable_by_position"][position]=_scope_summary(subset,state_map,groups,PRIMARY_EPSILON)
    return result


def generic_tie_aware_probe(states: list[dict[str,Any]], groups: dict[str,list[dict[str,Any]]]) -> tuple[dict[str,Any],dict[str,dict[float,float]]]:
    prompt_ids=[row["sample_id"] for row in states]; predictions:dict[str,dict[float,float]]={}; all_correct=all_pairs=possible=0; spearman=[]; undefined=0; set_correct=top2_correct=0
    for train_idx,test_idx,held in grouped_splits(prompt_ids):
        x=[];w=[];y=[]
        for index in train_idx:
            state=states[index]; feature=_feature(state)
            for row in groups[state["state_id"]]: x.append(feature);w.append(row["weight"]);y.append(row["mip"])
        model=_ridge().fit(_design(np.stack(x),np.asarray(w)),np.asarray(y))
        for index in test_idx:
            state=states[index]; group=groups[state["state_id"]]; feature=_feature(state); weights=np.asarray([r["weight"] for r in group]); true=np.asarray([r["mip"] for r in group]); pred=model.predict(_design(np.repeat(feature[None,:],len(group),axis=0),weights)); predictions[state["state_id"]]=dict(zip(weights.tolist(),pred.tolist()))
            correct,total,accuracy=pairwise_accuracy(true,pred,PRIMARY_EPSILON); all_correct+=correct;all_pairs+=total;possible+=len(true)*(len(true)-1)//2
            value=defined_spearman(true,pred,PRIMARY_EPSILON)
            if value is None: undefined+=1
            else:spearman.append(value)
            tied=set(oracle_set(dict(zip(weights,true)),PRIMARY_EPSILON)); order=np.argsort(-pred,kind="stable");set_correct+=float(weights[order[0]] in tied);top2_correct+=float(bool(set(weights[order[:2]])&tied))
    result={"split":"leave-one-unique-prompt-out","epsilon":PRIMARY_EPSILON,"informative_pair_count":all_pairs,"all_pair_count":possible,"informative_pair_fraction":all_pairs/possible,"pairwise_ranking_accuracy":all_correct/all_pairs if all_pairs else None,"mean_defined_spearman":fmean(spearman) if spearman else None,"defined_spearman_states":len(spearman),"undefined_spearman_states":undefined,"fraction_undefined_spearman":undefined/len(states),"oracle_set_accuracy":set_correct/len(states),"top2_oracle_set_coverage":top2_correct/len(states)}
    return result,predictions


def actionability_probe(states:list[dict[str,Any]],groups:dict[str,list[dict[str,Any]]])->tuple[dict[str,Any],dict[str,float]]:
    x=np.stack([_feature(s) for s in states]); y=np.asarray([len({r["candidate_next_token_id"] for r in groups[s["state_id"]]})>=2 for s in states],dtype=int); prompts=[s["sample_id"] for s in states]; logistic_prob=np.zeros(len(states)); forest_prob=np.zeros(len(states))
    for train,test,_ in grouped_splits(prompts):
        logistic=_logistic().fit(x[train],y[train]); logistic_prob[test]=logistic.predict_proba(x[test])[:,1]
        forest=RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=1).fit(x[train],y[train]);forest_prob[test]=forest.predict_proba(x[test])[:,1]
    def metrics(prob:np.ndarray)->dict[str,float]:
        return {"auroc":float(roc_auc_score(y,prob)),"auprc":float(average_precision_score(y,prob)),"balanced_accuracy":float(balanced_accuracy_score(y,prob>=.5))}
    return {"split":"leave-one-unique-prompt-out","positive_count":int(y.sum()),"positive_fraction":float(y.mean()),"logistic_regression":metrics(logistic_prob),"random_forest_diagnostic":metrics(forest_prob),"features":list(STATE_FEATURES)},dict(zip((s["state_id"] for s in states),logistic_prob.tolist()))


def actionable_advantage_probe(states:list[dict[str,Any]],groups:dict[str,list[dict[str,Any]]])->dict[str,Any]:
    actionable=[s for s in states if len({r["candidate_next_token_id"] for r in groups[s["state_id"]]})>=2];prompts=[s["sample_id"] for s in actionable];true_all=[];pred_all=[];correct=total=possible=0;spearman=[];undefined=0;set_correct=top2=0
    for train_idx,test_idx,_ in grouped_splits(prompts):
        x=[];w=[];y=[]
        for index in train_idx:
            state=actionable[index];feature=_feature(state);ref=next(r["mip"] for r in groups[state["state_id"]] if r["weight"]==1.0)
            for action in groups[state["state_id"]]:x.append(feature);w.append(action["weight"]);y.append(action["mip"]-ref)
        model=_ridge().fit(_design(np.stack(x),np.asarray(w)),np.asarray(y))
        for index in test_idx:
            state=actionable[index];group=groups[state["state_id"]];feature=_feature(state);weights=np.asarray([r["weight"] for r in group]);ref=next(r["mip"] for r in group if r["weight"]==1.0);true=np.asarray([r["mip"]-ref for r in group]);pred=model.predict(_design(np.repeat(feature[None,:],len(group),axis=0),weights));pred[np.isclose(weights,1.0)]=0.;true_all.extend(true);pred_all.extend(pred);c,n,_=pairwise_accuracy(true,pred,PRIMARY_EPSILON);correct+=c;total+=n;possible+=len(true)*(len(true)-1)//2;rho=defined_spearman(true,pred,PRIMARY_EPSILON)
            if rho is None:undefined+=1
            else:spearman.append(rho)
            tied=set(oracle_set(dict(zip(weights,true)),PRIMARY_EPSILON));order=np.argsort(-pred,kind="stable");set_correct+=float(weights[order[0]] in tied);top2+=float(bool(set(weights[order[:2]])&tied))
    truth=np.asarray(true_all);prediction=np.asarray(pred_all)
    return {"split":"leave-one-unique-prompt-out actionable states only","model":"StandardScaler + Ridge(alpha=10) advantage predictor","states":len(actionable),"utility_rows":len(true_all),"advantage_mse":float(np.mean((truth-prediction)**2)),"advantage_mae":float(np.mean(np.abs(truth-prediction))),"informative_pair_count":total,"all_pair_count":possible,"informative_pair_fraction":total/possible,"pairwise_ranking_accuracy":correct/total if total else None,"mean_defined_spearman":fmean(spearman) if spearman else None,"defined_spearman_states":len(spearman),"undefined_spearman_states":undefined,"fraction_undefined_spearman":undefined/len(actionable),"oracle_set_accuracy":set_correct/len(actionable),"top2_oracle_set_coverage":top2/len(actionable)}


def select_delta(training_predictions: Sequence[Mapping[str,Any]], delta_grid: Sequence[float]=DELTA_GRID) -> tuple[float,dict[str,float]]:
    """Select a conservative delta solely from caller-supplied training OOF rows."""
    if not training_predictions: raise ValueError("No training predictions for delta selection")
    scores={}
    for delta in delta_grid:
        utility=[]
        for row in training_predictions:
            intervene=float(row["actionable_probability"])>=.5 and float(row["predicted_max_advantage"])>delta
            utility.append(float(row["predicted_choice_true_utility"] if intervene else row["fixed_true_utility"]))
        scores[str(delta)]=fmean(utility)
    best=max(scores.values()); chosen=max(float(delta) for delta,value in ((key,scores[key]) for key in scores) if math.isclose(value,best,abs_tol=1e-12,rel_tol=0))
    return chosen,scores


def assert_delta_training_only(held_prompt: str, training_predictions: Sequence[Mapping[str,Any]]) -> list[str]:
    prompts=sorted({str(row["prompt_id"]) for row in training_predictions})
    if held_prompt in prompts:
        raise AssertionError("Held-out prompt leaked into delta selection")
    return prompts


def _inner_oof(outer_train: list[dict[str,Any]],groups:dict[str,list[dict[str,Any]]])->list[dict[str,Any]]:
    prompts=[s["sample_id"] for s in outer_train]; rows=[]
    for train_idx,test_idx,_ in grouped_splits(prompts):
        train_states=[outer_train[i] for i in train_idx]; test_states=[outer_train[i] for i in test_idx]
        labels=np.asarray([len({r["candidate_next_token_id"] for r in groups[s["state_id"]]})>=2 for s in train_states],dtype=int)
        classifier=_logistic().fit(np.stack([_feature(s) for s in train_states]),labels)
        actionable=[s for s,label in zip(train_states,labels) if label]
        x=[];w=[];y=[]
        for state in actionable:
            feature=_feature(state);ref=next(r["mip"] for r in groups[state["state_id"]] if r["weight"]==1.0)
            for action in groups[state["state_id"]]:x.append(feature);w.append(action["weight"]);y.append(action["mip"]-ref)
        reg=_ridge().fit(_design(np.stack(x),np.asarray(w)),np.asarray(y))
        for state in test_states:
            feature=_feature(state); prob=float(classifier.predict_proba(feature[None,:])[0,1]);group=groups[state["state_id"]];weights=np.asarray([r["weight"] for r in group]);pred=reg.predict(_design(np.repeat(feature[None,:],len(group),axis=0),weights));pred[np.isclose(weights,1.0)]=0.;choice=int(np.argmax(pred));ref=next(r["mip"] for r in group if r["weight"]==1.0)
            rows.append({"prompt_id":state["sample_id"],"actionable_probability":prob,"predicted_max_advantage":float(pred[choice]),"predicted_choice_true_utility":group[choice]["mip"],"fixed_true_utility":ref})
    return rows


def selective_policy(states:list[dict[str,Any]],groups:dict[str,list[dict[str,Any]]],generic_predictions:dict[str,dict[float,float]])->tuple[list[dict[str,Any]],dict[str,Any]]:
    predictions=[];prompts=[s["sample_id"] for s in states]
    for train_idx,test_idx,held in grouped_splits(prompts):
        train_states=[states[i] for i in train_idx];test_states=[states[i] for i in test_idx];labels=np.asarray([len({r["candidate_next_token_id"] for r in groups[s["state_id"]]})>=2 for s in train_states],dtype=int)
        classifier=_logistic().fit(np.stack([_feature(s) for s in train_states]),labels)
        actionable=[s for s,label in zip(train_states,labels) if label];x=[];w=[];y=[]
        for state in actionable:
            feature=_feature(state);ref=next(r["mip"] for r in groups[state["state_id"]] if r["weight"]==1.0)
            for action in groups[state["state_id"]]:x.append(feature);w.append(action["weight"]);y.append(action["mip"]-ref)
        reg=_ridge().fit(_design(np.stack(x),np.asarray(w)),np.asarray(y));inner=_inner_oof(train_states,groups);delta,delta_scores=select_delta(inner);delta_prompts=assert_delta_training_only(held,inner)
        for state in test_states:
            feature=_feature(state);prob=float(classifier.predict_proba(feature[None,:])[0,1]);group=groups[state["state_id"]];weights=np.asarray([r["weight"] for r in group]);true=np.asarray([r["mip"] for r in group]);pred=reg.predict(_design(np.repeat(feature[None,:],len(group),axis=0),weights));pred[np.isclose(weights,1.0)]=0.;best=int(np.argmax(pred));intervene=prob>=.5 and float(pred[best])>delta;selected=best if intervene else int(np.flatnonzero(np.isclose(weights,1.0))[0]);fixed=float(true[np.flatnonzero(np.isclose(weights,1.0))[0]]);tied=oracle_set(dict(zip(weights,true)),PRIMARY_EPSILON)
            generic=generic_predictions[state["state_id"]];generic_weight=max(WEIGHTS,key=lambda value:generic[value]);generic_utility=next(r["mip"] for r in group if r["weight"]==generic_weight)
            predictions.append({"prompt_id":state["sample_id"],"state_id":state["state_id"],"held_out_prompt":held,"delta":delta,"delta_training_prompts":json.dumps(delta_prompts),"actionable_probability":prob,"predicted_max_advantage":float(pred[best]),"predicted_candidate_weight":float(weights[best]),"predicted_candidate_true_utility":float(true[best]),"selected_weight":float(weights[selected]),"intervened":intervene,"selected_utility":float(true[selected]),"fixed_w1_utility":fixed,"true_advantage":float(true[selected]-fixed),"beneficial_intervention":bool(intervene and true[selected]>fixed+PRIMARY_EPSILON),"harmful_intervention":bool(intervene and true[selected]<fixed-PRIMARY_EPSILON),"oracle_set_membership":float(weights[selected]) in tied,"generic_weight":generic_weight,"generic_utility":generic_utility,"oracle_utility":float(true.max())})
    fixed=fmean(r["fixed_w1_utility"] for r in predictions);generic=fmean(r["generic_utility"] for r in predictions);selective=fmean(r["selected_utility"] for r in predictions);oracle=fmean(r["oracle_utility"] for r in predictions);interventions=[r for r in predictions if r["intervened"]]
    sensitivity={}
    for delta in DELTA_GRID:
        chosen=[];selected_weights=[];advantages=[]
        for row in predictions:
            intervene=row["actionable_probability"]>=.5 and row["predicted_max_advantage"]>delta
            value=row["predicted_candidate_true_utility"] if intervene else row["fixed_w1_utility"]
            chosen.append(value);selected_weights.append(row["predicted_candidate_weight"] if intervene else 1.0);advantages.append(value-row["fixed_w1_utility"])
        active=[value for value,weight in zip(advantages,selected_weights) if weight!=1.0]
        sensitivity[str(delta)]={"policy_mip":fmean(chosen),"gain_over_fixed":fmean(chosen)-fixed,"intervention_rate":sum(weight!=1.0 for weight in selected_weights)/len(selected_weights),"mean_true_advantage_on_interventions":fmean(active) if active else None,"fraction_harmful_interventions":sum(value<-PRIMARY_EPSILON for value in active)/len(active) if active else None,"selected_weight_distribution":dict(Counter(str(value) for value in selected_weights))}
    summary={"split":"nested leave-one-unique-prompt-out","delta_selection":"inner grouped OOF on outer-training prompts only","delta_grid":list(DELTA_GRID),"selected_delta_distribution":dict(Counter(str(r["delta"]) for r in predictions)),"fixed_delta_sensitivity":sensitivity,"fixed_w1_mip":fixed,"generic_policy_mip":generic,"selective_policy_mip":selective,"oracle_mip":oracle,"selective_gain_over_fixed":selective-fixed,"selective_relative_gain_over_fixed":(selective-fixed)/fixed,"selective_oracle_gap_capture":(selective-fixed)/(oracle-fixed) if oracle>fixed else None,"generic_oracle_gap_capture":(generic-fixed)/(oracle-fixed) if oracle>fixed else None,"selected_weight_distribution":dict(Counter(str(r["selected_weight"]) for r in predictions)),"intervention_rate":len(interventions)/len(predictions),"mean_true_advantage_on_interventions":fmean(r["true_advantage"] for r in interventions) if interventions else None,"fraction_harmful_interventions":fmean(float(r["harmful_intervention"]) for r in interventions) if interventions else None,"fraction_beneficial_interventions":fmean(float(r["beneficial_intervention"]) for r in interventions) if interventions else None,"oracle_set_accuracy":fmean(float(r["oracle_set_membership"]) for r in predictions)}
    return predictions,summary


def render_report(tie:dict[str,Any],actionable:dict[str,Any],structure:dict[str,Any],stage_a:dict[str,Any],selective:dict[str,Any],verdict:str)->str:
    primary=tie["epsilon_summary"][str(PRIMARY_EPSILON)];a=actionable["actionable"];logit=stage_a["logistic_regression"];generic=structure["generic_all_states"];stage_b=structure["stage_b_actionable_advantage"]
    return f"""# Selective Adaptive PARM: tie-aware offline analysis

Scientific verdict: **{verdict}**

## Tie-aware oracle (`epsilon={PRIMARY_EPSILON:g}`)

- Tied states: {primary['tied_oracle']} / {primary['states']}
- Unique states: {primary['unique_oracle']} / {primary['states']}
- Raw `w=0` labels caused only by exact argmax ties: {tie['raw_w0_labels_caused_only_by_exact_ties']} / {tie['raw_w0_oracle_labels']}
- Fraction with `w=1` in oracle set: {primary['fraction_w1_in_oracle_set']:.2%}

## Actionable states

- Count: {a['count']}
- Fixed `w=1` utility: {a['mean_fixed_w1_utility']:.9f}
- Oracle utility: {a['mean_tie_aware_oracle_utility']:.9f}
- Absolute gain: {a['absolute_gain']:.9f}; relative gain: {a['relative_gain']:.2%}
- Prompt-grouped bootstrap CI95: {a['prompt_grouped_bootstrap_ci95']}

## Tie-aware structure

- Generic informative pairs: {generic['informative_pair_count']} ({generic['informative_pair_fraction']:.2%})
- Generic pairwise accuracy: {generic['pairwise_ranking_accuracy']:.6f}
- Generic mean defined-state Spearman: {generic['mean_defined_spearman']:.6f}
- Undefined Spearman fraction: {generic['fraction_undefined_spearman']:.2%}
- Generic oracle-set accuracy / top-2 coverage: {generic['oracle_set_accuracy']:.2%} / {generic['top2_oracle_set_coverage']:.2%}
- Stage-B actionable advantage pairwise accuracy: {stage_b['pairwise_ranking_accuracy']:.6f}
- Stage-B defined-state Spearman: {stage_b['mean_defined_spearman']:.6f}

## Selective feasibility

- Stage-A logistic AUROC/AUPRC/balanced accuracy: {logit['auroc']:.6f} / {logit['auprc']:.6f} / {logit['balanced_accuracy']:.6f}
- Fixed / generic / selective / oracle MIP: {selective['fixed_w1_mip']:.9f} / {selective['generic_policy_mip']:.9f} / {selective['selective_policy_mip']:.9f} / {selective['oracle_mip']:.9f}
- Selective gain: {selective['selective_gain_over_fixed']:.9f}; gap capture: {selective['selective_oracle_gap_capture']:.2%}
- Intervention rate: {selective['intervention_rate']:.2%}
- Beneficial / harmful interventions: {selective['fraction_beneficial_interventions']:.2%} / {selective['fraction_harmful_interventions']:.2%}

All features precede scoring. Splits are grouped by unique prompt, and each
outer-fold delta is selected only from inner OOF predictions over training
prompts. This remains a small 10-prompt offline diagnostic.
"""


def run() -> dict[str,Any]:
    TIE_OUT.mkdir(parents=True,exist_ok=True);states,state_map,groups=_state_data();tie_rows,advantage,tie=build_tie_and_advantage(states,state_map,groups);actionable=actionable_analysis(states,state_map,groups);generic_structure,generic=generic_tie_aware_probe(states,groups);stage_b=actionable_advantage_probe(states,groups);structure={"primary_epsilon":PRIMARY_EPSILON,"generic_all_states":generic_structure,"stage_b_actionable_advantage":stage_b};stage_a,_=actionability_probe(states,groups);predictions,selective=selective_policy(states,groups,generic)
    primary=tie["epsilon_summary"][str(PRIMARY_EPSILON)];gain_not_ties=actionable["actionable"]["absolute_gain"]>0 and primary["fraction_unique_oracle"]>0;predictive=stage_b["pairwise_ranking_accuracy"] is not None and stage_b["pairwise_ranking_accuracy"]>.5 and stage_b["mean_defined_spearman"] is not None and stage_b["mean_defined_spearman"]>0;supported=gain_not_ties and predictive and selective["selective_gain_over_fixed"]>0 and 0<selective["intervention_rate"]<.9
    if supported:verdict="SELECTIVE_TOKEN_ROUTING_SUPPORTED"
    elif actionable["all"]["absolute_gain"]>0:verdict="TOKEN_HEADROOM_EXISTS_BUT_NOT_PREDICTABLE"
    else:verdict="TOKEN_LEVEL_HEADROOM_TOO_WEAK"
    atomic_csv(TIE_OUT/"tie_aware_oracle.csv",tie_rows);atomic_json(TIE_OUT/"actionable_summary.json",{**actionable,"tie_summary":tie});atomic_csv(TIE_OUT/"advantage_dataset.csv",advantage);atomic_json(TIE_OUT/"tie_aware_structure_probe.json",structure);atomic_json(TIE_OUT/"actionability_probe.json",stage_a);atomic_csv(TIE_OUT/"selective_policy_predictions.csv",predictions);atomic_json(TIE_OUT/"selective_router_summary.json",{**selective,"scientific_verdict":verdict});atomic_text(TIE_OUT/"report.md",render_report(tie,actionable,structure,stage_a,selective,verdict));return {"tie":tie,"actionable":actionable,"structure":structure,"actionability":stage_a,"selective":selective,"verdict":verdict}


if __name__=="__main__":
    result=run();print(json.dumps(result,indent=2,sort_keys=True))
