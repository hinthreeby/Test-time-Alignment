"""Leakage-safe grouped representation diagnostics over cached frozen features."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any,Mapping,Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from PARM_TARO.adaptive_parm.token_headroom.tie_aware import oracle_set,pairwise_accuracy,defined_spearman
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_csv,atomic_json,atomic_text
from .config import DELTA_GRID,EPSILON,OUT,PCA_DIMS,RIDGE_ALPHAS,SOURCE,TIE,TOP_K_VALUES,WEIGHTS
from .features import ACTION_MODEL_COLUMNS


def _csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


@dataclass
class Data:
    ids:list[str];prompts:list[str];scalar:np.ndarray;geometry:dict[int,np.ndarray];hidden:np.ndarray;action:dict[str,np.ndarray];utility:dict[str,np.ndarray];tokens:dict[str,np.ndarray];actionable:np.ndarray;fixed:np.ndarray


def load_data()->Data:
    scalar_rows=_csv(OUT/"scalar_features.csv");ids=[r["state_id"] for r in scalar_rows];prompts=[r["prompt_id"] for r in scalar_rows];scalar_names=[name for name in scalar_rows[0] if name not in {"state_id","prompt_id"}];scalar=np.asarray([[float(r[n]) for n in scalar_names] for r in scalar_rows])
    with np.load(OUT/"logit_geometry_features.npz",allow_pickle=False) as d:
        if d["state_ids"].tolist()!=ids:raise ValueError("Geometry state order mismatch")
        geometry={k:d[f"k{k}"].copy() for k in TOP_K_VALUES}
    with np.load(OUT/"hidden_features.npz",allow_pickle=False) as d:
        if d["state_ids"].tolist()!=ids:raise ValueError("Hidden state order mismatch")
        hidden=np.concatenate((d["h_base"],d["h_parm"],d["h_delta"],d["h_abs_delta"],d["cosine"][:,None]),axis=1)
    action_names=ACTION_MODEL_COLUMNS
    action_groups:dict[str,list[dict[str,str]]]=defaultdict(list)
    for row in _csv(OUT/"action_features.csv"):action_groups[row["state_id"]].append(row)
    action={state:np.asarray([[float(row[n]) for n in action_names] for row in sorted(rows,key=lambda x:float(x["weight"]))]) for state,rows in action_groups.items()}
    utility_groups:dict[str,list[dict[str,str]]]=defaultdict(list)
    for row in _csv(TIE/"advantage_dataset.csv"):utility_groups[row["state_id"]].append(row)
    utility={};tokens={};actionable=[];fixed=[]
    for state in ids:
        rows=sorted(utility_groups[state],key=lambda x:float(x["weight"]));values=np.asarray([float(r["utility"]) for r in rows]);advantages=np.asarray([float(r["advantage"]) for r in rows]);token=np.asarray([int(r["next_token_id"]) for r in rows])
        if not np.allclose(advantages,values-values[-1],atol=1e-12,rtol=0):raise ValueError(f"Tie-aware advantage mismatch: {state}")
        utility[state]=values;tokens[state]=token;actionable.append(len(set(token.tolist()))>=2);fixed.append(values[-1])
    if len(ids)!=250 or any(value.shape!=(5,) for value in utility.values()):raise ValueError("Incomplete rich-state inputs")
    return Data(ids,prompts,scalar,geometry,hidden,action,utility,tokens,np.asarray(actionable),np.asarray(fixed))


FAMILIES=("scalar","logit_k16","logit_k32","logit_k64","hidden_pca8","hidden_pca16","hidden_pca32","action_conditioned","combined")


def _state_matrix(data:Data,family:str,fit_indices:np.ndarray,apply_indices:np.ndarray)->np.ndarray:
    context=data.scalar[:,0:5]
    if family in {"scalar","action_conditioned"}:matrix=data.scalar if family=="scalar" else context
    elif family.startswith("logit_k"):
        k=int(family.removeprefix("logit_k"));matrix=np.concatenate((context,data.geometry[k]),axis=1)
    elif family.startswith("hidden_pca"):
        dim=int(family.removeprefix("hidden_pca"));projected,_=fit_train_only_pca(data.hidden[fit_indices],data.hidden[apply_indices],dim);return np.concatenate((context[apply_indices],projected),axis=1)
    elif family=="combined":
        projected,_=fit_train_only_pca(data.hidden[fit_indices],data.hidden[apply_indices],16);return np.concatenate((data.scalar[apply_indices],data.geometry[32][apply_indices],projected),axis=1)
    else:raise ValueError(family)
    return matrix[apply_indices]


def fit_train_only_pca(train_matrix:np.ndarray,apply_matrix:np.ndarray,dimension:int)->tuple[np.ndarray,dict[str,Any]]:
    if dimension>min(train_matrix.shape):raise ValueError("PCA dimension exceeds training-fold rank")
    pca=PCA(n_components=dimension,svd_solver="randomized",random_state=42).fit(train_matrix)
    return pca.transform(apply_matrix),{"fit_rows":len(train_matrix),"dimension":dimension,"training_mean":pca.mean_.copy()}


def prompt_folds(prompts:Sequence[str])->list[tuple[np.ndarray,np.ndarray,str]]:
    values=np.asarray(prompts);folds=[]
    for held in sorted(set(prompts)):
        train=np.flatnonzero(values!=held);test=np.flatnonzero(values==held)
        if set(values[train])&set(values[test]):raise AssertionError("Prompt leakage")
        folds.append((train,test,held))
    return folds


def _design(data:Data,family:str,state_indices:np.ndarray,state_matrix:np.ndarray)->np.ndarray:
    rows=[]
    include_action=family in {"action_conditioned","combined"}
    for local,index in enumerate(state_indices):
        state=state_matrix[local]
        for action_index,weight in enumerate(WEIGHTS):
            action=data.action[data.ids[index]][action_index] if include_action else np.empty(0)
            rows.append(np.concatenate((state,action,[weight,weight**2],state*weight,state*(weight**2))))
    return np.stack(rows)


def _targets(data:Data,indices:np.ndarray)->np.ndarray:
    return np.concatenate([data.utility[data.ids[i]]-data.utility[data.ids[i]][-1] for i in indices])


def _fit_predict(data:Data,family:str,train:np.ndarray,test:np.ndarray,alpha:float,fit_pca_indices:np.ndarray|None=None)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    fit_indices=train if fit_pca_indices is None else fit_pca_indices;train_state=_state_matrix(data,family,fit_indices,train);test_state=_state_matrix(data,family,fit_indices,test);xtrain=_design(data,family,train,train_state);xtest=_design(data,family,test,test_state);model=Pipeline((("scale",StandardScaler()),("ridge",Ridge(alpha=alpha)))).fit(xtrain,_targets(data,train));pred=model.predict(xtest).reshape(len(test),len(WEIGHTS));pred[:,-1]=0.;return pred,xtrain,xtest


def _inner_predictions(data:Data,family:str,outer_train:np.ndarray,alpha:float)->dict[int,np.ndarray]:
    actionable=outer_train[data.actionable[outer_train]];groups=np.asarray([data.prompts[i] for i in actionable]);unique=len(set(groups));splitter=GroupKFold(n_splits=min(3,unique));result={}
    for inner_train_pos,inner_test_pos in splitter.split(actionable,groups=groups):
        train=actionable[inner_train_pos];test=actionable[inner_test_pos];train_prompts={data.prompts[i] for i in train};pca_fit=np.asarray([i for i in outer_train if data.prompts[i] in train_prompts]);pred,_,_=_fit_predict(data,family,train,test,alpha,pca_fit)
        for index,row in zip(test,pred):result[int(index)]=row
    if set(result)!=set(actionable.tolist()):raise AssertionError("Inner grouped predictions incomplete")
    return result


def _select_hyperparameters(data:Data,family:str,outer_train:np.ndarray)->tuple[float,float,dict[str,Any]]:
    prediction_by_alpha={alpha:_inner_predictions(data,family,outer_train,alpha) for alpha in RIDGE_ALPHAS};mae={alpha:mean_absolute_error(np.concatenate([data.utility[data.ids[i]]-data.utility[data.ids[i]][-1] for i in sorted(pred)]),np.concatenate([pred[i] for i in sorted(pred)])) for alpha,pred in prediction_by_alpha.items()};chosen_alpha=min(RIDGE_ALPHAS,key=lambda value:(mae[value],value));pred=prediction_by_alpha[chosen_alpha];delta_scores={}
    for delta in DELTA_GRID:
        selected=[]
        for index in outer_train:
            if not data.actionable[index]:selected.append(data.fixed[index]);continue
            values=pred[int(index)];best=int(np.argmax(values));selected.append(data.utility[data.ids[index]][best] if values[best]>delta else data.fixed[index])
        delta_scores[delta]=fmean(selected)
    maximum=max(delta_scores.values());chosen_delta=max(delta for delta,value in delta_scores.items() if math.isclose(value,maximum,abs_tol=1e-12,rel_tol=0));return chosen_alpha,chosen_delta,{"ridge_mae":{str(k):v for k,v in mae.items()},"delta_training_mip":{str(k):v for k,v in delta_scores.items()}}


def _pairwise_logistic(data:Data,family:str,train:np.ndarray,test:np.ndarray)->tuple[int,int]:
    pca_fit=train;train_state=_state_matrix(data,family,pca_fit,train);test_state=_state_matrix(data,family,pca_fit,test);train_design=_design(data,family,train,train_state).reshape(len(train),5,-1);test_design=_design(data,family,test,test_state).reshape(len(test),5,-1);x=[];y=[]
    for local,index in enumerate(train):
        true=data.utility[data.ids[index]]
        for left in range(5):
            for right in range(left+1,5):
                if abs(true[left]-true[right])<=EPSILON:continue
                diff=train_design[local,left]-train_design[local,right];label=int(true[left]>true[right]);x.extend((diff,-diff));y.extend((label,1-label))
    if not x:return 0,0
    model=Pipeline((("scale",StandardScaler()),("logistic",LogisticRegression(C=1.0,max_iter=5000,random_state=42)))).fit(np.stack(x),np.asarray(y));correct=total=0
    for local,index in enumerate(test):
        true=data.utility[data.ids[index]]
        for left in range(5):
            for right in range(left+1,5):
                if abs(true[left]-true[right])<=EPSILON:continue
                predicted=int(model.predict((test_design[local,left]-test_design[local,right])[None,:])[0]);correct+=predicted==int(true[left]>true[right]);total+=1
    return correct,total


def evaluate_family(data:Data,family:str)->tuple[list[dict[str,Any]],dict[str,Any]]:
    output=[];logistic_correct=logistic_total=0
    for outer_train,outer_test,held in prompt_folds(data.prompts):
        alpha,delta,audit=_select_hyperparameters(data,family,outer_train);train_actionable=outer_train[data.actionable[outer_train]];pred,_,_=_fit_predict(data,family,train_actionable,outer_test,alpha,outer_train);c,n=_pairwise_logistic(data,family,train_actionable,outer_test[data.actionable[outer_test]]);logistic_correct+=c;logistic_total+=n
        for index,values in zip(outer_test,pred):
            true=data.utility[data.ids[index]];best=int(np.argmax(values));intervene=bool(data.actionable[index] and values[best]>delta);selected=best if intervene else 4;tied=oracle_set(dict(zip(WEIGHTS,true)),EPSILON)
            output.append({"feature_family":family,"prompt_id":data.prompts[index],"state_id":data.ids[index],"is_actionable":bool(data.actionable[index]),"selected_alpha":alpha,"selected_delta":delta,"selected_weight":WEIGHTS[selected],"intervened":intervene,"predicted_max_advantage":float(values[best]),"selected_utility":float(true[selected]),"fixed_w1_utility":float(true[-1]),"true_advantage":float(true[selected]-true[-1]),"oracle_utility":float(true.max()),"oracle_set_membership":WEIGHTS[selected] in tied,"predicted_advantages":json.dumps(values.tolist()),"training_prompt_count":len(set(data.prompts[i] for i in outer_train)),"held_out_prompt_excluded_from_tuning":held not in {data.prompts[i] for i in outer_train},"tuning_audit":json.dumps(audit,sort_keys=True)})
    actionable=[row for row in output if row["is_actionable"]];state_lookup={row["state_id"]:row for row in output};ridge_correct=ridge_total=0;spearman=[];undefined=0
    for row in actionable:
        true=data.utility[row["state_id"]];pred=np.asarray(json.loads(row["predicted_advantages"]));c,n,_=pairwise_accuracy(true,pred,EPSILON);ridge_correct+=c;ridge_total+=n;rho=defined_spearman(true,pred,EPSILON)
        if rho is None:undefined+=1
        else:spearman.append(rho)
    fixed=fmean(row["fixed_w1_utility"] for row in output);policy=fmean(row["selected_utility"] for row in output);oracle=fmean(row["oracle_utility"] for row in output);interventions=[row for row in output if row["intervened"]]
    target=[];prediction=[]
    for row in actionable:
        true=data.utility[row["state_id"]]-data.utility[row["state_id"]][-1];pred=np.asarray(json.loads(row["predicted_advantages"]));target.extend(true);prediction.extend(pred)
    summary={"feature_family":family,"states":len(output),"actionable_states":len(actionable),"advantage_mae":float(mean_absolute_error(target,prediction)),"informative_pair_count":ridge_total,"pairwise_accuracy":ridge_correct/ridge_total if ridge_total else None,"pairwise_logistic_accuracy":logistic_correct/logistic_total if logistic_total else None,"mean_defined_spearman":fmean(spearman) if spearman else None,"defined_spearman_states":len(spearman),"undefined_spearman_fraction":undefined/len(actionable),"oracle_set_accuracy":fmean(float(row["oracle_set_membership"]) for row in output),"policy_mip":policy,"fixed_w1_mip":fixed,"gain_over_fixed":policy-fixed,"oracle_mip":oracle,"oracle_gap_capture":(policy-fixed)/(oracle-fixed) if oracle>fixed else None,"intervention_rate":len(interventions)/len(output),"beneficial_intervention_rate":sum(row["true_advantage"]>EPSILON for row in interventions)/len(interventions) if interventions else None,"harmful_intervention_rate":sum(row["true_advantage"]<-EPSILON for row in interventions)/len(interventions) if interventions else None,"selected_weight_distribution":json.dumps(Counter(str(row["selected_weight"]) for row in output),sort_keys=True),"hyperparameters_selected_on_training_prompts_only":all(row["held_out_prompt_excluded_from_tuning"] for row in output)}
    return output,summary


def render_report(summaries:list[dict[str,Any]],best:dict[str,Any],verdict:str)->str:
    def number(value:Any,spec:str)->str:return "NA" if value is None else format(value,spec)
    lines=["# Rich-state advantage probe","",f"Verdict: **{verdict}**","","All results use leave-one-unique-prompt-out evaluation. PCA, Ridge alpha, and intervention delta are fitted/selected only from each outer fold's training prompts.","","| Family | Pairwise | Spearman | Policy MIP | Gain | Gap capture | Harmful | Intervention |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:lines.append(f"| {row['feature_family']} | {number(row['pairwise_accuracy'],'.4f')} | {number(row['mean_defined_spearman'],'.4f')} | {row['policy_mip']:.6f} | {row['gain_over_fixed']:.6f} | {number(row['oracle_gap_capture'],'.2%')} | {number(row['harmful_intervention_rate'],'.2%')} | {row['intervention_rate']:.2%} |")
    lines.extend(("",f"Best held-out policy: **{best['feature_family']}**, MIP `{best['policy_mip']:.9f}`.","","Scorer values are labels only and are absent from every inference feature. No backbone or PBLoRA parameter is trained."));return "\n".join(lines)+"\n"


def analyze()->dict[str,Any]:
    data=load_data();all_predictions=[];summaries=[]
    for family in FAMILIES:
        print(f"Evaluating {family}",flush=True);pred,summary=evaluate_family(data,family);all_predictions.extend(pred);summaries.append(summary)
    best=max(summaries,key=lambda row:row["policy_mip"]);supported=(best["pairwise_accuracy"] is not None and best["pairwise_accuracy"]>.5 and best["mean_defined_spearman"] is not None and best["mean_defined_spearman"]>0 and best["gain_over_fixed"]>0 and best["oracle_gap_capture"] is not None and best["oracle_gap_capture"]>0 and best["harmful_intervention_rate"] is not None and best["harmful_intervention_rate"]<.6061)
    strong=supported and best["pairwise_accuracy"]>=.60 and best["oracle_gap_capture"]>=.20
    verdict="RICH_STATE_ROUTING_SUPPORTED" if strong else ("RICH_STATE_ROUTING_WEAK" if supported else "ADAPTIVE_ROUTING_NO_GO")
    atomic_csv(OUT/"fold_predictions.csv",all_predictions);atomic_csv(OUT/"feature_family_summary.csv",summaries);best_rows=[row for row in all_predictions if row["feature_family"]==best["feature_family"]];atomic_csv(OUT/"best_policy_predictions.csv",best_rows);result={"status":"COMPLETE","verdict":verdict,"primary_subset":"64 actionable states","secondary_subset":"all 250 states","best_family":best,"families":summaries,"baselines":{"previous_pairwise":.405405,"previous_spearman":-.129514,"previous_selective_mip":.6028838391,"fixed_w1":.6068836809,"previous_harmful_interventions":.6061}};atomic_json(OUT/"summary.json",result);atomic_text(OUT/"report.md",render_report(summaries,best,verdict));return result
