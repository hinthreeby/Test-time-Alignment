"""CPU-only safety and determinism tests for the rich-state probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from PARM_TARO.adaptive_parm.rich_state_probe.analysis import Data,evaluate_family,fit_train_only_pca,prompt_folds
from PARM_TARO.adaptive_parm.rich_state_probe.features import (
    ACTION_MODEL_COLUMNS,INFERENCE_SCALAR_COLUMNS,action_feature_rows,atomic_npz,
    cache_valid,logit_geometry,prefix_hash,validate_prefix_record,
)


def test_prefix_only_record_and_future_leakage_rejection() -> None:
    row={"state_id":"s","prompt":"raw","prefix_token_ids":json.dumps([1,2,3]),"reference_position":"2"}
    assert validate_prefix_record(row)==[1,2,3]
    with pytest.raises(ValueError,match="future"):
        validate_prefix_record({**row,"future_token_ids":"[4]"})


def test_pca_is_fit_on_training_fold_only() -> None:
    train=np.asarray([[0.,0.],[2.,2.],[4.,4.]])
    held=np.asarray([[100.,-100.]])
    transformed,audit=fit_train_only_pca(train,held,1)
    assert transformed.shape==(1,1)
    np.testing.assert_allclose(audit["training_mean"],[2.,2.])
    assert not np.allclose(audit["training_mean"],np.vstack((train,held)).mean(0))


def test_prompt_folds_are_grouped() -> None:
    prompts=["a","a","b","b","c"]
    for train,test,held in prompt_folds(prompts):
        assert held not in {prompts[i] for i in train}
        assert {prompts[i] for i in test}=={held}


def test_canonical_fusion_action_features() -> None:
    base=np.log(np.asarray([.8,.15,.05]));parm=np.log(np.asarray([.05,.15,.8]));rows=action_feature_rows("s",base,parm)
    assert [row["weight"] for row in rows]==[0.,.25,.5,.75,1.]
    assert rows[0]["top1_token_id_audit_only"]==0
    assert rows[-1]["top1_token_id_audit_only"]==2
    assert rows[-1]["top1_differs_from_w1"]==0.


def test_no_scorer_values_are_inference_features() -> None:
    names={name.lower() for name in (*INFERENCE_SCALAR_COLUMNS,*ACTION_MODEL_COLUMNS)}
    assert all(term not in name for name in names for term in ("reward","cost","mip","utility","score"))
    assert "top1_token_id_audit_only" not in ACTION_MODEL_COLUMNS


def test_resume_cache_validation(tmp_path:Path) -> None:
    path=tmp_path/"s.npz";sha=prefix_hash([1,2,3]);arrays={"state_id":np.asarray("s"),"prefix_sha256":np.asarray(sha),"base_logprobs":np.asarray([-.1,-2.]),"parm_logprobs":np.asarray([-1.,-.2]),"base_hidden":np.asarray([1.,2.]),"parm_hidden":np.asarray([2.,3.])};atomic_npz(path,**arrays)
    assert cache_valid(path,"s",sha)
    assert not cache_valid(path,"s","wrong")


def test_feature_extraction_is_deterministic() -> None:
    base=np.log(np.asarray([.6,.2,.1,.1]));parm=np.log(np.asarray([.1,.2,.6,.1]))
    np.testing.assert_array_equal(logit_geometry(base,parm,2),logit_geometry(base,parm,2))
    assert action_feature_rows("s",base,parm)==action_feature_rows("s",base,parm)


def test_nested_grouped_scalar_analysis_smoke() -> None:
    rng=np.random.default_rng(42);ids=[];prompts=[];utility={};tokens={};action={};actionable=[];fixed=[]
    for prompt in range(10):
        for position in range(2):
            state=f"p{prompt}_s{position}";ids.append(state);prompts.append(f"p{prompt}")
            if position==0:
                values=np.asarray([.55+.001*prompt,.53,.51,.50,.50]);token=np.asarray([1,1,2,2,2]);is_actionable=True
            else:
                values=np.full(5,.5);token=np.full(5,3);is_actionable=False
            utility[state]=values;tokens[state]=token;action[state]=rng.normal(size=(5,len(ACTION_MODEL_COLUMNS)));actionable.append(is_actionable);fixed.append(values[-1])
    count=len(ids);data=Data(ids,prompts,rng.normal(size=(count,len(INFERENCE_SCALAR_COLUMNS))),{16:rng.normal(size=(count,192)),32:rng.normal(size=(count,384)),64:rng.normal(size=(count,768))},rng.normal(size=(count,65)),action,utility,tokens,np.asarray(actionable),np.asarray(fixed))
    predictions,summary=evaluate_family(data,"scalar")
    assert len(predictions)==20
    assert summary["states"]==20 and summary["actionable_states"]==10
    assert summary["hyperparameters_selected_on_training_prompts_only"] is True
