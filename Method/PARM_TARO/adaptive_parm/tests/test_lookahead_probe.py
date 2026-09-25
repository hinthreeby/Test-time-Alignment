"""CPU-only tests for selective lookahead scoring and analysis."""
from __future__ import annotations
import json
import numpy as np
from PARM_TARO.adaptive_parm.lookahead_probe.analysis import compute_estimates,defined_spearman,evaluate,informative_pair_accuracy,select_weight
from PARM_TARO.adaptive_parm.lookahead_probe.scoring import branch_key,continuation_tokens,horizon_means

def test_continuation_boundary_is_prefix_only_and_future_is_rollout_target()->None:
    state={"generated_token_count":"2"};rollout={"rollout_id":"r","response_token_ids":[10,11,20,21],"rollout_horizon":2,"candidate_next_token_id":20}
    assert continuation_tokens(rollout,state)==[20,21]

def test_branch_key_is_deterministic_and_alpha_sensitive()->None:
    assert branch_key([1],[2],[1.,0.])==branch_key([1],[2],[1.,0.])
    assert branch_key([1],[2],[1.,0.])!=branch_key([1],[2],[0.,1.])

def test_horizon_scores_use_available_prefix_and_length_normalization()->None:
    values=[-1.,-3.,-5.];means=horizon_means(values)
    assert means[1]==(1,-1.) and means[2]==(2,-2.) and means[32]==(3,-3.)

def test_informative_pairs_ignore_utility_ties()->None:
    count,accuracy=informative_pair_accuracy([1.,1.,2.],[0.,9.,10.])
    assert count==2 and accuracy==1.

def test_undefined_spearman_and_conservative_score_tie()->None:
    assert defined_spearman([1.,1.,1.],[1.,2.,3.]) is None
    assert select_weight([0.,.5,1.],[2.,1.,2.])==2

def test_compute_estimate_selective_branching()->None:
    rows=compute_estimates(.256);row=next(x for x in rows if x["candidate_set"]=="W3" and x["horizon"]==4)
    assert np.isclose(row["estimated_relative_forward_token_multiplier"],1+.256*2*4)

def test_direct_policy_evaluation_prefers_correct_internal_ranking()->None:
    rows=[]
    for weight,utility,score in zip((0.,.25,.5,.75,1.),(.2,.3,.4,.5,.45),(0.,1.,2.,3.,2.5)):
        rows.append({"state_id":"s","sample_id":"p","weight":weight,"mip":utility,"parm_score":score,"base_score":-score,"ratio_score":2*score,"horizon":8,"is_actionable":True})
    summary,predictions=evaluate(rows,"parm",8,"actionable")
    assert summary["policy_utility"]==.5
    assert summary["gain_over_fixed"]>0
    assert predictions[0]["selected_weight"]==.75
