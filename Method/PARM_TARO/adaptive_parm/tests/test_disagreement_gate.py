"""CPU-only tests for disagreement-gated PARM."""
from __future__ import annotations
import numpy as np
from PARM_TARO.adaptive_parm.disagreement_gate.analysis import bootstrap_prompt_difference,normalize
from PARM_TARO.adaptive_parm.disagreement_gate.config import ALPHAS,METHODS
from PARM_TARO.adaptive_parm.disagreement_gate.manifest import build_manifest,select_cases
from PARM_TARO.adaptive_parm.disagreement_gate.generation import disagreement_weight_and_token

def hard_gate(base:np.ndarray,guide:np.ndarray)->tuple[float,int]:
    import torch
    weight,token=disagreement_weight_and_token(torch.tensor(base)[None],torch.tensor(guide)[None]);return weight,int(token.item())
def test_gate_rule_and_canonical_endpoints()->None:
    base=np.asarray([3.,1.]);guide=np.asarray([2.,0.]);assert hard_gate(base,guide)==(1.,0)
    guide=np.asarray([0.,4.]);assert hard_gate(base,guide)==(0.,0)
def test_greedy_hard_gate_is_token_equivalent_to_base()->None:
    rng=np.random.default_rng(42)
    for _ in range(100):
        base=rng.normal(size=31);guide=rng.normal(size=31);_,token=hard_gate(base,guide);assert token==int(base.argmax())
def test_manifest_is_fresh_deterministic_and_expandable()->None:
    rows=build_manifest();assert len(rows)==1000 and len({r['sample_id'] for r in rows})==200
    diagnostic={r['sample_id'] for r in __import__('PARM_TARO.adaptive_parm.token_headroom.io',fromlist=['read_jsonl']).read_jsonl(__import__('PARM_TARO.adaptive_parm.disagreement_gate.config',fromlist=['DIAGNOSTIC_MANIFEST']).DIAGNOSTIC_MANIFEST)}
    assert not ({r['sample_id'] for r in rows}&diagnostic);assert len(select_cases(rows,50))==250 and select_cases(rows,50)==rows[:250]
def test_protocol_normalization_clips()->None:
    assert normalize(-1.,0.,10.)==0. and normalize(11.,0.,10.)==1. and normalize(5.,0.,10.)==.5
def test_prompt_cluster_bootstrap_not_alpha_rows()->None:
    rows=[]
    for prompt in ('p0','p1'):
        for alpha in ALPHAS:
            rows.extend(({'sample_id':prompt,'method':'parm_fixed','mip':.5},{'sample_id':prompt,'method':'disagreement_gate','mip':.6}))
    low,high=bootstrap_prompt_difference(rows,reps=100);assert np.isclose(low,.1) and np.isclose(high,.1)
def test_exact_method_set_has_no_tuned_margin_gate()->None:
    assert METHODS==('base','parm_fixed','midpoint','disagreement_gate')
