"""CPU tests for tie-aware selective routing utilities."""

from __future__ import annotations

import numpy as np

from PARM_TARO.adaptive_parm.token_headroom.tie_aware import (
    assert_delta_training_only,
    defined_spearman,
    grouped_splits,
    informative_pairs,
    oracle_set,
    select_delta,
)


def test_exact_utility_ties_and_membership() -> None:
    tied=oracle_set({0.0:.7,.5:.7,1.0:.6},0.0)
    assert tied==(0.0,.5)
    assert 0.0 in tied and 1.0 not in tied


def test_near_ties_respect_epsilon() -> None:
    values={0.0:1.0,1.0:1.0-5e-7,.5:1.0-2e-5}
    assert oracle_set(values,1e-8)==(0.0,)
    assert oracle_set(values,1e-6)==(0.0,1.0)


def test_informative_pair_filtering() -> None:
    pairs=informative_pairs([1.0,1.0+5e-7,.5],1e-6)
    assert pairs==[(0,2),(1,2)]


def test_undefined_spearman_is_none_not_zero() -> None:
    assert defined_spearman([1.,1.,1.],[.1,.2,.3],1e-6) is None
    assert defined_spearman([1.,2.,3.],[.1,.2,.3],1e-6)==1.0


def test_grouped_split_has_no_prompt_overlap() -> None:
    ids=["a","a","b","b","c"]
    splits=grouped_splits(ids)
    assert len(splits)==3
    values=np.asarray(ids)
    for train,test,held in splits:
        assert held not in set(values[train])
        assert set(values[test])=={held}


def test_delta_selection_uses_only_supplied_training_rows() -> None:
    rows=[{"prompt_id":"train-a","actionable_probability":1.,"predicted_max_advantage":.02,"predicted_choice_true_utility":.8,"fixed_true_utility":.7},{"prompt_id":"train-b","actionable_probability":1.,"predicted_max_advantage":.001,"predicted_choice_true_utility":.6,"fixed_true_utility":.7}]
    delta,scores=select_delta(rows,(0.,.005,.05))
    assert delta==.005
    assert set(scores)=={"0.0","0.005","0.05"}
    assert assert_delta_training_only("held",rows)==["train-a","train-b"]


def test_delta_training_guard_rejects_held_prompt() -> None:
    rows=[{"prompt_id":"held"}]
    import pytest
    with pytest.raises(AssertionError,match="leaked"):
        assert_delta_training_only("held",rows)
