from __future__ import annotations

import numpy as np
import pytest
import torch

from PARM_TARO.adaptive_parm.prompt_router.feature_extraction import compute_prompt_features
from PARM_TARO.adaptive_parm.prompt_router.train import _pairwise_accuracy, utility_design


def test_prompt_features_are_finite_and_pre_generation_local() -> None:
    generator = torch.Generator().manual_seed(7)
    a = torch.log_softmax(torch.randn(101, generator=generator), -1)
    b = torch.log_softmax(torch.randn(101, generator=generator), -1)
    result = compute_prompt_features(a, b, prompt_token_length=17, top_k=10)
    assert result["prompt_token_length"] == 17
    assert all(np.isfinite(value) for value in result.values())
    assert 0 <= result["topk_overlap"] <= 1
    assert result["top1_disagreement"] in (0.0, 1.0)
    assert result["js_divergence"] >= -1e-7
    assert result["symmetric_kl"] >= -1e-7


def test_identical_policies_have_zero_disagreement() -> None:
    a = torch.log_softmax(torch.arange(20, dtype=torch.float32), -1)
    result = compute_prompt_features(a, a.clone(), prompt_token_length=3, top_k=5)
    assert result["top1_disagreement"] == 0
    assert result["topk_overlap"] == 1
    assert result["js_divergence"] == pytest.approx(0.0, abs=1e-7)
    assert result["symmetric_kl"] == pytest.approx(0.0, abs=1e-7)


def test_utility_design_contains_action_interactions() -> None:
    state = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    weight = np.asarray([0.25, 0.5])
    design = utility_design(state, weight)
    assert design.shape == (2, 8)
    np.testing.assert_allclose(design[:, 4:6], state * weight[:, None])


def test_pairwise_ranking_accuracy() -> None:
    true = np.asarray([0.1, 0.3, 0.2])
    assert _pairwise_accuracy(true, true) == 1.0
    assert _pairwise_accuracy(true, -true) == 0.0
