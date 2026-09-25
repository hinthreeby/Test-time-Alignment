"""CPU-only tests for the token-headroom experiment contract."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from PARM_TARO.adaptive_parm.fusion import adaptive_parm_logits
from PARM_TARO.adaptive_parm.token_headroom.analysis import _normalize
from PARM_TARO.adaptive_parm.token_headroom.config import (
    REFERENCE_WEIGHT,
    ROLLOUT_HORIZON,
    WEIGHTS,
)
from PARM_TARO.adaptive_parm.token_headroom.generation import select_state_positions
from PARM_TARO.adaptive_parm.token_headroom.run_token_headroom import ALPHAS, build_manifest
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import (
    bootstrap_import_paths,
    required_python_paths,
    subprocess_environment,
)


def test_relative_state_positions_are_unique_bounded_and_at_most_five() -> None:
    for length in range(1, 100):
        positions = select_state_positions(length)
        assert positions == sorted(set(positions))
        assert 1 <= len(positions) <= 5
        assert min(positions) >= 0
        assert max(positions) < length
    assert select_state_positions(0) == []
    assert select_state_positions(64) == [0, 16, 32, 47, 63]


def test_canonical_actions_use_only_current_base_guide_state() -> None:
    base = torch.log_softmax(torch.tensor([4.0, 1.0, -2.0]), dim=-1)
    guide = torch.log_softmax(torch.tensor([-1.0, 2.0, 3.0]), dim=-1)
    actions = [int(adaptive_parm_logits(base, guide, weight).argmax()) for weight in WEIGHTS]
    assert actions[0] == int(base.argmax())
    assert actions[-1] == int(guide.argmax())
    assert all(action in range(3) for action in actions)


def test_experiment_contract_is_one_step_then_reference() -> None:
    assert WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert REFERENCE_WEIGHT == 1.0
    assert ROLLOUT_HORIZON == 32


def test_phase09_normalization_is_clipped_linear() -> None:
    assert _normalize(-1.0, 0.0, 10.0) == 0.0
    assert _normalize(5.0, 0.0, 10.0) == 0.5
    assert _normalize(11.0, 0.0, 10.0) == 1.0


def test_manifest_is_group_complete_and_validation_only() -> None:
    rows = build_manifest(10)
    assert len(rows) == 50
    assert len({row["sample_id"] for row in rows}) == 10
    assert {tuple(row["requested_alpha"]) for row in rows} == set(ALPHAS)
    assert all(row["alpha_order"] == ["helpfulness", "harmlessness"] for row in rows)
    assert all("test" not in json.dumps(row).lower() for row in rows)


def test_structure_features_never_include_scorer_labels() -> None:
    from PARM_TARO.adaptive_parm.token_headroom.analysis import STATE_FEATURES

    forbidden = {"reward", "cost", "mip", "pcs", "helpfulness_raw", "harmlessness_raw"}
    assert forbidden.isdisjoint(STATE_FEATURES)
    assert np.isfinite(np.zeros(len(STATE_FEATURES))).all()


def test_relocated_router_path_bootstrap_and_child_environment() -> None:
    from PARM_TARO.adaptive_parm.token_headroom.config import ROOT

    expected = tuple(str(path) for path in required_python_paths(ROOT))
    assert expected[0].endswith("/Method/Router_Apdative")
    assert bootstrap_import_paths(ROOT) == expected
    assert tuple(sys.path[:3]) == expected
    environment = subprocess_environment(ROOT, {"PYTHONPATH": "/existing/one:/existing/two"})
    assert environment["PYTHONPATH"].split(os.pathsep) == [*expected, "/existing/one", "/existing/two"]
