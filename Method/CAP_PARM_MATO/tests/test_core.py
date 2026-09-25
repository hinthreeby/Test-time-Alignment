from __future__ import annotations

import torch
import torch.nn.functional as F

from src.controllers.dynamic_preference import DynamicPreferenceController
from src.decoding.fusion import fuse_scores
from src.models.token_router import TokenRouter
from src.models.objective_tracker import ObjectiveTracker
from src.models.parm_adapter import preference_to_pblora


def test_fusion_endpoints_and_product():
    base = torch.tensor([[1.0, 2.0, -1.0]])
    parm = torch.tensor([[0.0, -2.0, 3.0]])
    assert torch.equal(fuse_scores(base, parm, 0.0), base)
    assert torch.equal(fuse_scores(base, parm, 1.0), parm)
    assert torch.allclose(fuse_scores(base, parm, 0.5), (base + parm) / 2)
    expected = F.log_softmax(base, -1) + 0.5 * F.log_softmax(parm, -1)
    assert torch.allclose(fuse_scores(base, parm, 0.5, "parm_product"), expected)


def test_dynamic_preference_stays_in_simplex_and_prioritizes_low_reward():
    controller = DynamicPreferenceController(
        temperature=0.5, smoothing=1.0, kl_budget=0.1
    )
    alpha, info = controller.update([0.5, 0.5], [0.5, 0.5], [1.0, -1.0])
    assert all(value >= 0 for value in alpha)
    assert abs(sum(alpha) - 1.0) < 1e-8
    assert alpha[1] > alpha[0]
    assert info.kl_to_user <= 0.1 + 1e-8


def test_equal_rewards_keep_user_preference():
    controller = DynamicPreferenceController(smoothing=1.0)
    alpha, _ = controller.update([0.7, 0.3], [0.7, 0.3], [2.0, 2.0])
    assert torch.allclose(torch.tensor(alpha), torch.tensor([0.7, 0.3]), atol=1e-6)


def test_router_feature_schemas_and_save_load(tmp_path):
    logits = torch.randn(3, 101)
    alpha = torch.tensor([0.6, 0.4])
    deficits = torch.tensor([0.1, -0.1])
    for schema, expected in (("full_v1", 40), ("compact_v1", 10)):
        router = TokenRouter(top_k=32 if schema == "full_v1" else 10,
                             feature_schema=schema).eval()
        features = router.build_features(logits, alpha, deficits, 3, 20)
        assert features.shape == (3, expected)
        before = router(logits, alpha, deficits, 3, 20).gate_prob
        path = tmp_path / f"{schema}.pt"
        router.save(str(path), metadata={"fixture": True})
        loaded = TokenRouter.load(str(path)).eval()
        after = loaded(logits, alpha, deficits, 3, 20).gate_prob
        assert loaded.feature_schema == schema
        assert torch.allclose(before, after)


def test_router_rejects_wrong_feature_dimension():
    router = TokenRouter(feature_schema="compact_v1", top_k=10)
    try:
        router.gate_logits_from_features(torch.zeros(2, 9))
    except ValueError as error:
        assert "dimension 10" in str(error)
    else:
        raise AssertionError("wrong feature dimension was accepted")


def test_objective_tracker_normalization_and_save_load(tmp_path):
    tracker = ObjectiveTracker(top_k=4, hidden_dim=16, dropout=0.0).eval()
    tracker.set_target_normalization(torch.tensor([-3.0, 2.0]), torch.tensor([2.0, 4.0]))
    features = torch.randn(3, tracker.input_dim)
    before = tracker.forward_features(features)
    assert before.rewards.shape == (3, 2)
    assert ((before.uncertainty >= 0) & (before.uncertainty <= 1)).all()
    path = tmp_path / "tracker.pt"
    tracker.save(str(path), metadata={"fixture": True})
    loaded = ObjectiveTracker.load(str(path)).eval()
    after = loaded.forward_features(features)
    assert torch.allclose(before.rewards, after.rewards)
    assert torch.allclose(before.uncertainty, after.uncertainty)


def test_public_preference_is_mapped_to_pblora_order():
    assert preference_to_pblora([0.8, 0.2]) == [0.2, 0.8]
