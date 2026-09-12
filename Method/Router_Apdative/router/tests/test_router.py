from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from router.checkpoint import load_router_checkpoint, save_router_checkpoint
from router.features import build_router_features, compute_guided_scores, standardize_candidate_scores
from router.model import RADTokenRouter


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def tensors(self, batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        base_logits = torch.randn(batch_size, 20)
        reward_scores = torch.randn(batch_size, 20)
        return base_logits, reward_scores

    def test_architecture_is_exactly_40_64_1_with_tanh(self) -> None:
        router = RADTokenRouter(beta_max=30.0)
        self.assertIsInstance(router.mlp[0], nn.Linear)
        self.assertEqual(router.mlp[0].in_features, 40)
        self.assertEqual(router.mlp[0].out_features, 64)
        self.assertIsInstance(router.mlp[1], nn.Tanh)
        self.assertIsInstance(router.mlp[2], nn.Linear)
        self.assertEqual(router.mlp[2].in_features, 64)
        self.assertEqual(router.mlp[2].out_features, 1)

    def test_shapes_and_beta_range(self) -> None:
        router = RADTokenRouter(beta_max=12.5)
        base_logits, reward_scores = self.tensors(batch_size=3)
        beta, gate = router(base_logits, reward_scores)
        self.assertEqual(tuple(beta.shape), (3, 1))
        self.assertEqual(tuple(gate.shape), (3, 1))
        self.assertTrue(bool((gate >= 0).all() and (gate <= 1).all()))
        self.assertTrue(bool((beta >= 0).all() and (beta <= 12.5).all()))

    def test_shape_assertions_are_strict(self) -> None:
        router = RADTokenRouter(beta_max=10.0)
        with self.assertRaises(AssertionError):
            router(torch.randn(2, 19), torch.randn(2, 19))
        with self.assertRaises(AssertionError):
            router(torch.randn(2, 20), torch.randn(2, 20, 1))
        with self.assertRaises(AssertionError):
            build_router_features(torch.randn(2, 20), torch.randn(3, 20))

    def test_shift_invariance_of_normalized_features_and_router_output(self) -> None:
        router = RADTokenRouter(beta_max=10.0)
        base_logits, reward_scores = self.tensors(batch_size=5)
        shifted_base = base_logits + 17.0
        shifted_reward = reward_scores - 9.0
        self.assertTrue(torch.allclose(
            standardize_candidate_scores(base_logits),
            standardize_candidate_scores(shifted_base),
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            standardize_candidate_scores(reward_scores),
            standardize_candidate_scores(shifted_reward),
            atol=1e-5,
        ))
        beta1, gate1 = router(base_logits, reward_scores)
        beta2, gate2 = router(shifted_base, shifted_reward)
        self.assertTrue(torch.allclose(beta1, beta2, atol=1e-5))
        self.assertTrue(torch.allclose(gate1, gate2, atol=1e-5))

    def test_constant_inputs_normalize_to_zero_without_nan(self) -> None:
        values = torch.full((2, 20), 3.14)
        normalized = standardize_candidate_scores(values)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6))

    def test_guided_scores_use_raw_values_separate_from_normalization(self) -> None:
        base_logits, rad_reward_scores = self.tensors(batch_size=2)
        beta = torch.tensor([[2.0], [0.5]])
        guided = compute_guided_scores(base_logits, rad_reward_scores, beta)
        expected = base_logits + beta * rad_reward_scores
        self.assertTrue(torch.allclose(guided, expected))
        normalized = build_router_features(base_logits, rad_reward_scores)
        self.assertEqual(tuple(normalized.shape), (2, 40))

    def test_gradients_reach_router_parameters_only_and_optimizer_excludes_models(self) -> None:
        router = RADTokenRouter(beta_max=10.0)
        frozen_base_lm = nn.Linear(3, 3)
        frozen_reward_model = nn.Linear(3, 1)
        for model in [frozen_base_lm, frozen_reward_model]:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
                parameter.grad = None
        optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        router_param_ids = {id(parameter) for parameter in router.parameters()}
        frozen_param_ids = {
            id(parameter)
            for model in [frozen_base_lm, frozen_reward_model]
            for parameter in model.parameters()
        }
        self.assertEqual(optimizer_param_ids, router_param_ids)
        self.assertTrue(optimizer_param_ids.isdisjoint(frozen_param_ids))

        base_logits = torch.randn(3, 20, requires_grad=True)
        rad_reward_scores = torch.randn(3, 20, requires_grad=True)
        beta, _gate = router(base_logits, rad_reward_scores)
        guided = compute_guided_scores(base_logits, rad_reward_scores, beta)
        loss = guided.mean()
        loss.backward()
        parameter_grads = [parameter.grad for parameter in router.parameters()]
        self.assertTrue(all(grad is not None for grad in parameter_grads))
        self.assertTrue(any(float(grad.abs().sum()) > 0.0 for grad in parameter_grads))
        self.assertIsNone(base_logits.grad)
        self.assertIsNone(rad_reward_scores.grad)
        self.assertTrue(all(parameter.grad is None for parameter in frozen_base_lm.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_reward_model.parameters()))

    def test_checkpoint_round_trip_preserves_outputs(self) -> None:
        router = RADTokenRouter(beta_max=15.0, beta_init=3.0)
        base_logits, reward_scores = self.tensors(batch_size=2)
        beta_before, gate_before = router(base_logits, reward_scores)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.pt"
            save_router_checkpoint(router, path)
            loaded = load_router_checkpoint(path)
        beta_after, gate_after = loaded(base_logits, reward_scores)
        self.assertTrue(torch.equal(beta_before, beta_after))
        self.assertTrue(torch.equal(gate_before, gate_after))
        self.assertEqual(loaded.config.beta_max, 15.0)
        self.assertEqual(loaded.config.beta_init, 3.0)
        self.assertEqual(loaded.config.top_k, 20)
        self.assertEqual(loaded.config.hidden_dim, 64)
        self.assertEqual(loaded.config.eps, 1e-6)

    def test_beta_init_sets_initial_bias(self) -> None:
        router = RADTokenRouter(beta_max=20.0, beta_init=5.0)
        final_weight = router.mlp[-1].weight.detach()
        final_bias = router.mlp[-1].bias.detach()
        expected_gate = torch.sigmoid(final_bias)
        self.assertTrue(torch.equal(final_weight, torch.zeros_like(final_weight)))
        self.assertTrue(torch.allclose(expected_gate, torch.full_like(expected_gate, 0.25)))

    def test_beta_init_produces_requested_initial_beta(self) -> None:
        router = RADTokenRouter(beta_max=20.0, beta_init=5.0)
        base_logits, reward_scores = self.tensors(batch_size=8)

        beta, gate = router(base_logits, reward_scores)

        self.assertTrue(
            torch.allclose(beta, torch.full_like(beta, 5.0), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(gate, torch.full_like(gate, 0.25), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
