from __future__ import annotations

import inspect
import unittest

import torch
import torch.nn.functional as functional
from torch import nn

from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.config import TARORouterConfig
from router_v2.model import TAROTokenRouter
from router_v2.objective import bernoulli_entropy, compute_taro_objective


def make_config(**overrides: object) -> TARORouterConfig:
    values: dict[str, object] = {
        "vocab_size": 17,
        "top_k": 3,
        "token_embedding_dim": 4,
        "hidden_dim": 128,
    }
    values.update(overrides)
    return TARORouterConfig(**values)


def make_topk_batch() -> TAROTopKBatch:
    return TAROTopKBatch(
        base_token_ids=torch.tensor([[1, 3, 5], [2, 4, 6]]),
        base_logits=torch.tensor([[4.0, 3.0, 2.0], [5.0, 3.0, 1.0]]),
        reward_token_ids=torch.tensor([[7, 9, 11], [8, 10, 12]]),
        reward_logits=torch.tensor([[8.0, 6.0, 1.0], [7.0, 4.0, 2.0]]),
    )


class CandidateSelectionTests(unittest.TestCase):
    def test_base_and_reward_topk_are_independent(self) -> None:
        base_logits = torch.tensor([[9.0, 8.0, 0.0, -1.0, -2.0, -3.0]])
        reward_logits = torch.tensor([[-3.0, -2.0, -1.0, 0.0, 8.0, 9.0]])

        topk = select_taro_topk(base_logits, reward_logits, top_k=2)

        self.assertEqual(topk.base_token_ids.tolist(), [[0, 1]])
        self.assertEqual(topk.reward_token_ids.tolist(), [[5, 4]])
        self.assertEqual(topk.base_logits.tolist(), [[9.0, 8.0]])
        self.assertEqual(topk.reward_logits.tolist(), [[9.0, 8.0]])

    def test_different_topk_token_sets_are_supported_by_model(self) -> None:
        config = make_config(top_k=2)
        model = TAROTokenRouter(config)
        base_logits = torch.tensor(
            [[9.0, 8.0, 0.0, -1.0, -2.0, -3.0] + [-4.0] * 11]
        )
        reward_logits = torch.tensor(
            [[-3.0, -2.0, -1.0, 0.0, 8.0, 9.0] + [-4.0] * 11]
        )

        output = model.forward_from_full_logits(base_logits, reward_logits)

        self.assertEqual(output.topk.base_token_ids.tolist(), [[0, 1]])
        self.assertEqual(output.topk.reward_token_ids.tolist(), [[5, 4]])
        self.assertEqual(output.alpha.shape, (1, 1))

    def test_gold_is_not_inserted_into_router_features(self) -> None:
        base_logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, -20.0]])
        reward_logits = torch.tensor([[6.0, 7.0, 8.0, 9.0, -30.0]])
        gold_token_id = 4

        topk = select_taro_topk(base_logits, reward_logits, top_k=2)

        self.assertNotIn(gold_token_id, topk.base_token_ids[0].tolist())
        self.assertNotIn(gold_token_id, topk.reward_token_ids[0].tolist())

    def test_topk_construction_requires_no_gold_information(self) -> None:
        signature = inspect.signature(select_taro_topk)
        self.assertNotIn("gold_token_ids", signature.parameters)
        base_logits = torch.randn(2, 11)
        reward_logits = torch.randn(2, 11)

        topk = select_taro_topk(base_logits, reward_logits, top_k=3)

        self.assertEqual(topk.base_token_ids.shape, (2, 3))
        self.assertEqual(topk.reward_token_ids.shape, (2, 3))


class FeatureAndArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.config = make_config()
        self.model = TAROTokenRouter(self.config)

    def test_exact_feature_dimension(self) -> None:
        topk = make_topk_batch()
        features = self.model.build_topk_features(topk)
        expected_dim = 2 * self.config.top_k * (
            self.config.token_embedding_dim + 1
        )

        self.assertEqual(features.shape, (2, expected_dim))
        self.assertEqual(self.model.feature_dim, expected_dim)
        self.assertEqual(self.model.gate_mlp[0].in_features, expected_dim)

    def test_candidate_ordering_is_preserved(self) -> None:
        topk = make_topk_batch()
        with torch.no_grad():
            for token_id in range(self.config.vocab_size):
                self.model.token_embedding.weight[token_id].fill_(float(token_id))
        features = self.model.build_topk_features(topk)
        width = self.config.token_embedding_dim + 1
        base_pairs = features[:, : self.config.top_k * width].reshape(
            2,
            self.config.top_k,
            width,
        )
        reward_pairs = features[:, self.config.top_k * width :].reshape(
            2,
            self.config.top_k,
            width,
        )

        torch.testing.assert_close(base_pairs[..., 0], topk.base_logits)
        torch.testing.assert_close(reward_pairs[..., 0], topk.reward_logits)
        for index in range(self.config.top_k):
            expected_base_embedding = topk.base_token_ids[:, index].float()
            expected_reward_embedding = topk.reward_token_ids[:, index].float()
            torch.testing.assert_close(
                base_pairs[:, index, 1],
                expected_base_embedding,
            )
            torch.testing.assert_close(
                reward_pairs[:, index, 1],
                expected_reward_embedding,
            )

    def test_consistent_pair_permutation_reorders_feature_blocks(self) -> None:
        topk = make_topk_batch()
        permutation = torch.tensor([2, 0, 1])
        permuted = TAROTopKBatch(
            base_token_ids=topk.base_token_ids.index_select(-1, permutation),
            base_logits=topk.base_logits.index_select(-1, permutation),
            reward_token_ids=topk.reward_token_ids.index_select(-1, permutation),
            reward_logits=topk.reward_logits.index_select(-1, permutation),
        )
        original_features = self.model.build_topk_features(topk)
        permuted_features = self.model.build_topk_features(permuted)
        width = self.config.token_embedding_dim + 1

        original_pairs = original_features.reshape(2, 2, self.config.top_k, width)
        permuted_pairs = permuted_features.reshape(2, 2, self.config.top_k, width)

        torch.testing.assert_close(
            permuted_pairs,
            original_pairs.index_select(-2, permutation),
        )

    def test_token_identity_affects_router_output(self) -> None:
        config = make_config(top_k=2, token_embedding_dim=2)
        model = TAROTokenRouter(config)
        with torch.no_grad():
            model.token_embedding.weight.zero_()
            model.token_embedding.weight[1, 0] = 2.0
            model.token_embedding.weight[3, 0] = -2.0
            first_layer = model.gate_mlp[0]
            final_layer = model.gate_mlp[2]
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, 1] = 1.0
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            final_layer.weight[0, 0] = 1.0
        common = {
            "base_logits": torch.zeros(1, 2),
            "reward_token_ids": torch.tensor([[5, 6]]),
            "reward_logits": torch.zeros(1, 2),
        }
        first = TAROTopKBatch(
            base_token_ids=torch.tensor([[1, 2]]),
            **common,
        )
        second = TAROTopKBatch(
            base_token_ids=torch.tensor([[3, 2]]),
            **common,
        )

        first_alpha = model.predict_alpha(first)
        second_alpha = model.predict_alpha(second)

        self.assertFalse(torch.equal(first_alpha, second_alpha))

    def test_mlp_is_hidden_128_tanh_scalar(self) -> None:
        self.assertEqual(self.model.gate_mlp[0].in_features, self.model.feature_dim)
        self.assertEqual(self.model.gate_mlp[0].out_features, 128)
        self.assertIsInstance(self.model.gate_mlp[1], nn.Tanh)
        self.assertEqual(self.model.gate_mlp[2].in_features, 128)
        self.assertEqual(self.model.gate_mlp[2].out_features, 1)
        self.assertFalse(any(isinstance(module, nn.GELU) for module in self.model.modules()))

    def test_gate_shape_and_strict_range(self) -> None:
        alpha = self.model.predict_alpha(make_topk_batch())

        self.assertEqual(alpha.shape, (2, 1))
        self.assertTrue(bool((alpha > 0.0).all()))
        self.assertTrue(bool((alpha < 1.0).all()))


class RoutingAndObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4321)
        self.config = make_config()
        self.model = TAROTokenRouter(self.config)

    def test_alpha_near_zero_approaches_base_logits(self) -> None:
        base = torch.tensor([[1.0, 2.0, 3.0]])
        reward = torch.tensor([[7.0, 8.0, 9.0]])
        alpha = torch.tensor([[1e-7]])

        guided = TAROTokenRouter.route_logits(alpha, base, reward)

        torch.testing.assert_close(guided, base, atol=1e-6, rtol=0)

    def test_alpha_near_one_approaches_reward_logits(self) -> None:
        base = torch.tensor([[1.0, 2.0, 3.0]])
        reward = torch.tensor([[7.0, 8.0, 9.0]])
        alpha = torch.tensor([[1.0 - 1e-7]])

        guided = TAROTokenRouter.route_logits(alpha, base, reward)

        torch.testing.assert_close(guided, reward, atol=1e-6, rtol=0)

    def test_alpha_half_is_arithmetic_midpoint(self) -> None:
        base = torch.tensor([[1.0, 2.0, 3.0]])
        reward = torch.tensor([[7.0, 8.0, 9.0]])
        alpha = torch.tensor([[0.5]])

        guided = TAROTokenRouter.route_logits(alpha, base, reward)

        torch.testing.assert_close(guided, (base + reward) / 2.0)

    def test_full_vocabulary_nll_matches_cross_entropy_without_gold_topk(self) -> None:
        base_logits = torch.full((1, self.config.vocab_size), -5.0)
        reward_logits = torch.full((1, self.config.vocab_size), -6.0)
        base_logits[0, :3] = torch.tensor([9.0, 8.0, 7.0])
        reward_logits[0, 3:6] = torch.tensor([7.0, 8.0, 9.0])
        gold_token_ids = torch.tensor([16])

        output = self.model.forward_from_full_logits(base_logits, reward_logits)
        objective = compute_taro_objective(output, gold_token_ids, self.config)
        expected = functional.cross_entropy(output.guided_logits, gold_token_ids)

        self.assertNotIn(16, output.topk.base_token_ids[0].tolist())
        self.assertNotIn(16, output.topk.reward_token_ids[0].tolist())
        torch.testing.assert_close(objective.nll_loss, expected)
        torch.testing.assert_close(objective.total_loss, expected)

    def test_entropy_formula_is_correct(self) -> None:
        alpha = torch.tensor([[0.2], [0.75]])
        expected = -alpha * alpha.log() - (1.0 - alpha) * (1.0 - alpha).log()

        actual = bernoulli_entropy(alpha, eps=1e-6)

        torch.testing.assert_close(actual, expected)

    def test_entropy_mode_off_has_zero_contribution(self) -> None:
        base_logits = torch.randn(2, self.config.vocab_size)
        reward_logits = torch.randn(2, self.config.vocab_size)
        gold_token_ids = torch.tensor([3, 11])
        output = self.model.forward_from_full_logits(base_logits, reward_logits)

        objective = compute_taro_objective(output, gold_token_ids, self.config)

        torch.testing.assert_close(
            objective.entropy_contribution,
            torch.zeros_like(objective.entropy_contribution),
        )
        torch.testing.assert_close(objective.total_loss, objective.nll_loss)

    def test_entropy_mode_on_adds_configured_term(self) -> None:
        config = make_config(
            mode="taro_topk_nll_entropy",
            entropy_weight=0.25,
        )
        model = TAROTokenRouter(config)
        base_logits = torch.randn(2, config.vocab_size)
        reward_logits = torch.randn(2, config.vocab_size)
        gold_token_ids = torch.tensor([3, 11])
        output = model.forward_from_full_logits(base_logits, reward_logits)

        objective = compute_taro_objective(output, gold_token_ids, config)
        expected_entropy = bernoulli_entropy(
            output.alpha,
            eps=config.alpha_eps,
        ).mean()

        torch.testing.assert_close(objective.entropy, expected_entropy)
        torch.testing.assert_close(
            objective.entropy_contribution,
            0.25 * expected_entropy,
        )
        torch.testing.assert_close(
            objective.total_loss,
            objective.nll_loss + 0.25 * expected_entropy,
        )

    def test_gradients_only_flow_to_router(self) -> None:
        base_logits = torch.randn(
            2,
            self.config.vocab_size,
            requires_grad=True,
        )
        reward_logits = torch.randn(
            2,
            self.config.vocab_size,
            requires_grad=True,
        )
        gold_token_ids = torch.tensor([1, 12])
        output = self.model.forward_from_full_logits(base_logits, reward_logits)
        objective = compute_taro_objective(output, gold_token_ids, self.config)

        objective.total_loss.backward()

        self.assertIsNone(base_logits.grad)
        self.assertIsNone(reward_logits.grad)
        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        self.assertGreater(
            self.model.token_embedding.weight.grad.abs().sum().item(),
            0.0,
        )
        for layer_index in (0, 2):
            layer = self.model.gate_mlp[layer_index]
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
