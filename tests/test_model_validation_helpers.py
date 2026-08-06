from __future__ import annotations

import unittest

import torch

from scripts.validate_models import (
    assert_all_frozen,
    dummy_router_optimization_step,
    freeze_model,
    frozen_parameter_stats,
    parameter_checksum,
    tensor_stats,
    validate_frozen_models_survive_router_step,
)


class ModelValidationHelperTests(unittest.TestCase):
    def test_freeze_model_disables_all_gradients_and_eval_mode(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
        model.train()
        stats = freeze_model(model)
        self.assertTrue(stats.all_frozen)
        self.assertFalse(model.training)
        self.assertEqual(stats.trainable_parameters, 0)
        assert_all_frozen(model, "toy")

    def test_frozen_model_stays_frozen_after_dummy_router_step(self) -> None:
        model = torch.nn.Linear(2, 2)
        before = freeze_model(model)
        optimizer_contains_only_router = dummy_router_optimization_step()
        after = frozen_parameter_stats(model)
        self.assertTrue(optimizer_contains_only_router)
        self.assertEqual(before, after)
        assert_all_frozen(model, "toy")

    def test_frozen_base_and_reward_have_no_grads_and_stable_checksums(self) -> None:
        base_model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh())
        reward_model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        freeze_model(base_model)
        freeze_model(reward_model)
        base_checksum_before = parameter_checksum(base_model)
        reward_checksum_before = parameter_checksum(reward_model)

        probe = validate_frozen_models_survive_router_step(base_model, reward_model)

        self.assertTrue(probe.optimizer_contains_only_router)
        self.assertTrue(probe.base_requires_grad_all_false)
        self.assertTrue(probe.reward_requires_grad_all_false)
        self.assertTrue(probe.base_grads_all_none)
        self.assertTrue(probe.reward_grads_all_none)
        self.assertTrue(probe.base_checksum_unchanged)
        self.assertTrue(probe.reward_checksum_unchanged)
        self.assertEqual(probe.base_checksum_before, base_checksum_before)
        self.assertEqual(probe.reward_checksum_before, reward_checksum_before)
        self.assertEqual(probe.base_checksum_after, base_checksum_before)
        self.assertEqual(probe.reward_checksum_after, reward_checksum_before)

    def test_tensor_stats(self) -> None:
        stats = tensor_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertGreater(stats["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
