from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from router_v2.candidates import TAROTopKBatch
from router_v2.checkpoint import save_taro_checkpoint
from router_v2.config import TARORouterConfig
from router_v2.model import TAROTokenRouter
from router_v2.smart_checkpoint import (
    SMART_CHECKPOINT_FORMAT,
    build_smart_checkpoint_payload,
    load_smart_checkpoint,
    save_smart_checkpoint,
    validate_smart_checkpoint_payload,
)
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import (
    SMART_CONFIDENCE_FEATURE_NAMES,
    SmartRouterBatch,
    SmartTokenRouter,
)
from router_v2.scripts.smoke_smart_router import DEFAULT_CONFIG


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def smart_config(**overrides: object) -> SmartRouterConfig:
    values: dict[str, object] = {
        "variant": "custom",
        "vocab_size": 31,
        "top_k": 3,
        "token_embedding_dim": 4,
        "candidate_hidden_dim": 5,
        "candidate_feature_mode": "token_aware_mean_max",
        "use_confidence": False,
        "use_position": False,
        "use_history": False,
        "use_preference": False,
        "max_position": 7,
        "history_hidden_dim": 6,
        "preference_dim": 2,
        "preference_hidden_dim": 4,
        "preference_embedding_dim": 3,
        "fusion_hidden_dim": 8,
        "fusion_bottleneck_dim": 4,
        "lambda_max": 1.5,
        "device": "cpu",
    }
    values.update(overrides)
    return SmartRouterConfig(**values)


def make_topk(
    leading_shape: tuple[int, ...] = (2,),
    *,
    vocab_size: int = 31,
    top_k: int = 3,
) -> TAROTopKBatch:
    rows = 1
    for value in leading_shape:
        rows *= value
    base_ids = torch.arange(rows * top_k).reshape(*leading_shape, top_k)
    base_ids = base_ids.remainder(vocab_size)
    guide_ids = (base_ids + 5).remainder(vocab_size)
    rank = torch.arange(top_k, dtype=torch.float32)
    rank = (float(top_k) - rank).reshape((1,) * len(leading_shape) + (top_k,))
    offset = torch.arange(rows, dtype=torch.float32).reshape(
        *leading_shape,
        1,
    )
    return TAROTopKBatch(
        base_token_ids=base_ids.long(),
        base_logits=rank.expand(*leading_shape, top_k) + 0.1 * offset,
        reward_token_ids=guide_ids.long(),
        reward_logits=(1.3 * rank).expand(*leading_shape, top_k) - 0.05 * offset,
    )


def history_batch(
    topk: TAROTopKBatch,
    *,
    selected_score: torch.Tensor | None = None,
    preference: torch.Tensor | None = None,
) -> SmartRouterBatch:
    batch_size, sequence_length = topk.base_logits.shape[:2]
    return SmartRouterBatch(
        topk=topk,
        position=torch.arange(sequence_length).expand(batch_size, -1),
        selected_score=(
            torch.zeros(batch_size, sequence_length)
            if selected_score is None
            else selected_score
        ),
        preference=preference,
    )


class SmartConfigTests(unittest.TestCase):
    def test_shipped_ablation_presets_load_with_exact_toggles(self) -> None:
        config_dir = WORKSPACE_ROOT / "router_v2" / "configs"
        expected = SmartRouterConfig.VARIANT_PRESETS
        for variant, toggles in expected.items():
            path = config_dir / f"{variant}.json"
            with self.subTest(variant=variant):
                config = SmartRouterConfig.load_json(path)
                self.assertEqual(config.variant, variant)
                self.assertEqual(config.feature_toggles, toggles)

    def test_named_variant_rejects_mislabeled_feature_ablation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Use variant='custom'"):
            SmartRouterConfig(
                variant="v2_topk_confidence",
                use_confidence=False,
                use_position=False,
                use_history=False,
                use_preference=False,
            )

    def test_custom_variant_can_toggle_every_group_independently(self) -> None:
        config = smart_config(
            candidate_feature_mode="disabled",
            use_confidence=False,
            use_position=True,
            use_history=False,
            use_preference=False,
        )
        self.assertEqual(
            config.feature_toggles,
            {
                "candidate_feature_mode": "disabled",
                "use_confidence": False,
                "use_position": True,
                "use_history": False,
                "use_preference": False,
            },
        )

    def test_smart_json_schemas_are_separate_and_valid(self) -> None:
        schema_dir = WORKSPACE_ROOT / "router_v2" / "schemas"
        for name in (
            "smart_router_config.schema.json",
            "smart_router_checkpoint.schema.json",
        ):
            with (schema_dir / name).open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_production_preference_dimension_and_smoke_config_are_generic(
        self,
    ) -> None:
        alpha_config = SmartRouterConfig.load_json(DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG.name, "v2_topk_state_history_alpha.json")
        self.assertEqual(alpha_config.vocab_size, 50257)
        self.assertEqual(alpha_config.top_k, 20)
        self.assertEqual(alpha_config.history_hidden_dim, 32)
        multi_objective = replace(alpha_config, preference_dim=2)
        model = SmartTokenRouter(multi_objective)
        first_layer = model.preference_encoder[0]
        self.assertIsInstance(first_layer, nn.Linear)
        self.assertEqual(first_layer.in_features, 2)

        rad_config = SmartRouterConfig.load_json(
            DEFAULT_CONFIG.with_name("v2_topk_state_history.json")
        )
        self.assertFalse(rad_config.use_preference)


class SmartArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_token_aware_encoder_has_separate_streams_and_exact_dimension(self) -> None:
        config = smart_config(use_confidence=True)
        model = SmartTokenRouter(config)
        groups = model.build_static_feature_groups(
            SmartRouterBatch(topk=make_topk())
        )

        self.assertIsNot(model.base_candidate_encoder, model.guide_candidate_encoder)
        self.assertEqual(groups["candidate"].shape, (2, 4 * 5))
        self.assertEqual(
            groups["confidence"].shape,
            (2, len(SMART_CONFIDENCE_FEATURE_NAMES)),
        )
        self.assertEqual(model.feature_dim, 20 + 11)

    def test_token_identity_affects_encoded_candidates(self) -> None:
        model = SmartTokenRouter(smart_config())
        first = make_topk()
        second = TAROTopKBatch(
            base_token_ids=(first.base_token_ids + 1).remainder(31),
            base_logits=first.base_logits.clone(),
            reward_token_ids=(first.reward_token_ids + 1).remainder(31),
            reward_logits=first.reward_logits.clone(),
        )
        first_features = model.build_static_feature_groups(
            SmartRouterBatch(first)
        )["candidate"]
        second_features = model.build_static_feature_groups(
            SmartRouterBatch(second)
        )["candidate"]

        self.assertFalse(torch.allclose(first_features, second_features))

    def test_mean_max_pool_is_invariant_to_consistent_candidate_permutation(self) -> None:
        model = SmartTokenRouter(smart_config())
        topk = make_topk()
        permutation = torch.tensor([2, 0, 1])
        permuted = TAROTopKBatch(
            base_token_ids=topk.base_token_ids.index_select(-1, permutation),
            base_logits=topk.base_logits.index_select(-1, permutation),
            reward_token_ids=topk.reward_token_ids.index_select(-1, permutation),
            reward_logits=topk.reward_logits.index_select(-1, permutation),
        )

        original = model.build_static_feature_groups(
            SmartRouterBatch(topk)
        )["candidate"]
        changed = model.build_static_feature_groups(
            SmartRouterBatch(permuted)
        )["candidate"]
        torch.testing.assert_close(changed, original)

    def test_architecture_contains_router_gru_and_no_backbone_hidden_api(self) -> None:
        model = SmartTokenRouter(
            smart_config(use_history=True, use_confidence=True)
        )
        self.assertIsInstance(model.history_gru, nn.GRUCell)
        self.assertEqual(model.history_gru.input_size, 4)
        self.assertEqual(model.history_gru.hidden_size, 6)
        self.assertNotIn("hidden_states", SmartRouterBatch.__annotations__)
        self.assertNotIn(
            "backbone_hidden_states",
            inspect.signature(model.forward).parameters,
        )

    def test_lambda_is_scalar_bounded_and_routes_every_vocab_item(self) -> None:
        config = smart_config(lambda_max=1.5)
        model = SmartTokenRouter(config)
        output = model.predict_lambda(SmartRouterBatch(make_topk()))

        self.assertEqual(output.lambda_t.shape, (2, 1))
        self.assertTrue(bool((output.lambda_t > 0).all()))
        self.assertTrue(bool((output.lambda_t < config.lambda_max).all()))
        base = torch.tensor([[1.0, 2.0, 3.0]])
        guide = torch.tensor([[3.0, 6.0, 9.0]])
        strength = torch.tensor([[0.25]])
        routed = SmartTokenRouter.route_logits(strength, base, guide)
        torch.testing.assert_close(routed, base + 0.25 * (guide - base))


class FeatureToggleAndPerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_candidate_group_on_and_off(self) -> None:
        topk = make_topk()
        changed = TAROTopKBatch(
            base_token_ids=(topk.base_token_ids + 1).remainder(31),
            base_logits=topk.base_logits,
            reward_token_ids=(topk.reward_token_ids + 1).remainder(31),
            reward_logits=topk.reward_logits,
        )
        enabled = SmartTokenRouter(smart_config())
        first = enabled.predict_lambda(SmartRouterBatch(topk)).lambda_t
        second = enabled.predict_lambda(SmartRouterBatch(changed)).lambda_t
        self.assertFalse(torch.allclose(first, second))

        disabled = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_position=True,
            )
        )
        positions = torch.tensor([1, 2])
        first = disabled.predict_lambda(
            SmartRouterBatch(topk, position=positions)
        ).lambda_t
        second = disabled.predict_lambda(
            SmartRouterBatch(changed, position=positions)
        ).lambda_t
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_confidence_group_on_and_off(self) -> None:
        topk = make_topk()
        changed = TAROTopKBatch(
            base_token_ids=topk.base_token_ids,
            base_logits=topk.base_logits * torch.tensor([1.0, 0.2, 0.1]),
            reward_token_ids=topk.reward_token_ids,
            reward_logits=topk.reward_logits * torch.tensor([0.1, 0.4, 1.0]),
        )
        enabled = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_confidence=True,
            )
        )
        first = enabled.predict_lambda(SmartRouterBatch(topk)).lambda_t
        second = enabled.predict_lambda(SmartRouterBatch(changed)).lambda_t
        self.assertFalse(torch.allclose(first, second))

        disabled = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_position=True,
            )
        )
        position = torch.tensor([2, 2])
        first = disabled.predict_lambda(
            SmartRouterBatch(topk, position=position)
        ).lambda_t
        second = disabled.predict_lambda(
            SmartRouterBatch(changed, position=position)
        ).lambda_t
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_position_group_on_and_off(self) -> None:
        topk = make_topk()
        enabled = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_position=True,
            )
        )
        early = enabled.predict_lambda(
            SmartRouterBatch(topk, position=torch.tensor([0, 0]))
        ).lambda_t
        late = enabled.predict_lambda(
            SmartRouterBatch(topk, position=torch.tensor([7, 7]))
        ).lambda_t
        self.assertFalse(torch.allclose(early, late))

        disabled = SmartTokenRouter(smart_config())
        early = disabled.predict_lambda(
            SmartRouterBatch(topk, position=torch.tensor([0, 0]))
        ).lambda_t
        late = disabled.predict_lambda(
            SmartRouterBatch(topk, position=torch.tensor([7, 7]))
        ).lambda_t
        torch.testing.assert_close(early, late, rtol=0, atol=0)

    def test_history_perturbation_is_causal(self) -> None:
        config = smart_config(
            candidate_feature_mode="disabled",
            use_history=True,
        )
        model = SmartTokenRouter(config)
        topk = make_topk((2, 4))
        baseline_scores = torch.zeros(2, 4)
        perturbed_scores = baseline_scores.clone()
        perturbed_scores[:, 1] = 10.0
        baseline = model.predict_lambda(
            history_batch(topk, selected_score=baseline_scores)
        ).lambda_t
        perturbed = model.predict_lambda(
            history_batch(topk, selected_score=perturbed_scores)
        ).lambda_t

        torch.testing.assert_close(
            baseline[:, :2],
            perturbed[:, :2],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.allclose(baseline[:, 2:], perturbed[:, 2:]))

        future_scores = baseline_scores.clone()
        future_scores[:, -1] = 10.0
        future_perturbed = model.predict_lambda(
            history_batch(topk, selected_score=future_scores)
        ).lambda_t
        torch.testing.assert_close(
            baseline,
            future_perturbed,
            rtol=0,
            atol=0,
        )

    def test_history_without_confidence_does_not_read_topk_statistics(self) -> None:
        model = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_confidence=False,
                use_history=True,
            )
        )
        topk = make_topk((2, 4))
        changed = TAROTopKBatch(
            base_token_ids=(topk.base_token_ids + 3).remainder(31),
            base_logits=topk.base_logits * 7.0,
            reward_token_ids=(topk.reward_token_ids + 4).remainder(31),
            reward_logits=-topk.reward_logits,
        )
        original = model.predict_lambda(history_batch(topk)).lambda_t
        perturbed = model.predict_lambda(history_batch(changed)).lambda_t
        torch.testing.assert_close(original, perturbed, rtol=0, atol=0)

    def test_preference_alpha_shuffle_and_disabled_invariance(self) -> None:
        topk = make_topk((2,))
        preference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        enabled = SmartTokenRouter(
            smart_config(
                candidate_feature_mode="disabled",
                use_preference=True,
            )
        )
        original = enabled.predict_lambda(
            SmartRouterBatch(topk, preference=preference)
        ).lambda_t
        shuffled = enabled.predict_lambda(
            SmartRouterBatch(topk, preference=preference.flip(0))
        ).lambda_t
        torch.testing.assert_close(original[0], shuffled[1])
        torch.testing.assert_close(original[1], shuffled[0])
        self.assertFalse(torch.allclose(original[0], original[1]))

        disabled = SmartTokenRouter(smart_config())
        first = disabled.predict_lambda(
            SmartRouterBatch(topk, preference=preference)
        ).lambda_t
        second = disabled.predict_lambda(
            SmartRouterBatch(topk, preference=preference.flip(0))
        ).lambda_t
        torch.testing.assert_close(first, second, rtol=0, atol=0)


class SmartGradientAndCheckpointTests(unittest.TestCase):
    def test_only_smart_router_parameters_receive_gradients(self) -> None:
        torch.manual_seed(44)
        config = smart_config(
            use_confidence=True,
            use_position=True,
            use_history=True,
            use_preference=True,
        )
        model = SmartTokenRouter(config)
        original = make_topk((2, 3))
        base_topk_logits = original.base_logits.clone().requires_grad_(True)
        guide_topk_logits = original.reward_logits.clone().requires_grad_(True)
        topk = TAROTopKBatch(
            base_token_ids=original.base_token_ids,
            base_logits=base_topk_logits,
            reward_token_ids=original.reward_token_ids,
            reward_logits=guide_topk_logits,
        )
        selected_score = torch.zeros(2, 3, requires_grad=True)
        preference = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        full_base = torch.randn(2, 3, 31, requires_grad=True)
        full_guide = torch.randn(2, 3, 31, requires_grad=True)
        output = model(
            history_batch(
                topk,
                selected_score=selected_score,
                preference=preference,
            ),
            full_base,
            full_guide,
        )
        if output.guided_logits is None:
            self.fail("Expected routed full-vocabulary logits")
        (output.guided_logits.square().mean() + output.lambda_t.mean()).backward()

        self.assertIsNone(base_topk_logits.grad)
        self.assertIsNone(guide_topk_logits.grad)
        self.assertIsNone(full_base.grad)
        self.assertIsNone(full_guide.grad)
        self.assertIsNone(selected_score.grad)
        self.assertIsNone(preference.grad)
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_smart_checkpoint_round_trip_and_no_overwrite(self) -> None:
        torch.manual_seed(55)
        config = smart_config(
            use_confidence=True,
            use_position=True,
            use_history=True,
            use_preference=True,
        )
        model = SmartTokenRouter(config).eval()
        batch = history_batch(
            make_topk((2, 3)),
            preference=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        expected = model.predict_lambda(batch)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "smart-router.pt"
            save_smart_checkpoint(
                model,
                path,
                training_state={"step": 9},
                metadata={"test": True},
            )
            with self.assertRaises(FileExistsError):
                save_smart_checkpoint(model, path)
            loaded, payload = load_smart_checkpoint(path)
            actual = loaded.eval().predict_lambda(batch)

        torch.testing.assert_close(actual.lambda_t, expected.lambda_t, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.router_state,
            expected.router_state,
            rtol=0,
            atol=0,
        )
        self.assertEqual(payload["format"], SMART_CHECKPOINT_FORMAT)
        self.assertEqual(payload["training_state"]["step"], 9)
        self.assertTrue(payload["metadata"]["test"])

    def test_checkpoint_mirrors_config_and_rejects_taro_payload(self) -> None:
        model = SmartTokenRouter(smart_config())
        payload = build_smart_checkpoint_payload(model)
        payload["lambda_max"] = 99.0
        with self.assertRaisesRegex(ValueError, "does not match embedded config"):
            validate_smart_checkpoint_payload(payload)

        taro = TAROTokenRouter(
            TARORouterConfig(
                vocab_size=31,
                top_k=3,
                token_embedding_dim=4,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "taro.pt"
            save_taro_checkpoint(taro, path)
            with self.assertRaisesRegex(ValueError, "Smart Router"):
                load_smart_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
