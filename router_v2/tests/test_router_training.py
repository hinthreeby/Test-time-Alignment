from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as functional
from torch import nn

from router_v2.candidates import TAROTopKBatch
from router_v2.guide_model.data import RADSample
from router_v2.training.audit import (
    assert_frozen_models,
    assert_optimizer_contains_only,
    compare_frozen_snapshots,
    hash_tree,
)
from router_v2.training.config import RouterTrainingConfig
from router_v2.training.data import (
    CachedRouterBatch,
    CachedSequence,
    _sequence_ranges,
    collate_cached_sequences,
)
from router_v2.training.diagnostics import LambdaDiagnostics
from router_v2.training.engine import (
    _validate_stage_order,
    evaluate_fixed_lambda,
)
from router_v2.training.objective import compute_router_training_objective
from router_v2.training.online import OnlineLogitBatch
from router_v2.training.online import compute_online_logit_batch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def training_config(**overrides: object) -> RouterTrainingConfig:
    values: dict[str, object] = {
        "stage": "state",
        "router_kind": "smart",
        "router_config_path": (
            "router_v2/configs/v2_topk_confidence_position.json"
        ),
        "output_dir": "results/router_v2/training/state",
        "enforce_stage_order": False,
    }
    values.update(overrides)
    return RouterTrainingConfig(**values)


def sample(sample_id: str) -> RADSample:
    return RADSample(
        sample_id=sample_id,
        prompt="p",
        continuation="abc",
        label=1,
        source="unit",
        source_split="train",
        source_index=0,
    )


def cached_sequence(
    sample_id: str,
    length: int,
    *,
    preference: torch.Tensor | None = None,
) -> CachedSequence:
    top_k = 2
    token_ids = torch.arange(length * top_k).reshape(length, top_k) % 11
    logits = torch.arange(length * top_k, dtype=torch.float32).reshape(
        length,
        top_k,
    )
    return CachedSequence(
        sample=sample(sample_id),
        position=torch.arange(length),
        base_token_ids=token_ids.long(),
        base_logits=logits,
        guide_token_ids=(token_ids + 3).remainder(11).long(),
        guide_logits=logits + 0.5,
        gold_token_id=torch.arange(length).remainder(11).long(),
        preference=preference,
    )


class TrainingConfigTests(unittest.TestCase):
    def test_all_stage5_configs_and_schema_load(self) -> None:
        config_dir = WORKSPACE_ROOT / "router_v2" / "configs"
        loaded = {
            path.name: RouterTrainingConfig.load_json(path)
            for path in sorted(config_dir.glob("train_stage5_*.json"))
        }
        self.assertEqual(len(loaded), 6)
        self.assertEqual(loaded["train_stage5_taro_rad.json"].stage, "taro")
        self.assertEqual(loaded["train_stage5_state_rad.json"].stage, "state")
        self.assertEqual(loaded["train_stage5_history_rad.json"].stage, "history")
        alpha = loaded["train_stage5_alpha_rad_blocked.json"]
        self.assertEqual(alpha.stage, "alpha")
        self.assertEqual(alpha.preference_source, "cache")

        schema_path = (
            WORKSPACE_ROOT
            / "router_v2"
            / "schemas"
            / "router_training_config.schema.json"
        )
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertFalse(schema["additionalProperties"])

    def test_stage_and_regularizer_contracts(self) -> None:
        with self.assertRaises(ValueError):
            training_config(stage="taro", router_kind="smart")
        with self.assertRaises(ValueError):
            RouterTrainingConfig(
                stage="alpha",
                router_kind="smart",
                preference_source="none",
            )
        with self.assertRaises(ValueError):
            RouterTrainingConfig(smoothness_weight=0.1)
        regularized = training_config(
            stage="history",
            router_config_path=(
                "router_v2/configs/v2_topk_state_history.json"
            ),
            smoothness_weight=0.1,
            strength_weight=0.2,
        )
        self.assertEqual(regularized.smoothness_weight, 0.1)
        self.assertEqual(regularized.strength_weight, 0.2)

    def test_stage_order_requires_passing_predecessor(self) -> None:
        config = training_config(enforce_stage_order=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "state"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "predecessor"):
                _validate_stage_order(config, output)
            predecessor = root / "taro"
            predecessor.mkdir()
            (predecessor / "run_status.json").write_text(
                json.dumps({"status": "PASS"}),
                encoding="utf-8",
            )
            _validate_stage_order(config, output)


class TrainingObjectiveTests(unittest.TestCase):
    def test_masked_nll_matches_direct_cross_entropy(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(2, 3, 7, requires_grad=True)
        gold = torch.tensor([[1, 2, 3], [4, 5, 0]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        gate = torch.full((2, 3, 1), 0.4, requires_grad=True)
        lambda_t = 2.0 * gate

        objective = compute_router_training_objective(
            logits,
            gold,
            mask,
            gate,
            lambda_t,
            lambda_max=2.0,
        )
        direct = functional.cross_entropy(
            logits[mask],
            gold[mask],
        )

        torch.testing.assert_close(objective.nll_loss, direct)
        torch.testing.assert_close(objective.total_loss, direct)
        self.assertEqual(objective.token_count, 5)

    def test_optional_entropy_smoothness_and_strength_are_exact(self) -> None:
        logits = torch.tensor(
            [[[3.0, 1.0], [1.0, 3.0], [2.0, 0.0]]]
        )
        gold = torch.tensor([[0, 1, 0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        gate = torch.tensor([[[0.2], [0.5], [0.8]]], requires_grad=True)
        lambda_t = 2.0 * gate
        objective = compute_router_training_objective(
            logits,
            gold,
            mask,
            gate,
            lambda_t,
            lambda_max=2.0,
            entropy_weight=0.3,
            smoothness_weight=0.4,
            strength_weight=0.5,
        )
        expected_entropy = -(
            gate[..., 0] * gate[..., 0].log()
            + (1.0 - gate[..., 0]) * (1.0 - gate[..., 0]).log()
        ).mean()
        expected_smoothness = torch.tensor([(0.6**2), (0.6**2)]).mean()
        expected_strength = gate[..., 0].square().mean()

        torch.testing.assert_close(objective.entropy, expected_entropy)
        torch.testing.assert_close(objective.smoothness, expected_smoothness)
        torch.testing.assert_close(objective.strength, expected_strength)
        torch.testing.assert_close(
            objective.total_loss,
            objective.nll_loss
            + 0.3 * expected_entropy
            + 0.4 * expected_smoothness
            + 0.5 * expected_strength,
        )


class CacheSequenceTests(unittest.TestCase):
    def test_collation_preserves_samples_and_masks_padding(self) -> None:
        first = cached_sequence("a", 3, preference=torch.tensor([0.2, 0.8]))
        second = cached_sequence("b", 2, preference=torch.tensor([0.7, 0.3]))
        batch = collate_cached_sequences([first, second])

        self.assertEqual(batch.sample_ids, ("a", "b"))
        self.assertEqual(batch.topk.base_token_ids.shape, (2, 3, 2))
        self.assertEqual(
            batch.valid_mask.tolist(),
            [[True, True, True], [True, True, False]],
        )
        self.assertEqual(batch.preference.shape, (2, 2))

    def test_sequence_ranges_reject_noncontiguous_duplicate_sample(self) -> None:
        self.assertEqual(
            _sequence_ranges(["a", "a", "b", "b"]),
            [(0, 2, "a"), (2, 4, "b")],
        )
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            _sequence_ranges(["a", "b", "a"])


class OnlineLogitTests(unittest.TestCase):
    def test_teacher_forced_full_logits_are_fp32_frozen_and_aligned(self) -> None:
        vocabulary = {"p": 0, "a": 1, "b": 2, "c": 3}

        class Tokenizer:
            pad_token_id = 4

            def encode(self, text: str, *, add_special_tokens: bool):
                del add_special_tokens
                return [vocabulary[character] for character in text]

        class Model(nn.Module):
            def __init__(self, offset: float) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(offset))

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                use_cache: bool,
            ) -> SimpleNamespace:
                del attention_mask, position_ids, use_cache
                vocab = torch.arange(5, dtype=torch.float32)
                logits = vocab.view(1, 1, 5).expand(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    -1,
                )
                return SimpleNamespace(logits=logits + self.anchor)

        sequence = cached_sequence("a", 3)
        sequence = CachedSequence(
            **{
                **sequence.__dict__,
                "gold_token_id": torch.tensor([1, 2, 3]),
            }
        )
        cached = collate_cached_sequences([sequence])
        base_model = Model(0.0)
        guide_model = Model(0.5)
        online = compute_online_logit_batch(
            base_model,
            guide_model,
            Tokenizer(),
            cached,
            device=torch.device("cpu"),
            max_length=8,
            max_continuation_tokens=3,
        )

        self.assertEqual(online.base_logits.shape, (1, 3, 5))
        self.assertEqual(online.guide_logits.shape, (1, 3, 5))
        self.assertEqual(online.base_logits.dtype, torch.float32)
        self.assertFalse(online.base_logits.requires_grad)
        self.assertFalse(online.guide_logits.requires_grad)
        expected = torch.tensor([1.0, 2.0, 3.0]) - torch.logsumexp(
            torch.arange(5, dtype=torch.float32),
            dim=0,
        )
        torch.testing.assert_close(
            online.base_selected_logprob[0],
            expected,
        )
        self.assertIsNone(base_model.anchor.grad)
        self.assertIsNone(guide_model.anchor.grad)


class DiagnosticsAndBaselineTests(unittest.TestCase):
    def test_diagnostics_include_all_required_views_and_resume_state(self) -> None:
        logits = torch.randn(1, 3, 5)
        gold = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        gate = torch.tensor([[[0.2], [0.4], [0.8]]])
        objective = compute_router_training_objective(
            logits,
            gold,
            mask,
            gate,
            gate,
            lambda_max=1.0,
        )
        diagnostics = LambdaDiagnostics(lambda_max=1.0, alpha_bins=2)
        diagnostics.update(
            objective,
            gate,
            mask,
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.1, 0.2, 0.4]]),
            torch.tensor([[0.25, 0.75]]),
        )
        resumed = LambdaDiagnostics.from_state_dict(diagnostics.state_dict())
        summary = resumed.finalize()

        self.assertIn("p05", summary["lambda"])
        self.assertEqual(len(summary["lambda_by_position"]), 3)
        self.assertIsNotNone(summary["lambda_vs_base_entropy_pearson"])
        self.assertIsNotNone(summary["lambda_vs_js_pearson"])
        self.assertEqual(
            summary["lambda_by_alpha_bucket"]["status"],
            "available",
        )

    def test_same_average_fixed_baseline_uses_full_logits(self) -> None:
        topk = TAROTopKBatch(
            base_token_ids=torch.tensor([[[0, 1], [1, 0]]]),
            base_logits=torch.tensor([[[3.0, 2.0], [4.0, 1.0]]]),
            reward_token_ids=torch.tensor([[[1, 0], [0, 1]]]),
            reward_logits=torch.tensor([[[4.0, 1.0], [2.0, 1.0]]]),
        )
        cached = CachedRouterBatch(
            sample_ids=("a",),
            samples=(sample("a"),),
            topk=topk,
            gold_token_ids=torch.tensor([[1, 0]]),
            position=torch.tensor([[0, 1]]),
            valid_mask=torch.tensor([[True, True]]),
            preference=None,
        )
        base = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]]])
        guide = torch.tensor([[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]])
        online = OnlineLogitBatch(
            base_logits=base,
            guide_logits=guide,
            base_selected_logprob=torch.zeros(1, 2),
            extraction_batch=None,
        )

        class FakeCache:
            manifest = {
                "extraction_parameters": {"max_continuation_tokens": 2}
            }

            def iter_batches(self, **_: object):
                yield cached

        config = training_config(batch_size=1, max_validation_samples=1)
        with mock.patch(
            "router_v2.training.engine._online_batch",
            return_value=online,
        ):
            result = evaluate_fixed_lambda(
                0.25,
                1.0,
                config,
                FakeCache(),
                object(),
                object(),
                object(),
                object(),
                torch.device("cpu"),
            )
        guided = base + 0.25 * (guide - base)
        expected = functional.cross_entropy(
            guided.reshape(-1, 3),
            cached.gold_token_ids.reshape(-1),
        )
        self.assertAlmostEqual(result["nll"], float(expected), places=6)
        self.assertEqual(result["lambda"], 0.25)


class FrozenAuditTests(unittest.TestCase):
    def test_optimizer_boundary_and_tree_snapshot(self) -> None:
        frozen = nn.Linear(2, 2)
        frozen.requires_grad_(False)
        router = nn.Linear(2, 1)
        assert_frozen_models(frozen)
        optimizer = torch.optim.AdamW(router.parameters())
        assert_optimizer_contains_only(optimizer, router.parameters())
        with self.assertRaises(ValueError):
            mixed = torch.optim.AdamW(
                list(router.parameters()) + list(frozen.parameters())
            )
            assert_optimizer_contains_only(mixed, router.parameters())

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "value.txt").write_text("stable", encoding="utf-8")
            before = {
                "base_model": {"checkpoint_sha256": "a"},
                "guide_model": {"checkpoint_sha256": "b"},
                "parm": hash_tree(root),
            }
            unchanged = compare_frozen_snapshots(before, before)
            self.assertTrue(unchanged["pass"])
            (root / "value.txt").write_text("changed", encoding="utf-8")
            after = {**before, "parm": hash_tree(root)}
            self.assertFalse(compare_frozen_snapshots(before, after)["pass"])


if __name__ == "__main__":
    unittest.main()
