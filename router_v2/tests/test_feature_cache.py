from __future__ import annotations

import copy
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from router_v2.cache.features import (
    DERIVED_FEATURE_NAMES,
    compute_confidence_disagreement,
)
from router_v2.cache.io import (
    load_cache_shard,
    require_path_within,
    save_cache_shard,
    sha256_file,
)
from router_v2.cache.schema import (
    ROUTER_FEATURE_FIELDS,
    TARGET_FIELDS,
    build_cache_shard,
    build_cache_shard_from_full_logits,
    validate_cache_shard,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def make_full_shard(
    *,
    gold_token_id: torch.Tensor | None = None,
) -> dict[str, object]:
    base_full_logits = torch.tensor(
        [
            [9.0, 8.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
            [-4.0, -3.0, -2.0, -1.0, 0.0, 7.0, 8.0, 9.0],
        ]
    )
    guide_full_logits = torch.tensor(
        [
            [-4.0, -3.0, -2.0, -1.0, 0.0, 7.0, 8.0, 9.0],
            [9.0, 8.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        ]
    )
    if gold_token_id is None:
        gold_token_id = torch.tensor([4, 3])
    return build_cache_shard_from_full_logits(
        split="train",
        shard_index=0,
        sample_ids=["sample-a", "sample-b"],
        position=torch.tensor([0, 5]),
        base_full_logits=base_full_logits,
        guide_full_logits=guide_full_logits,
        gold_token_id=gold_token_id,
        top_k=2,
        max_position=10,
        source={"dataset": "unit-test"},
    )


def assert_feature_mapping_equal(
    case: unittest.TestCase,
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    for field in ROUTER_FEATURE_FIELDS:
        first_value = first[field]
        second_value = second[field]
        if isinstance(first_value, torch.Tensor):
            torch.testing.assert_close(first_value, second_value)
        elif isinstance(first_value, dict):
            case.assertEqual(set(first_value), set(second_value))
            for name in first_value:
                torch.testing.assert_close(first_value[name], second_value[name])
        else:
            case.assertEqual(first_value, second_value)


class CacheCandidateTests(unittest.TestCase):
    def test_full_logits_select_independent_topk_in_descending_order(self) -> None:
        shard = make_full_shard()

        self.assertEqual(shard["base_token_ids"].tolist(), [[0, 1], [7, 6]])
        self.assertEqual(shard["guide_token_ids"].tolist(), [[7, 6], [0, 1]])
        self.assertEqual(shard["base_logits"].tolist(), [[9.0, 8.0], [9.0, 8.0]])
        self.assertEqual(shard["guide_logits"].tolist(), [[9.0, 8.0], [9.0, 8.0]])

    def test_gold_is_target_only_and_is_not_forced_into_topk(self) -> None:
        shard = make_full_shard()

        self.assertNotIn(4, shard["base_token_ids"][0].tolist())
        self.assertNotIn(4, shard["guide_token_ids"][0].tolist())
        self.assertNotIn(3, shard["base_token_ids"][1].tolist())
        self.assertNotIn(3, shard["guide_token_ids"][1].tolist())
        self.assertNotIn("gold_token_id", shard["router_feature_fields"])
        self.assertEqual(tuple(shard["target_fields"]), TARGET_FIELDS)

    def test_derived_feature_api_cannot_accept_gold(self) -> None:
        signature = inspect.signature(compute_confidence_disagreement)

        self.assertNotIn("gold_token_id", signature.parameters)
        self.assertNotIn("gold_token_ids", signature.parameters)

    def test_changing_gold_does_not_change_router_inputs_or_features(self) -> None:
        first = make_full_shard(gold_token_id=torch.tensor([4, 3]))
        second = make_full_shard(gold_token_id=torch.tensor([2, 5]))

        assert_feature_mapping_equal(self, first, second)
        self.assertFalse(torch.equal(first["gold_token_id"], second["gold_token_id"]))


class DerivedFeatureTests(unittest.TestCase):
    def test_known_identical_distributions(self) -> None:
        token_ids = torch.tensor([[1, 2]])
        logits = torch.tensor([[2.0, 0.0]])
        features = compute_confidence_disagreement(
            token_ids,
            logits,
            token_ids,
            logits,
            torch.tensor([5]),
            max_position=10,
        )
        probs = torch.softmax(logits, dim=-1)
        expected_entropy = -(probs * probs.log()).sum(dim=-1)

        torch.testing.assert_close(features.base_entropy, expected_entropy)
        torch.testing.assert_close(features.guide_entropy, expected_entropy)
        torch.testing.assert_close(features.base_margin, torch.tensor([2.0]))
        torch.testing.assert_close(features.guide_margin, torch.tensor([2.0]))
        torch.testing.assert_close(features.js_divergence, torch.tensor([0.0]))
        torch.testing.assert_close(features.top1_agreement, torch.tensor([1.0]))
        torch.testing.assert_close(features.topk_overlap, torch.tensor([1.0]))
        torch.testing.assert_close(features.rank_correlation, torch.tensor([1.0]))
        torch.testing.assert_close(features.normalized_position, torch.tensor([0.5]))

    def test_disjoint_support_has_log_two_js_and_zero_overlap(self) -> None:
        features = compute_confidence_disagreement(
            torch.tensor([[1, 2]]),
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[3, 4]]),
            torch.tensor([[5.0, 1.0]]),
            torch.tensor([0]),
            max_position=10,
        )

        torch.testing.assert_close(
            features.js_divergence,
            torch.tensor([math.log(2.0)]),
        )
        torch.testing.assert_close(features.topk_overlap, torch.tensor([0.0]))
        torch.testing.assert_close(features.top1_agreement, torch.tensor([0.0]))
        torch.testing.assert_close(features.rank_correlation, torch.tensor([0.0]))

    def test_consistent_token_logit_pair_permutation_is_consistent(self) -> None:
        base_ids = torch.tensor([[1, 2, 3]])
        base_logits = torch.tensor([[4.0, 2.0, 1.0]])
        guide_ids = torch.tensor([[3, 2, 1]])
        guide_logits = torch.tensor([[5.0, 3.0, 0.0]])
        position = torch.tensor([7])
        permutation = torch.tensor([2, 0, 1])
        original = compute_confidence_disagreement(
            base_ids,
            base_logits,
            guide_ids,
            guide_logits,
            position,
            max_position=20,
        )
        permuted = compute_confidence_disagreement(
            base_ids.index_select(-1, permutation),
            base_logits.index_select(-1, permutation),
            guide_ids.index_select(-1, permutation),
            guide_logits.index_select(-1, permutation),
            position,
            max_position=20,
        )

        torch.testing.assert_close(original.stacked(), permuted.stacked())

    def test_rows_do_not_read_future_or_neighbor_records(self) -> None:
        first_base_ids = torch.tensor([[1, 2]])
        first_base_logits = torch.tensor([[3.0, 1.0]])
        first_guide_ids = torch.tensor([[2, 4]])
        first_guide_logits = torch.tensor([[5.0, 0.0]])
        position = torch.tensor([2])
        isolated = compute_confidence_disagreement(
            first_base_ids,
            first_base_logits,
            first_guide_ids,
            first_guide_logits,
            position,
            max_position=10,
        )
        batched = compute_confidence_disagreement(
            torch.cat((first_base_ids, torch.tensor([[7, 8]]))),
            torch.cat((first_base_logits, torch.tensor([[100.0, -100.0]]))),
            torch.cat((first_guide_ids, torch.tensor([[9, 10]]))),
            torch.cat((first_guide_logits, torch.tensor([[-50.0, 50.0]]))),
            torch.tensor([2, 10]),
            max_position=10,
        )

        torch.testing.assert_close(isolated.stacked()[0], batched.stacked()[0])

    def test_normalized_position_uses_fixed_configured_limit(self) -> None:
        features = compute_confidence_disagreement(
            torch.tensor([[1], [2]]),
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[3], [4]]),
            torch.tensor([[2.0], [1.0]]),
            torch.tensor([3, 7]),
            max_position=10,
        )

        torch.testing.assert_close(
            features.normalized_position,
            torch.tensor([0.3, 0.7]),
        )


class CacheContractTests(unittest.TestCase):
    def test_raw_values_reconstruct_every_derived_feature(self) -> None:
        shard = make_full_shard()
        summary = validate_cache_shard(shard)

        self.assertEqual(summary["num_records"], 2)
        self.assertEqual(set(shard["derived"]), set(DERIVED_FEATURE_NAMES))

    def test_raw_candidate_order_is_preserved(self) -> None:
        shard = build_cache_shard(
            split="validation",
            shard_index=4,
            sample_ids=["ordered"],
            position=torch.tensor([1]),
            base_token_ids=torch.tensor([[8, 3, 5]]),
            base_logits=torch.tensor([[9.0, 7.0, 1.0]]),
            guide_token_ids=torch.tensor([[2, 6, 1]]),
            guide_logits=torch.tensor([[8.0, 4.0, 0.0]]),
            gold_token_id=torch.tensor([7]),
            max_position=12,
            vocab_size=10,
            source={"dataset": "unit-test"},
        )

        self.assertEqual(shard["base_token_ids"].tolist(), [[8, 3, 5]])
        self.assertEqual(shard["guide_token_ids"].tolist(), [[2, 6, 1]])

    def test_tampered_derived_feature_is_rejected(self) -> None:
        shard = make_full_shard()
        tampered = copy.deepcopy(shard)
        tampered["derived"]["base_entropy"][0] += 0.1

        with self.assertRaisesRegex(ValueError, "cannot be reconstructed"):
            validate_cache_shard(tampered)

    def test_target_field_cannot_be_declared_as_router_feature(self) -> None:
        shard = make_full_shard()
        shard["router_feature_fields"] = list(ROUTER_FEATURE_FIELDS) + [
            "gold_token_id"
        ]

        with self.assertRaisesRegex(ValueError, "router_feature_fields"):
            validate_cache_shard(shard)

    def test_write_boundary_rejects_v1_and_other_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "router_v2_cache"
            inside = root / "rad" / "train" / "shard.pt"

            self.assertEqual(
                require_path_within(inside, root, label="cache output"),
                inside.resolve(),
            )
            with self.assertRaisesRegex(
                ValueError,
                "must be under",
            ):
                require_path_within(
                    root.parent / "router_cache" / "rad.pt",
                    root,
                    label="cache output",
                )

    def test_save_load_round_trip_and_no_overwrite(self) -> None:
        shard = make_full_shard()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "shard-00000.pt"
            summary = save_cache_shard(shard, path)
            loaded = load_cache_shard(path)

            self.assertEqual(summary["sha256"], sha256_file(path))
            self.assertEqual(loaded["sample_ids"], shard["sample_ids"])
            torch.testing.assert_close(
                loaded["derived"]["js_divergence"],
                shard["derived"]["js_divergence"],
            )
            with self.assertRaises(FileExistsError):
                save_cache_shard(shard, path)


class SourceInventoryTests(unittest.TestCase):
    def test_generated_inventory_records_exact_local_schemas_and_hashes(self) -> None:
        path = WORKSPACE_ROOT / "dataset" / "router_v2_train" / "source_inventory.json"
        with path.open("r", encoding="utf-8") as handle:
            inventory = json.load(handle)

        expected_rad = {
            "train": (
                20000,
                "92486a32e7c451422682afdd3d0c55a47a3016cc977542cb600a6172937c924b",
            ),
            "validation": (
                2000,
                "70ee7d2fa35e311f4e0fa4a493564a9188313586668d2986c4e9038eaccac3fd",
            ),
            "dev": (
                500,
                "da38aa8f9760f355d605b0774bfa043692bc4afd1ad30e39bd15b417c4e0a691",
            ),
        }
        for split, (rows, digest) in expected_rad.items():
            actual = inventory["rad"]["splits"][split]
            self.assertTrue(actual["schema_valid"])
            self.assertEqual(actual["rows"], rows)
            self.assertEqual(actual["sha256"], digest)
        pku = inventory["pku_safe_rlhf"]["inventory"]
        self.assertTrue(inventory["pku_safe_rlhf"]["schema_inventory_complete"])
