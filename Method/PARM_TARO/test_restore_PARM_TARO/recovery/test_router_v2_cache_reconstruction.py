from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from router_v2.cache import features, io, schema


ROOT = Path(__file__).resolve().parents[2]
RECONSTRUCTED = (
    ROOT / "router_v2/cache/__init__.py",
    ROOT / "router_v2/cache/features.py",
    ROOT / "router_v2/cache/io.py",
    ROOT / "router_v2/cache/schema.py",
    ROOT / "router_v2/cache/inventory.py",
)


def make_shard(preference: torch.Tensor | None = None) -> dict[str, object]:
    return schema.build_cache_shard(
        split="train",
        shard_index=0,
        sample_ids=["a", "b"],
        position=torch.tensor([0, 3], dtype=torch.long),
        base_token_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        base_logits=torch.tensor([[4.0, 2.0, 0.0], [3.0, 1.0, -1.0]]),
        guide_token_ids=torch.tensor([[1, 3, 7], [6, 5, 4]]),
        guide_logits=torch.tensor([[3.0, 1.0, 0.0], [4.0, 2.0, 0.0]]),
        gold_token_id=torch.tensor([7, 2], dtype=torch.long),
        max_position=8,
        vocab_size=10,
        source={"dataset": "synthetic"},
        preference_vector=preference,
    )


def test_reconstruction_markers_are_present() -> None:
    for path in RECONSTRUCTED:
        lines = path.read_text(encoding="utf-8").splitlines()[:4]
        assert lines == [
            "# RECONSTRUCTED SOURCE",
            "# Original file was lost because router_v2/cache was gitignored.",
            "# Reconstructed from imports, documentation, schemas, and historical artifacts.",
            "# Do not treat this file as byte-identical to the historical implementation.",
        ]


def test_schema_serialization_round_trip_is_deterministic(tmp_path: Path) -> None:
    shard = make_shard(torch.tensor([[0.25, 0.75], [0.5, 0.5]]))
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    io.save_cache_shard(shard, first)
    io.save_cache_shard(shard, second)
    loaded = io.load_cache_shard(first)
    assert schema.validate_cache_shard(loaded)["valid"] is True
    assert loaded["sample_ids"] == ["a", "b"]
    torch.testing.assert_close(loaded["derived"]["js_divergence"], shard["derived"]["js_divergence"])
    # Determinism here is semantic: PyTorch zip container names may differ.
    second_loaded = io.load_cache_shard(second)
    torch.testing.assert_close(loaded["base_logits"], second_loaded["base_logits"])
    manifest_a = tmp_path / "a.json"
    manifest_b = tmp_path / "b.json"
    io.write_json_atomic(manifest_a, {"b": 2, "a": 1})
    io.write_json_atomic(manifest_b, {"a": 1, "b": 2})
    assert manifest_a.read_bytes() == manifest_b.read_bytes()


def test_required_field_validation() -> None:
    shard = make_shard()
    del shard["position"]
    with pytest.raises(ValueError, match="missing required fields"):
        schema.validate_cache_shard(shard)


def test_candidate_length_agreement() -> None:
    with pytest.raises(ValueError, match="agree"):
        schema.build_cache_shard(
            split="train", shard_index=0, sample_ids=["a"],
            position=torch.tensor([0]),
            base_token_ids=torch.tensor([[1, 2]]), base_logits=torch.tensor([[1.0, 0.0]]),
            guide_token_ids=torch.tensor([[1, 2, 3]]), guide_logits=torch.tensor([[1.0, 0.0, -1.0]]),
            gold_token_id=torch.tensor([1]), max_position=4, vocab_size=8, source={},
        )


def test_gold_index_semantics_and_no_gold_leakage() -> None:
    shard = make_shard()
    # Independent base/guide Top-K lists make a single historical gold_index
    # ambiguous. The reconstructed V2 contract stores gold only as a target.
    assert "gold_index" not in shard
    assert "gold_token_id" not in shard["router_feature_fields"]
    assert tuple(shard["target_fields"]) == ("gold_token_id",)
    base_row = shard["base_token_ids"][0].tolist()
    gold = int(shard["gold_token_id"][0])
    expected_index = base_row.index(gold) if gold in base_row else -1
    assert expected_index == -1
    signature = inspect.signature(features.compute_confidence_disagreement)
    assert "gold_token_id" not in signature.parameters


@pytest.mark.parametrize("preference", [None, torch.tensor([[0.2, 0.8], [0.7, 0.3]])])
def test_preference_vector_optionality(preference: torch.Tensor | None) -> None:
    shard = make_shard(preference)
    assert (shard["preference_vector"] is None) == (preference is None)


def test_finite_logits_and_features_are_required() -> None:
    shard = make_shard()
    shard["base_logits"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        schema.validate_cache_shard(shard)


def test_no_future_token_or_history_fields() -> None:
    shard = make_shard()
    forbidden = {"future_token", "future_tokens", "next_token", "previous_lambda", "history", "selected_score"}
    assert forbidden.isdisjoint(shard)
    assert forbidden.isdisjoint(shard["router_feature_fields"])


def test_derived_feature_bounds() -> None:
    result = features.compute_confidence_disagreement(
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[2.0, 0.0], [5.0, 1.0]]),
        torch.tensor([[1, 2], [7, 8]]),
        torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        torch.tensor([0, 4]),
        max_position=4,
    )
    assert torch.isfinite(result.stacked()).all()
    assert torch.all(result.js_divergence >= 0)
    assert torch.all(result.js_divergence <= math.log(2.0) + 1e-6)
    assert torch.all((result.topk_overlap >= 0) & (result.topk_overlap <= 1))
    assert set(result.top1_agreement.tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(result.js_divergence[0], torch.tensor(0.0))
    torch.testing.assert_close(result.topk_overlap, torch.tensor([1.0, 0.0]))


def test_json_atomic_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    io.write_json_atomic(path, {"step": 1})
    with pytest.raises(FileExistsError):
        io.write_json_atomic(path, {"step": 2})
    assert json.loads(path.read_text()) == {"step": 1}
