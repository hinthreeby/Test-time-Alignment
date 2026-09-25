"""Regression tests for resume-safe token-headroom score bookkeeping."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from PARM_TARO.adaptive_parm.token_headroom import scoring


def test_summary_uses_canonical_provenance_fields() -> None:
    summary = scoring._build_scoring_summary(
        kind="reward",
        model_path=Path("/models/reward"),
        rollout_count=5,
        scores={"a": 1.25, "b": -0.5},
        memory={"before": {"free_mib": 10000}},
        resolution={
            "path": "/models/safe-rlhf-source",
            "git_commit": "abc123",
            "provenance_status": "EXACT_COMMIT_RECOVERED",
        },
    )
    assert summary["safe_rlhf_source"] == "/models/safe-rlhf-source"
    assert summary["safe_rlhf_commit"] == "abc123"
    assert summary["safe_rlhf_provenance"] == "EXACT_COMMIT_RECOVERED"
    assert summary["unique_responses"] == 2
    assert summary["finite"] is True


def test_summary_tolerates_missing_optional_provenance_without_fabricating() -> None:
    summary = scoring._build_scoring_summary(
        kind="cost",
        model_path=Path("/models/cost"),
        rollout_count=1,
        scores={"a": 0.0},
        memory={},
        resolution={"path": "/recovered/runtime"},
    )
    assert summary["safe_rlhf_source"] == "/recovered/runtime"
    assert summary["safe_rlhf_commit"] is None
    assert summary["safe_rlhf_provenance"] is None


def test_resume_with_complete_state_skips_model_and_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    state_root = output / "score_state"
    reward_state = state_root / "reward"
    reward_state.mkdir(parents=True)
    rollouts = [
        {"rollout_id": "r1", "state_id": "s1", "response_key": "key-a", "prompt": "p", "response": "x"},
        {"rollout_id": "r2", "state_id": "s2", "response_key": "key-a", "prompt": "p", "response": "x"},
        {"rollout_id": "r3", "state_id": "s3", "response_key": "key-b", "prompt": "q", "response": "y"},
    ]
    (output / "counterfactual_rollouts.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (output / "counterfactual_rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts), encoding="utf-8"
    )
    for key, value in (("key-a", 1.5), ("key-b", -0.25)):
        (reward_state / f"{key}.json").write_text(
            json.dumps({"response_key": key, "score": value}), encoding="utf-8"
        )

    monkeypatch.setattr(scoring, "OUT", output)
    monkeypatch.setattr(scoring, "SCORE_STATE", state_root)
    monkeypatch.setattr(scoring, "REWARD", Path("/models/reward"))
    monkeypatch.setattr(scoring, "get_gpu_status", lambda: {"gpu": "not-used"})
    monkeypatch.setattr(
        scoring,
        "resolve_safe_rlhf_source",
        lambda: {"ready": True, "path": "/runtime", "git_commit": None, "provenance_status": "RECOVERED"},
    )
    monkeypatch.setattr(scoring, "_load", lambda *_args, **_kwargs: pytest.fail("resume attempted to load scorer"))
    monkeypatch.setattr(scoring, "require_free_vram", lambda *_args, **_kwargs: pytest.fail("resume attempted GPU preflight"))

    summary = scoring.run_scoring("reward", resume=True)
    assert summary["status"] == "PASS"
    assert summary["rollouts"] == 3
    assert summary["unique_responses"] == 2
    assert summary["safe_rlhf_source"] == "/runtime"
    with (output / "reward_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["response_key"] for row in rows} == {"key-a", "key-b"}
    assert (output / "reward_scoring_summary.json").is_file()
