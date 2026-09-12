"""Read-only V1 result import and provenance validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from router_v2.cache.io import sha256_file
from router_v2.evaluation.rad.data import PromptRecord


V1_METHOD_MAP = {
    "base_lm": "v1_base",
    "best_fixed_beta": "v1_fixed",
    "best_heuristic": "v1_heuristic",
    "learned_router": "v1_router",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def audit_v1_results(
    results_dir: str | Path,
    *,
    project_root: Path,
    test_prompt_offset_per_class: int,
    test_prompt_limit_per_class: int,
    max_new_tokens: int,
    sentiment_classifier_path: str | Path,
    seeds: Sequence[int],
) -> dict[str, Any]:
    root = Path(results_dir).resolve()
    config_path = root / "config.json"
    report_path = root / "stage7_report.json"
    outputs_path = root / "per_sample_outputs.jsonl"
    for path in (config_path, report_path, outputs_path):
        if not path.is_file():
            raise FileNotFoundError(f"V1 baseline artifact is absent: {path}")
    config = _load_json(config_path)
    report = _load_json(report_path)
    base_model_path = Path(config["base_model_path"])
    if not base_model_path.is_absolute():
        base_model_path = project_root / base_model_path
    base_model_config_path = base_model_path / "config.json"
    base_model_config = (
        _load_json(base_model_config_path)
        if base_model_config_path.is_file()
        else {}
    )
    base_model_eos_token_id = base_model_config.get("eos_token_id")
    checks = {
        "max_new_tokens_match": (
            int(config["decoding"]["max_new_tokens"]) == max_new_tokens
        ),
        "test_limit_covers_request": (
            int(config["test_prompt_limit_per_class"])
            >= test_prompt_limit_per_class
        ),
        "test_offset_matches": (
            int(config["validation_prompt_limit_per_class"])
            == test_prompt_offset_per_class
        ),
        "requested_seeds_available": set(int(value) for value in seeds)
        <= set(int(value) for value in config["seeds"]),
        "generation_failures_zero": (
            int(report["sample_counts"]["failed_generations"]) == 0
        ),
        "base_model_eos_token_id_available": isinstance(
            base_model_eos_token_id,
            int,
        ),
    }
    source_hashes = report["reproducibility"]["hashes"].get(
        "original_fixed_rad_source", {}
    )
    source_checks: dict[str, bool] = {}
    for recorded_path, expected_hash in source_hashes.items():
        path = Path(recorded_path)
        if not path.is_absolute():
            path = project_root / path
        source_checks[str(path.resolve())] = (
            path.is_file() and sha256_file(path) == expected_hash
        )
    checks["v1_source_hashes_match"] = bool(source_checks) and all(
        source_checks.values()
    )
    checkpoint_path = Path(config["router_checkpoint_path"])
    expected_checkpoint_hash = report["reproducibility"]["hashes"].get(
        "router_checkpoint_sha256"
    )
    checks["v1_checkpoint_hash_matches"] = (
        checkpoint_path.is_file()
        and expected_checkpoint_hash is not None
        and sha256_file(checkpoint_path) == expected_checkpoint_hash
    )
    v1_load_error = None
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        from router.rad.router_integration import load_router_for_inference

        router = load_router_for_inference(
            checkpoint_path,
            beta_max=float(config["decoding"]["beta_max"]),
            top_k=int(config["decoding"]["top_k"]),
            map_location="cpu",
        )
        base_probe = torch.linspace(2.0, -1.0, 20).unsqueeze(0)
        reward_probe = torch.linspace(0.0, 1.0, 20).unsqueeze(0)
        with torch.inference_mode():
            beta_probe, gate_probe = router(base_probe, reward_probe)
        checks["v1_checkpoint_load_and_forward"] = (
            beta_probe.shape == (1, 1)
            and gate_probe.shape == (1, 1)
            and bool(torch.isfinite(beta_probe).all())
            and bool(torch.isfinite(gate_probe).all())
            and all(not parameter.requires_grad for parameter in router.parameters())
        )
    except Exception as error:  # recorded as audit evidence, then fails closed
        v1_load_error = f"{type(error).__name__}: {error}"
        checks["v1_checkpoint_load_and_forward"] = False
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    classifier_checkpoint = Path(sentiment_classifier_path) / "pytorch_model.bin"
    expected_classifier_hash = report["reproducibility"]["hashes"].get(
        "sentiment_classifier_checkpoint_sha256"
    )
    checks["sentiment_classifier_hash_matches"] = (
        classifier_checkpoint.is_file()
        and expected_classifier_hash is not None
        and sha256_file(classifier_checkpoint) == expected_classifier_hash
    )
    return {
        "results_dir": str(root),
        "config_path": str(config_path),
        "report_path": str(report_path),
        "outputs_path": str(outputs_path),
        "base_model_eos_token_id": base_model_eos_token_id,
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "report_sha256": sha256_file(report_path),
            "outputs_sha256": sha256_file(outputs_path),
            "base_model_config_sha256": (
                sha256_file(base_model_config_path)
                if base_model_config_path.is_file()
                else None
            ),
            "router_checkpoint_sha256": (
                sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
            ),
            "sentiment_classifier_checkpoint_sha256": (
                sha256_file(classifier_checkpoint)
                if classifier_checkpoint.is_file()
                else None
            ),
        },
        "source_hash_checks": source_checks,
        "v1_validation_selection": report.get("best_validation_configuration"),
        "v1_load_error": v1_load_error,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _compact_v1_record(
    row: Mapping[str, Any],
    *,
    mapped_method: str,
    prompt: PromptRecord,
    eos_token_id: int,
) -> dict[str, Any]:
    beta_max = float(row.get("beta_max", 30.0))
    if beta_max <= 0.0:
        raise ValueError("V1 beta_max must be positive")
    beta_history = [float(value) for value in row.get("beta_history", [])]
    normalized = [min(max(value / beta_max, 0.0), 1.0) for value in beta_history]
    latency = dict(row["latency"])
    selected_token_ids = [int(value) for value in row["selected_token_ids"]]
    source_eos_generated = bool(row.get("eos_generated", False))
    terminated_on_eos = bool(
        selected_token_ids and selected_token_ids[-1] == eos_token_id
    )
    return {
        "prompt_id": prompt.prompt_id,
        "source_prompt_id": str(row["prompt_id"]),
        "source_index": prompt.source_index,
        "prompt": str(row["prompt"]),
        "prompt_sentiment_class": str(row["prompt_sentiment_class"]),
        "method": mapped_method,
        "legacy_method": str(row["method"]),
        "method_family": "v1_rad_scalar_reward",
        "ppl_model_family": "v1_gpt2_large_full_sequence",
        "ppl_protocol": "legacy_prompt_plus_generation_full_sequence",
        "seed": int(row["seed"]),
        "generated_text": str(row["generated_text"]),
        "selected_token_ids": selected_token_ids,
        "lambda_history": normalized,
        "raw_beta_history": beta_history,
        "perplexity": float(row["base_lm_perplexity"]),
        "classifier_sentiment_success": int(
            row["classifier_sentiment_success"]
        ),
        "classifier_target_probability": float(
            row["classifier_target_probability"]
        ),
        "classifier_predicted_label": str(row["classifier_predicted_label"]),
        "classifier_predicted_label_id": int(
            row["classifier_predicted_label_id"]
        ),
        "eos_generated": terminated_on_eos,
        "terminated_on_eos": terminated_on_eos,
        "source_eos_generated": source_eos_generated,
        "eos_metadata_mismatch": source_eos_generated != terminated_on_eos,
        "latency": {
            "total_generation_time": float(latency["total_generation_time"]),
            "average_latency_per_token": float(
                latency["average_latency_per_token"]
            ),
            "generated_tokens": len(selected_token_ids),
        },
        "failure_status": str(row.get("failure_status", "success")),
        "source": "read_only_v1_output",
    }


def import_v1_records(
    outputs_path: str | Path,
    *,
    prompts: Sequence[PromptRecord],
    seeds: Sequence[int],
    eos_token_id: int,
) -> list[dict[str, Any]]:
    if eos_token_id < 0:
        raise ValueError("V1 eos_token_id must be non-negative")
    prompts_by_source_identity: dict[tuple[str, str, str], PromptRecord] = {}
    for prompt in prompts:
        source_identity = (
            prompt.source_prompt_id or prompt.prompt_id,
            prompt.prompt_sentiment_class,
            prompt.prompt,
        )
        if source_identity in prompts_by_source_identity:
            raise ValueError(f"Ambiguous V1 source prompt identity: {source_identity}")
        prompts_by_source_identity[source_identity] = prompt
    target_keys = {
        (prompt.prompt_id, int(seed), old_method)
        for prompt in prompts
        for seed in seeds
        for old_method in V1_METHOD_MAP
    }
    imported: dict[tuple[str, int, str], dict[str, Any]] = {}
    with Path(outputs_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            old_method = str(row.get("method"))
            if old_method not in V1_METHOD_MAP:
                continue
            source_identity = (
                str(row.get("prompt_id")),
                str(row.get("prompt_sentiment_class")),
                str(row.get("prompt")),
            )
            prompt = prompts_by_source_identity.get(source_identity)
            if prompt is None:
                continue
            key = (
                prompt.prompt_id,
                int(row.get("seed", -1)),
                old_method,
            )
            if key not in target_keys:
                continue
            if key in imported:
                raise ValueError(f"Duplicate V1 result record: {key}")
            imported[key] = _compact_v1_record(
                row,
                mapped_method=V1_METHOD_MAP[key[2]],
                prompt=prompt,
                eos_token_id=eos_token_id,
            )
    missing = sorted(target_keys - set(imported))
    if missing:
        raise ValueError(f"V1 baseline is missing {len(missing)} records: {missing[:3]}")
    return [imported[key] for key in sorted(imported)]
