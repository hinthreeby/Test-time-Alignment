"""Audit Stage 9 response-loss spans over all train/validation records."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import (
    SequenceTooLongError,
    inspect_response_boundary,
    load_examples,
    tokenize_response,
)
from PARM_TARO.training.runtime import PROJECT_ROOT, project_path
from router_v2.cache.io import write_json_atomic
from router_v2.guide_model.provenance import tokenizer_descriptor


DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_tokenization_boundary_audit.json"
)


def _audit_split(
    tokenizer: Any,
    config: ParmRouterTrainingConfig,
    split: str,
) -> dict[str, Any]:
    examples = load_examples(project_path(config.data_root), split)
    affected: list[dict[str, Any]] = []
    unexpected_errors: list[dict[str, str | int]] = []
    boundary_isolation_failures: list[dict[str, str | int]] = []
    too_long: list[dict[str, str | int]] = []
    tokenized_responses = 0
    for example in examples:
        for response_index in (0, 1):
            diagnostic = None
            try:
                tokenized = tokenize_response(
                    tokenizer,
                    example,
                    response_index,
                    max_length=config.max_length,
                    max_continuation_tokens=config.max_continuation_tokens,
                    include_eos_target=config.include_eos_target,
                )
                diagnostic = tokenized.boundary_diagnostic
                tokenized_responses += 1
                prefix_shift = int(tokenizer.bos_token_id is not None)
                target_sequence_indices = {
                    int(position) + 1
                    for position in tokenized.logit_positions.tolist()
                }
                for crossing in diagnostic.crossing_tokens:
                    sequence_index = crossing.token_index + prefix_shift
                    crossing_is_context_only = (
                        sequence_index < tokenized.input_ids.shape[1]
                        and int(tokenized.input_ids[0, sequence_index])
                        == crossing.token_id
                        and sequence_index not in target_sequence_indices
                    )
                    if not crossing_is_context_only:
                        boundary_isolation_failures.append(
                            {
                                "sample_id": example.sample_id,
                                "response_index": response_index,
                                "token_index": crossing.token_index,
                                "token_id": crossing.token_id,
                            }
                        )
            except SequenceTooLongError as error:
                too_long.append(
                    {
                        "sample_id": example.sample_id,
                        "response_index": response_index,
                        "error": str(error),
                    }
                )
                diagnostic = inspect_response_boundary(
                    tokenizer, example, response_index
                )
            except Exception as error:
                unexpected_errors.append(
                    {
                        "sample_id": example.sample_id,
                        "response_index": response_index,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if diagnostic is not None and diagnostic.crossing_tokens:
                affected.append(diagnostic.to_dict())
    affected_sample_ids = sorted(
        {str(item["sample_id"]) for item in affected}
    )
    response_count = len(examples) * 2
    return {
        "examples": len(examples),
        "responses": response_count,
        "tokenized_responses": tokenized_responses,
        "expected_sequence_too_long": len(too_long),
        "unexpected_error_count": len(unexpected_errors),
        "unexpected_errors": unexpected_errors,
        "boundary_isolation_failure_count": len(
            boundary_isolation_failures
        ),
        "boundary_isolation_failures": boundary_isolation_failures,
        "affected_examples": len(affected_sample_ids),
        "affected_example_fraction": len(affected_sample_ids) / len(examples),
        "affected_responses": len(affected),
        "affected_response_fraction": len(affected) / response_count,
        "boundary_tokens_excluded": sum(
            int(item["boundary_tokens_excluded"]) for item in affected
        ),
        "affected": affected,
        "sequence_too_long_records": too_long,
        "pass": not unexpected_errors and not boundary_isolation_failures,
    }


def audit(config: ParmRouterTrainingConfig) -> dict[str, Any]:
    tokenizer_path = project_path(config.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Stage 9 tokenization audit requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = {
        split: _audit_split(tokenizer, config, split)
        for split in ("train", "validation")
    }
    checks = {
        "concatenated_tokenization_used": True,
        "crossing_tokens_preserved_as_context_and_excluded_from_loss": all(
            split["boundary_isolation_failure_count"] == 0
            for split in splits.values()
        ),
        "train_has_no_unexpected_tokenization_errors": splits["train"]["pass"],
        "validation_has_no_unexpected_tokenization_errors": splits[
            "validation"
        ]["pass"],
    }
    return {
        "schema_version": 1,
        "stage": 9,
        "method_label": "PARM_TARO_TOKENIZATION_BOUNDARY_AUDIT",
        "config": {
            "data_root": str(project_path(config.data_root)),
            "tokenizer_path": str(tokenizer_path),
            "max_length": config.max_length,
            "max_continuation_tokens": config.max_continuation_tokens,
            "include_eos_target": config.include_eos_target,
        },
        "tokenizer": tokenizer_descriptor(tokenizer, tokenizer_path),
        "splits": splits,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "NOT_PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = replace(
        ParmRouterTrainingConfig.load_json(args.config),
        device="cpu",
        allow_cpu_fallback=True,
    )
    output = args.output.resolve()
    reports_root = (PROJECT_ROOT / "PARM_TARO/reports").resolve()
    try:
        output.relative_to(reports_root)
    except ValueError as error:
        raise ValueError(
            "Tokenization audit must stay under PARM_TARO/reports"
        ) from error
    report = audit(config)
    write_json_atomic(output, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
