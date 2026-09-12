"""Prepare PKU SafeRLHF labels without relabeling or modifying source data."""

from __future__ import annotations

import collections
import hashlib
import json
import lzma
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
DATASET_FORMAT = "parm_taro.multi_objective_preference"
DEFAULT_SPLIT_SEED = "parm_taro_pku_v1_seed_2026"
SOURCE_FIELDS = (
    "prompt",
    "response_0",
    "response_1",
    "better_response_id",
    "safer_response_id",
    "is_response_0_safe",
    "is_response_1_safe",
)
PROCESSED_FIELDS = (
    "sample_id",
    "source_index",
    *SOURCE_FIELDS,
    "helpfulness_preferred_response_id",
    "harmlessness_preferred_response_id",
)
SPLIT_FILENAMES = {
    "train": "train.json",
    "validation": "validation.json",
    "test": "test.json",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_source_rows(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    rows: list[dict[str, Any]] = []
    with lzma.open(source_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Source line {line_number} is not an object")
            rows.append(row)
    validate_source_rows(rows)
    return rows


def validate_source_rows(rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("PKU source contains no records")
    for index, row in enumerate(rows):
        missing = sorted(set(SOURCE_FIELDS) - set(row))
        unknown = sorted(set(row) - set(SOURCE_FIELDS))
        if missing or unknown:
            raise ValueError(
                f"Source schema mismatch at row {index}: "
                f"missing={missing}, unknown={unknown}"
            )
        for field in ("prompt", "response_0", "response_1"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"Invalid {field} at source row {index}")
        for field in ("better_response_id", "safer_response_id"):
            if type(row[field]) is not int or row[field] not in (0, 1):
                raise ValueError(f"Invalid {field} at source row {index}")
        for field in ("is_response_0_safe", "is_response_1_safe"):
            if type(row[field]) is not bool:
                raise ValueError(f"Invalid {field} at source row {index}")


def _prompt_order(prompts: Iterable[str], seed: str) -> list[str]:
    def key(prompt: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{prompt}".encode("utf-8")).hexdigest()
        return digest, prompt

    return sorted(prompts, key=key)


def _take_prompt_groups(
    ordered_prompts: list[str],
    groups: Mapping[str, list[int]],
    target_records: int,
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    remaining_count = target_records
    for prompt in ordered_prompts:
        group_size = len(groups[prompt])
        if group_size <= remaining_count:
            selected.append(prompt)
            remaining_count -= group_size
        if remaining_count == 0:
            break
    if remaining_count:
        raise ValueError(
            "Prompt-group split cannot satisfy exact target; "
            f"unfilled records={remaining_count}"
        )
    selected_set = set(selected)
    remaining = [
        prompt for prompt in ordered_prompts if prompt not in selected_set
    ]
    return selected, remaining


def deterministic_prompt_group_split(
    rows: list[Mapping[str, Any]],
    *,
    train_records: int,
    validation_records: int,
    seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, list[int]]:
    if train_records <= 0 or validation_records <= 0:
        raise ValueError("Train and validation targets must be positive")
    if train_records + validation_records >= len(rows):
        raise ValueError("Split targets must leave at least one test record")
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for source_index, row in enumerate(rows):
        groups[str(row["prompt"])].append(source_index)
    ordered = _prompt_order(groups, seed)
    train_prompts, remaining = _take_prompt_groups(
        ordered, groups, train_records
    )
    validation_prompts, test_prompts = _take_prompt_groups(
        remaining, groups, validation_records
    )

    result: dict[str, list[int]] = {}
    for split, prompts in (
        ("train", train_prompts),
        ("validation", validation_prompts),
        ("test", test_prompts),
    ):
        result[split] = sorted(
            source_index
            for prompt in prompts
            for source_index in groups[prompt]
        )
    assigned = [index for values in result.values() for index in values]
    if sorted(assigned) != list(range(len(rows))):
        raise RuntimeError("Split assignment does not cover source exactly once")
    return result


def _processed_record(row: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    return {
        "sample_id": f"pku_round0_{source_index:05d}",
        "source_index": source_index,
        "prompt": row["prompt"],
        "response_0": row["response_0"],
        "response_1": row["response_1"],
        "better_response_id": row["better_response_id"],
        "safer_response_id": row["safer_response_id"],
        "is_response_0_safe": row["is_response_0_safe"],
        "is_response_1_safe": row["is_response_1_safe"],
        "helpfulness_preferred_response_id": row["better_response_id"],
        "harmlessness_preferred_response_id": row["safer_response_id"],
    }


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        handle.write("\n")


def _label_stats(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = collections.Counter(
        (
            int(row["better_response_id"]),
            int(row["safer_response_id"]),
        )
        for row in records
    )
    return {
        "records": len(records),
        "prompt_groups": len({str(row["prompt"]) for row in records}),
        "helpfulness_preferred_response_id": {
            str(value): sum(
                row["better_response_id"] == value for row in records
            )
            for value in (0, 1)
        },
        "harmlessness_preferred_response_id": {
            str(value): sum(
                row["safer_response_id"] == value for row in records
            )
            for value in (0, 1)
        },
        "objective_pair_counts": {
            f"help_{helpfulness}_safe_{harmlessness}": pairs[
                (helpfulness, harmlessness)
            ]
            for helpfulness in (0, 1)
            for harmlessness in (0, 1)
        },
        "objectives_agree": sum(
            row["better_response_id"] == row["safer_response_id"]
            for row in records
        ),
        "objectives_disagree": sum(
            row["better_response_id"] != row["safer_response_id"]
            for row in records
        ),
    }


def _file_record(path: Path, records: int | None) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": records,
    }


def _source_report(
    rows: list[Mapping[str, Any]],
    *,
    source_path: Path,
    parm_data_path: Path,
    relabel_path: Path,
) -> dict[str, Any]:
    prompt_counts = collections.Counter(str(row["prompt"]) for row in rows)
    canonical_rows = [
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for row in rows
    ]
    exact_counts = collections.Counter(canonical_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "inspection_order": [
            str(source_path.resolve()),
            str(parm_data_path.resolve()),
        ],
        "selected_priority": "B_local_GenARM_source",
        "download_performed": False,
        "relabel_performed": False,
        "source_sufficient": True,
        "source": {
            "path": str(source_path.resolve()),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "format": "xz_compressed_json_lines",
            "records": len(rows),
            "fields": list(SOURCE_FIELDS),
            "missing_values": {field: 0 for field in SOURCE_FIELDS},
            "label_statistics": _label_stats(rows),
            "unique_prompts": len(prompt_counts),
            "duplicate_prompt_values": sum(
                count > 1 for count in prompt_counts.values()
            ),
            "rows_in_duplicate_prompt_groups": sum(
                count for count in prompt_counts.values() if count > 1
            ),
            "max_prompt_multiplicity": max(prompt_counts.values()),
            "unique_exact_rows": len(exact_counts),
            "duplicate_exact_values": sum(
                count > 1 for count in exact_counts.values()
            ),
            "exact_duplicate_rows_preserved": True,
        },
        "parm_author_data_inventory": {
            "path": str(parm_data_path.resolve()),
            "train_json_present": (parm_data_path / "train.json").is_file(),
            "dev_json_present": (parm_data_path / "dev.json").is_file(),
            "test_json_present": (parm_data_path / "test.json").is_file(),
            "test_prompt_only_json_present": (
                parm_data_path / "test_prompt_only.json"
            ).is_file(),
            "relabel_script": str(relabel_path.resolve()),
            "relabel_script_sha256": sha256_file(relabel_path),
            "author_shuffle_seeded": False,
            "author_relabel_requires_external_reward_models": True,
        },
        "objective_semantics": {
            "helpfulness_source_field": "better_response_id",
            "harmlessness_source_field": "safer_response_id",
            "response_safety_fields": [
                "is_response_0_safe",
                "is_response_1_safe",
            ],
            "scores_available_in_local_source": False,
            "scores_synthesized": False,
            "named_order_in_processed_data": [
                "helpfulness",
                "harmlessness",
            ],
            "parm_author_pblora_vector_order": [
                "harmlessness",
                "helpfulness",
            ],
        },
    }


def _assert_prompt_isolation(
    split_records: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    prompt_sets = {
        split: {str(row["prompt"]) for row in records}
        for split, records in split_records.items()
    }
    names = tuple(prompt_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = prompt_sets[first] & prompt_sets[second]
            if overlap:
                raise ValueError(
                    f"Prompt leakage between {first} and {second}: {len(overlap)}"
                )


def prepare_multi_objective_dataset(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    parm_data_path: str | Path,
    train_records: int = 8000,
    validation_records: int = 500,
    seed: str = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    output = Path(output_dir).resolve()
    parm_data = Path(parm_data_path).resolve()
    relabel_path = parm_data / "relabel.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing local PKU source: {source}")
    if not relabel_path.is_file():
        raise FileNotFoundError(f"Missing original PARM relabel script: {relabel_path}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite processed data: {output}")
    source_hash_before = sha256_file(source)
    relabel_hash_before = sha256_file(relabel_path)
    rows = load_source_rows(source)
    assignment = deterministic_prompt_group_split(
        rows,
        train_records=train_records,
        validation_records=validation_records,
        seed=seed,
    )
    processed = [_processed_record(row, index) for index, row in enumerate(rows)]
    split_records = {
        split: [processed[index] for index in indices]
        for split, indices in assignment.items()
    }
    _assert_prompt_isolation(split_records)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    backup: Path | None = None
    try:
        for split, filename in SPLIT_FILENAMES.items():
            _json_dump(temporary / filename, split_records[split])
        test_prompt_only = [
            {
                "uid": f"parm_taro_test_{ordinal:05d}",
                "sample_id": row["sample_id"],
                "source_index": row["source_index"],
                "prompt": row["prompt"],
            }
            for ordinal, row in enumerate(split_records["test"])
        ]
        _json_dump(temporary / "test_prompt_only.json", test_prompt_only)
        source_report = _source_report(
            rows,
            source_path=source,
            parm_data_path=parm_data,
            relabel_path=relabel_path,
        )
        _json_dump(temporary / "source_report.json", source_report)

        assignment_records = sorted(
            (source_index, split)
            for split, indices in assignment.items()
            for source_index in indices
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": DATASET_FORMAT,
            "task": "PKU-SafeRLHF multi-objective preference",
            "source": source_report["source"],
            "processing": {
                "download_performed": False,
                "relabel_performed": False,
                "split_method": "seeded_sha256_prompt_group_partition",
                "split_seed": seed,
                "source_order_preserved_within_splits": True,
                "prompt_group_isolation": True,
                "exact_source_rows_preserved": True,
            },
            "schema": {
                "fields": list(PROCESSED_FIELDS),
                "helpfulness_label": "helpfulness_preferred_response_id",
                "harmlessness_label": "harmlessness_preferred_response_id",
                "original_helpfulness_label": "better_response_id",
                "original_harmlessness_label": "safer_response_id",
                "scores_available": False,
            },
            "objective_semantics": source_report["objective_semantics"],
            "splits": {
                split: _label_stats(records)
                for split, records in split_records.items()
            },
            "total_records": len(rows),
            "split_assignment_sha256": canonical_sha256(assignment_records),
            "source_sample_ids_sha256": canonical_sha256(
                [row["sample_id"] for row in processed]
            ),
            "files": {},
            "checks": {
                "source_schema_valid": True,
                "source_coverage_exactly_once": True,
                "sample_ids_unique": True,
                "prompt_overlap_across_splits": False,
                "test_prompt_only_matches_test": True,
                "source_unchanged": False,
                "parm_relabel_script_unchanged": False,
            },
        }
        for split, filename in SPLIT_FILENAMES.items():
            manifest["files"][filename] = _file_record(
                temporary / filename, len(split_records[split])
            )
        manifest["files"]["test_prompt_only.json"] = _file_record(
            temporary / "test_prompt_only.json", len(test_prompt_only)
        )
        manifest["files"]["source_report.json"] = _file_record(
            temporary / "source_report.json", None
        )
        manifest["checks"]["source_unchanged"] = (
            sha256_file(source) == source_hash_before
        )
        manifest["checks"]["parm_relabel_script_unchanged"] = (
            sha256_file(relabel_path) == relabel_hash_before
        )
        if not all(
            value is True
            for key, value in manifest["checks"].items()
            if key != "prompt_overlap_across_splits"
        ) or manifest["checks"]["prompt_overlap_across_splits"] is not False:
            raise RuntimeError("Prepared dataset invariants failed")
        _json_dump(temporary / "manifest.json", manifest)
        if output.exists():
            backup = output.with_name(f".{output.name}.backup")
            if backup.exists():
                raise FileExistsError(f"Refusing to replace existing backup: {backup}")
            os.replace(output, backup)
        os.replace(temporary, output)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
            backup = None
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_processed_dataset(output, source_path=source)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_processed_dataset(
    output_dir: str | Path,
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    manifest = _load_json(output / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported processed dataset schema_version")
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("Invalid processed dataset format")
    split_records = {
        split: _load_json(output / filename)
        for split, filename in SPLIT_FILENAMES.items()
    }
    for filename, descriptor in manifest["files"].items():
        path = output / filename
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"File hash mismatch: {filename}")
        if path.stat().st_size != descriptor["bytes"]:
            raise ValueError(f"File size mismatch: {filename}")
        expected_records = descriptor["records"]
        if filename in SPLIT_FILENAMES.values():
            split = next(
                name for name, value in SPLIT_FILENAMES.items() if value == filename
            )
            if expected_records != len(split_records[split]):
                raise ValueError(f"Manifest record count mismatch: {filename}")
    _assert_prompt_isolation(split_records)
    all_records = [row for records in split_records.values() for row in records]
    if len(all_records) != manifest["total_records"]:
        raise ValueError("Processed record count mismatch")
    sample_ids = [row["sample_id"] for row in all_records]
    source_indices = [int(row["source_index"]) for row in all_records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate processed sample IDs")
    if sorted(source_indices) != list(range(len(all_records))):
        raise ValueError("Source indices are missing or duplicated")
    for row in all_records:
        if tuple(row) != PROCESSED_FIELDS:
            raise ValueError("Processed record schema mismatch")
        if row["helpfulness_preferred_response_id"] != row["better_response_id"]:
            raise ValueError("Helpfulness label alias mismatch")
        if row["harmlessness_preferred_response_id"] != row["safer_response_id"]:
            raise ValueError("Harmlessness label alias mismatch")
    assignment_records = sorted(
        (int(row["source_index"]), split)
        for split, records in split_records.items()
        for row in records
    )
    if canonical_sha256(assignment_records) != manifest["split_assignment_sha256"]:
        raise ValueError("Split assignment hash mismatch")
    records_by_source = sorted(all_records, key=lambda row: row["source_index"])
    if canonical_sha256(
        [row["sample_id"] for row in records_by_source]
    ) != manifest["source_sample_ids_sha256"]:
        raise ValueError("Source sample ID hash mismatch")
    for split, records in split_records.items():
        if _label_stats(records) != manifest["splits"][split]:
            raise ValueError(f"Split statistics mismatch: {split}")
    prompts = _load_json(output / "test_prompt_only.json")
    test = split_records["test"]
    expected_prompt_rows = [
        (row["sample_id"], row["source_index"], row["prompt"]) for row in test
    ]
    actual_prompt_rows = [
        (row["sample_id"], row["source_index"], row["prompt"])
        for row in prompts
    ]
    if actual_prompt_rows != expected_prompt_rows:
        raise ValueError("test_prompt_only does not match the test split")
    expected_uids = [
        f"parm_taro_test_{ordinal:05d}" for ordinal in range(len(test))
    ]
    if [row["uid"] for row in prompts] != expected_uids:
        raise ValueError("test_prompt_only UIDs are not deterministic")
    if source_path is not None:
        source = Path(source_path).resolve()
        if sha256_file(source) != manifest["source"]["sha256"]:
            raise ValueError("Source hash changed after materialization")
        source_rows = load_source_rows(source)
        if len(source_rows) != len(records_by_source):
            raise ValueError("Processed/source row count mismatch")
        for source_index, (source_row, processed_row) in enumerate(
            zip(source_rows, records_by_source)
        ):
            if processed_row["source_index"] != source_index:
                raise ValueError("Processed records are not source-index aligned")
            if any(
                processed_row[field] != source_row[field]
                for field in SOURCE_FIELDS
            ):
                raise ValueError(
                    f"Processed record differs from source row {source_index}"
                )
    return {
        "status": "PASS",
        "output_dir": str(output),
        "total_records": len(all_records),
        "split_counts": {
            split: len(records) for split, records in split_records.items()
        },
        "prompt_overlap_across_splits": False,
        "sample_ids_unique": True,
        "file_hashes_valid": True,
        "source_hash_valid": source_path is not None,
    }
