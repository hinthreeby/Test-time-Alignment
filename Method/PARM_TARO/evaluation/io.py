"""Atomic Stage 10 output helpers and resumable JSONL journals."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(row) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("Incomplete Stage 10 journal append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write any JSON value; protocol locks remain no-overwrite."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite Stage 10 JSON: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite Stage 10 JSON: {output}")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path: str | Path, *, recover_trailing_partial: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    raw = source.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        complete = line.endswith(b"\n")
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if recover_trailing_partial and index == len(lines) - 1 and not complete:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {index + 1} is not an object")
        rows.append(value)
        valid_bytes += len(line)
    if recover_trailing_partial and valid_bytes != len(raw):
        with source.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def unique_rows(rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in output:
            raise ValueError(f"Duplicate Stage 10 resume key: {key}")
        output[key] = row
    return output


__all__ = [
    "append_jsonl", "canonical_json", "read_jsonl", "unique_rows", "write_json_atomic"
]
