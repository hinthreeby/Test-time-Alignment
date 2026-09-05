"""Deterministic model/tokenizer provenance for guide validation and caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from router_v2.cache.io import sha256_file


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_descriptor(tokenizer: Any, path: str | Path) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    sorted_vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in vocabulary.items()
    )
    descriptor = {
        "path": str(Path(path).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "length": len(tokenizer),
        "vocab_entries": len(vocabulary),
        "vocab_sha256": _canonical_sha256(sorted_vocabulary),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": tokenizer.padding_side,
        "model_max_length": int(tokenizer.model_max_length),
    }
    descriptor["semantic_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in descriptor.items()
            if key not in {"path", "semantic_sha256"}
        }
    )
    return descriptor


def tokenizers_exactly_compatible(
    base_tokenizer: Any,
    guide_tokenizer: Any,
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "get_vocab_equal": (
            base_tokenizer.get_vocab() == guide_tokenizer.get_vocab()
        ),
        "length_equal": len(base_tokenizer) == len(guide_tokenizer),
        "eos_token_id_equal": (
            base_tokenizer.eos_token_id == guide_tokenizer.eos_token_id
        ),
        "bos_token_id_equal": (
            base_tokenizer.bos_token_id == guide_tokenizer.bos_token_id
        ),
        "pad_token_id_equal": (
            base_tokenizer.pad_token_id == guide_tokenizer.pad_token_id
        ),
        "padding_side_equal": (
            base_tokenizer.padding_side == guide_tokenizer.padding_side
        ),
    }
    probe_ids = sorted(
        {
            0,
            1,
            2,
            max(0, len(base_tokenizer) // 2),
            max(0, len(base_tokenizer) - 1),
        }
    )
    checks["probe_decoding_equal"] = all(
        base_tokenizer.decode([token_id])
        == guide_tokenizer.decode([token_id])
        for token_id in probe_ids
    )
    return all(checks.values()), checks


def _weight_files(model_path: Path) -> list[Path]:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_path / index_name
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            names = sorted(set(index.get("weight_map", {}).values()))
            files = [model_path / name for name in names]
            if not files or any(not path.exists() for path in files):
                raise FileNotFoundError(
                    f"Incomplete checkpoint index at {index_path}"
                )
            return [index_path, *files]
    for name in (
        "adapter_model.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        path = model_path / name
        if path.exists():
            config_name = (
                "adapter_config.json"
                if name.startswith("adapter_")
                else "config.json"
            )
            files = [path]
            config_path = model_path / config_name
            if config_path.exists():
                files.insert(0, config_path)
            return files
    raise FileNotFoundError(f"No supported checkpoint weights in {model_path}")


def checkpoint_descriptor(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).resolve()
    records = []
    for file_path in _weight_files(model_path):
        records.append(
            {
                "path": file_path.name,
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return {
        "path": str(model_path),
        "files": records,
        "checkpoint_sha256": _canonical_sha256(records),
    }
