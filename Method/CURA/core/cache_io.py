from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_manifest(cache_dir: Path):
    path = cache_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_shard(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid CURA shard schema")
    if not isinstance(payload.get("rows"), list):
        raise ValueError("CURA shard rows must be a list")
    required = {
        "prompt_id", "step", "prefix_token_ids", "candidate_token_ids",
        "base_logits", "raw_scores", "signal_mask"
    }
    for row in payload["rows"]:
        missing = required - set(row)
        if missing:
            raise ValueError(f"CURA cache row misses: {sorted(missing)}")
        top_k = row["candidate_token_ids"].numel()
        if row["base_logits"].numel() != top_k or row["raw_scores"].shape[-1] != top_k:
            raise ValueError("CURA cache row has inconsistent candidate dimensions")


def atomic_shard(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    validate_shard(loaded)
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    torch.load(temporary, map_location="cpu", weights_only=False)
    os.replace(temporary, path)


def verify_cache(cache_dir: Path, strict=True):
    manifest = read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"Missing manifest: {cache_dir / 'manifest.json'}")
    errors = []
    for descriptor in manifest.get("completed_shards", []):
        path = cache_dir / descriptor.get("target_file", descriptor["file"])
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            expected_sha = descriptor.get("target_sha256", descriptor["sha256"])
            if sha256_file(path) != expected_sha:
                raise ValueError("checksum mismatch")
            validate_shard(torch.load(path, map_location="cpu", weights_only=False))
        except Exception as error:
            errors.append({"shard": descriptor["id"], "error": f"{type(error).__name__}: {error}"})
    if strict and errors:
        raise RuntimeError(f"Cache verification failed: {errors}")
    return {"valid": not errors, "checked_shards": len(manifest.get("completed_shards", [])), "errors": errors}


class ShardedCuraDataset(Dataset):
    """Map-style shard reader with optional in-memory preloading.

    Lazy mode keeps at most one shard in memory. Training uses ``preload=True``
    because shuffled indices would otherwise reload a shard for nearly every
    sample in a batch.
    """

    def __init__(self, cache_dir, preload=False):
        self.cache_dir = Path(cache_dir)
        self.manifest = read_manifest(self.cache_dir)
        if self.manifest is None:
            raise FileNotFoundError(f"Missing CURA cache manifest in {self.cache_dir}")
        self.index = []
        for descriptor in self.manifest.get("completed_shards", []):
            for local_index in range(int(descriptor["num_steps"])):
                self.index.append((descriptor.get("target_file", descriptor["file"]), local_index))
        self._loaded_name = None
        self._loaded_rows = None
        self._preloaded_rows = {}
        if preload:
            for filename, _ in self.index:
                if filename not in self._preloaded_rows:
                    self._preloaded_rows[filename] = self._load_rows(filename)

    def _load_rows(self, filename):
        payload = torch.load(self.cache_dir / filename, map_location="cpu", weights_only=False)
        validate_shard(payload)
        return payload["rows"]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        filename, local_index = self.index[index]
        if filename in self._preloaded_rows:
            return self._preloaded_rows[filename][local_index]
        if filename != self._loaded_name:
            self._loaded_name, self._loaded_rows = filename, self._load_rows(filename)
        return self._loaded_rows[local_index]
