#!/usr/bin/env python3
"""Read-only completeness checks for pinned PARM-TARO public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    configured = os.environ.get("TTA_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / ".git").exists():
            return root
        raise RuntimeError(f"TTA_PROJECT_ROOT is not a project root: {root}")
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Cannot locate project root containing .git")


ROOT = find_project_root(Path(__file__).resolve().parent)

ARTIFACTS: dict[str, dict[str, Any]] = {
    "tulu": {
        "repo_id": "allenai/tulu-2-7b",
        "revision": "3c6e328ae91fabdd0daf09de16887de9615c1f66",
        "target": "models/tulu-2-7b",
        "index": "pytorch_model.bin.index.json",
        "required": {
            "config.json": (583, None),
            "pytorch_model.bin.index.json": (23950, "e572e08c4d4e81c7916197f6fcd2956a2f05e5919f28d72c9ba4f351efae1e29"),
            "pytorch_model-00001-of-00002.bin": (9976620122, "6d90e5350a50e3a1ae608eadf4de08d5336b387561c44094e558789c0e1480d6"),
            "pytorch_model-00002-of-00002.bin": (3500310787, "3eb1b1833fa5b5b8bc9f0ffde237012fead0c18536047fa4c2fcfb8abc3c6425"),
            "tokenizer.model": (499723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
            "tokenizer_config.json": (1058, None),
            "special_tokens_map.json": (330, None),
        },
    },
    "beaver_reward": {
        "repo_id": "PKU-Alignment/beaver-7b-v1.0-reward",
        "revision": "375cd6a9f0d7e339d2199b05ba129a4a8906596d",
        "target": "models/beaver-7b-v1.0-reward",
        "expected_tree_sha256": "3410256f542b2e06225d2652102d5dff1ce356a4cd092249643ab03a999a47b3",
        "index": "model.safetensors.index.json",
        "required": {
            "config.json": (795, None), "model.safetensors.index.json": (24189, None),
            "tokenizer.json": (1842945, None), "tokenizer_config.json": (1102, None),
            "model-00001-of-00007.safetensors": (1981886992, "4267db7a60cf00f6d9d5b1302a1cccda259c99ef1301db01e99a51f1e9d7c06d"),
            "model-00002-of-00007.safetensors": (1990284240, "420f191a93af94520864c4033b465b24b6c9579ad079945b3970f47bad7fefce"),
            "model-00003-of-00007.safetensors": (1990284280, "c080daa129008641fa829d70c6461eab177587ec228e5a0e9030a5f1a0c77878"),
            "model-00004-of-00007.safetensors": (1990284288, "566c9611763038fc5a364a23a6f93ebcb1bb5d340964189d390c19d37541a632"),
            "model-00005-of-00007.safetensors": (1933644568, "7e8cdfe3ba1541f3c97640e4bb484872de5a8cf57a158fef46e02e47d28986a9"),
            "model-00006-of-00007.safetensors": (1933661176, "a752b653f30fa527146a16eb5ea05f3bdf558feb35a9dbde6192ba3ecf206ea7"),
            "model-00007-of-00007.safetensors": (1394692118, "994b7359787208059f3d0da2c6d5d1223311dc63af4e5d6fb14c05964485c7b1"),
        },
    },
    "beaver_cost": {
        "repo_id": "PKU-Alignment/beaver-7b-v1.0-cost",
        "revision": "c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28",
        "target": "models/beaver-7b-v1.0-cost",
        "expected_tree_sha256": "418b941682ced967d71194c059beb967ebd225c2b6713a713ccdcea1fa6d8454",
        "index": "model.safetensors.index.json",
        "required": {
            "config.json": (795, None), "model.safetensors.index.json": (24189, None),
            "tokenizer.json": (1842945, None), "tokenizer_config.json": (1102, None),
            "model-00001-of-00007.safetensors": (1981886992, "be090bf3f8466a74abfb7451464d91d3731158126d613ed13d39a2c10bb0782c"),
            "model-00002-of-00007.safetensors": (1990284240, "e13c7d20b7be6b5a9b21942ad5bf8019bef27a9e8d7e79b953fe1842e8f978a4"),
            "model-00003-of-00007.safetensors": (1990284280, "c5798701afd4a48d6371096e4f157d0340374cdb65d464df4e59a373ee743c27"),
            "model-00004-of-00007.safetensors": (1990284288, "e90fd5675b45666eca9fe4da21c686fe0099dae2aca3595bb1ce7c83afd1af80"),
            "model-00005-of-00007.safetensors": (1933644568, "4b2c255b57b22d1f1a437e2551b759db20ad5c1648205aa43e3185035007a2b9"),
            "model-00006-of-00007.safetensors": (1933661176, "0eac6e1724b3ac4c761ee82c5d8752ba3d5366703f65237aef62e1e42691c7f3"),
            "model-00007-of-00007.safetensors": (1394692118, "d75da02acfeae64926495bfc20240ad32ce1192abd3379be06f441145c2bc587"),
        },
    },
    "safe_rlhf": {
        "repo_id": "https://github.com/PKU-Alignment/safe-rlhf.git",
        "revision": None,
        "target": "models/safe-rlhf-source",
        "required": {"pyproject.toml": (None, None), "safe_rlhf/__init__.py": (None, None)},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: Path) -> str:
    rows = []
    for item in sorted((p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts and not p.name.startswith(".tta_")), key=lambda p: p.relative_to(path).as_posix()):
        rows.append(f"{item.relative_to(path).as_posix()}\t{item.stat().st_size}\t{sha256_file(item)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def inspect(name: str) -> dict[str, Any]:
    spec = ARTIFACTS[name]
    target = ROOT / spec["target"]
    missing = []
    mismatched = []
    file_checks = []
    for relative, expectation in spec["required"].items():
        expected_size, expected_sha = expectation
        path = target / relative
        if not path.is_file():
            missing.append(relative)
            file_checks.append({"path": relative, "exists": False, "size": None,
                                "expected_size": expected_size, "sha256": None,
                                "expected_sha256": expected_sha, "status": "MISSING"})
            continue
        size = path.stat().st_size
        actual_sha = sha256_file(path) if expected_sha else None
        size_ok = expected_size is None or size == expected_size
        sha_ok = expected_sha is None or actual_sha == expected_sha
        if not size_ok or not sha_ok:
            mismatched.append(relative)
        file_checks.append({"path": relative, "exists": True, "size": size,
                            "expected_size": expected_size, "sha256": actual_sha,
                            "expected_sha256": expected_sha,
                            "status": "PASS" if size_ok and sha_ok else "MISMATCH"})
    partials = []
    metadata_locks = []
    if target.exists():
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(target))
            if path.suffix == ".lock":
                metadata_locks.append({"path": relative, "bytes": path.stat().st_size})
            elif path.suffix in {".tmp", ".incomplete"} or ".incomplete" in path.name:
                partials.append(relative)
    actual_tree = hash_tree(target) if target.is_dir() and not missing and not partials else None
    expected_tree = spec.get("expected_tree_sha256")
    if expected_tree and actual_tree and actual_tree != expected_tree:
        mismatched.append("<tree_sha256>")
    index_shards = []
    index_name = spec.get("index")
    if index_name and (target / index_name).is_file():
        try:
            index_value = json.loads((target / index_name).read_text(encoding="utf-8"))
            index_shards = sorted(set(index_value.get("weight_map", {}).values()))
            for shard in index_shards:
                if not (target / shard).is_file() and shard not in missing:
                    missing.append(shard)
        except (OSError, json.JSONDecodeError):
            mismatched.append(index_name)
    marker = target / ".tta_artifact_revision.json"
    marker_value = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    revision_recorded = marker_value.get("repo_id") == spec["repo_id"] and marker_value.get("revision") == spec["revision"]
    complete = bool(spec["revision"] and revision_recorded and target.is_dir() and not missing and not mismatched and not partials)
    return {"artifact": name, "repo_id": spec["repo_id"], "revision": spec["revision"],
            "target": str(target), "exists": target.exists(), "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.is_dir() else 0,
            "missing": missing, "mismatched": mismatched, "partial_files": partials,
            "actual_tree_sha256": actual_tree, "expected_tree_sha256": expected_tree,
            "revision_recorded": revision_recorded, "index_file": index_name,
            "weight_shards_from_index": index_shards, "file_checks": file_checks,
            "metadata_lock_files": metadata_locks,
            "status": "PASS" if complete else ("UNKNOWN_REVISION" if spec["revision"] is None else "INCOMPLETE")}


def mark_revision(name: str) -> None:
    spec = ARTIFACTS[name]
    target = ROOT / spec["target"]
    if not target.is_dir() or not spec["revision"]:
        raise RuntimeError(f"Cannot mark absent or unpinned artifact: {name}")
    value = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
    fd, temporary = tempfile.mkstemp(prefix=".tta_revision.", suffix=".tmp", dir=target)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target / ".tta_artifact_revision.json")


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    core = [item for item in results if item["artifact"] in {"tulu", "beaver_reward", "beaver_cost"}]
    payload = {
        "status": "PASS" if all(item["status"] == "PASS" for item in core) else "FAIL",
        "cpu_only": True,
        "public_core_gate": ["tulu", "beaver_reward", "beaver_cost"],
        "safe_rlhf": {
            "required_by_current_source": True,
            "downloaded": False,
            "status": "DEFERRED_UNKNOWN_HISTORICAL_REVISION",
            "evidence": "PARM_TARO/evaluation/scoring.py imports safe_rlhf.models.AutoModelForScore",
        },
        "artifacts": results,
    }
    (path / "public_artifacts.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Public artifact restore", "", f"Core gate: **{payload['status']}**", "",
             "All downloads were CPU-only, revision-pinned, direct-to-local-dir, and checked for incomplete files.", "",
             "| Artifact | Revision | Status | Bytes | Weight shards | Target |", "|---|---|---|---:|---:|---|"]
    for item in core:
        lines.append(f"| {item['repo_id']} | `{item['revision']}` | {item['status']} | {item['bytes']} | {len(item['weight_shards_from_index'])} | `{item['target']}` |")
    lines += ["", "Safe-RLHF is required by the current Stage-10 scoring source, but was not cloned because its historical commit remains unknown.", ""]
    (path / "public_artifacts.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=tuple(ARTIFACTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--require-free-gib", type=float)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--public-core", action="store_true")
    parser.add_argument("--mark-revision", choices=tuple(ARTIFACTS))
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    if args.mark_revision:
        mark_revision(args.mark_revision)
        return 0
    if args.require_free_gib is not None:
        free = shutil.disk_usage(ROOT).free
        result = {"free_bytes": free, "free_gib": free / 1024**3,
                  "required_gib": args.require_free_gib,
                  "status": "PASS" if free >= args.require_free_gib * 1024**3 else "INSUFFICIENT_SPACE"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 3
    names = (["tulu", "beaver_reward", "beaver_cost"] if args.public_core else
             list(ARTIFACTS) if args.all or not args.artifact else [args.artifact])
    results = [inspect(name) for name in names]
    if args.write_report:
        write_report(args.write_report if args.write_report.is_absolute() else ROOT / args.write_report, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(item["status"] == "PASS" for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
