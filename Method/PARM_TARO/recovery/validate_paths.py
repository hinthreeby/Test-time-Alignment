"""CPU-only, resumable path audit for the relocated PARM-TARO workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".sh", ".py", ".md"}
PATH_KEYS = (
    "path", "root", "dir", "checkpoint", "manifest", "tokenizer", "model",
    "adapter", "cache", "dataset", "source", "output", "normalization", "protocol",
)
ABSOLUTE_RE = re.compile(r'''["'](/home/[^"'\n]+)["']''')
RELATIVE_RE = re.compile(
    r'''["']((?:models|results|dataset|checkpoints|training|PARM_TARO|router_v2|PARM|router|Method)/[^"'\n]*)["']'''
)
OLD_ROOT = Path("/home/jupyter-iec2024se10/Reward Decoding")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_root(start: Path) -> Path:
    configured = os.environ.get("TTA_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / ".git").exists():
            return root
        raise RuntimeError(f"TTA_PROJECT_ROOT is not a project root: {root}")
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("project root containing .git was not found")


ROOT = find_root(Path(__file__).resolve().parent)
OUT = ROOT / "results/parm_taro/recovery/path_audit"
STATE = OUT / "state.json"
CACHE = OUT / "references.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_files() -> list[Path]:
    roots = [
        ROOT / "Method/PARM_TARO", ROOT / "Method/Router_Apdative/router_v2",
        ROOT / "Method/PARM", ROOT / "Method/Router_Apdative/router",
        ROOT / "Method/RAD", ROOT / "configs", ROOT / "scripts", ROOT / "document",
    ]
    seen: set[Path] = set()
    found: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(ROOT)
            if "__pycache__" in relative.parts or "path_audit" in relative.parts:
                continue
            real = path.resolve()
            if real not in seen:
                seen.add(real)
                found.append(path)
    return sorted(found, key=lambda value: value.as_posix())


def fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.relative_to(ROOT)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def json_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            joined = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str) and any(token in key.lower() for token in PATH_KEYS):
                yield joined, child
            yield from json_paths(child, joined)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from json_paths(child, f"{prefix}[{index}]")


def scan(paths: Iterable[Path]) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source in paths:
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        source_name = source.relative_to(ROOT).as_posix()
        for pattern in (ABSOLUTE_RE, RELATIVE_RE):
            for match in pattern.finditer(text):
                refs[(match.group(1), "text-reference")].add(source_name)
        if source.suffix.lower() == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            for key, raw in json_paths(value):
                if "/" in raw or raw.startswith("."):
                    refs[(raw, f"config:{key}")].add(source_name)
    return [
        {"path": path, "reference_type": kind, "sources": sorted(sources)}
        for (path, kind), sources in sorted(refs.items())
    ]


def moved_target(path: Path) -> Path | None:
    try:
        relative = path.relative_to(OLD_ROOT)
    except ValueError:
        return None
    candidate = ROOT / relative
    return candidate if candidate.exists() else None


def inspect_path(
    raw: str,
    path_type: str,
    source: str,
    *,
    required: bool,
    expected_kind: str = "any",
) -> dict[str, Any]:
    expanded = os.path.expandvars(raw)
    dynamic = "$" in expanded or "<" in expanded or "REQUIRED_AFTER" in expanded
    path = Path(expanded)
    resolved_input = path if path.is_absolute() else ROOT / path
    relocated = moved_target(resolved_input) if resolved_input.is_absolute() else None
    actual = relocated or resolved_input
    physical = actual.resolve(strict=False)
    exists = actual.exists()
    kind_ok = exists and (
        expected_kind == "any" or (expected_kind == "file" and actual.is_file())
        or (expected_kind == "dir" and actual.is_dir())
    )
    readable = kind_ok and os.access(actual, os.R_OK)
    writable = kind_ok and os.access(actual, os.W_OK)
    if dynamic:
        status = "CONFIG_DRIVEN"
    elif relocated is not None:
        status = "EXISTS_BUT_MOVED"
    elif not kind_ok:
        status = "BROKEN" if required else ("HARD_CODED" if path.is_absolute() else "CONFIG_DRIVEN")
    elif path.is_absolute():
        status = "HARD_CODED"
    elif resolved_input.is_symlink() or any(parent.is_symlink() for parent in (resolved_input, *resolved_input.parents) if parent != parent.parent):
        status = "EXISTS_BUT_MOVED"
    elif path_type.startswith("config"):
        status = "CONFIG_DRIVEN"
    else:
        status = "RELATIVE_SAFE"
    return {
        "path": raw,
        "type": path_type,
        "exists": bool(kind_ok),
        "readable": bool(readable),
        "writable": bool(writable),
        "source_of_path": source,
        "status": status,
        "required": required,
        "resolved_path": str(physical),
    }


def canonical_entries() -> list[dict[str, Any]]:
    specs = [
        ("PARM_TARO", "source-dir", True, "dir"),
        ("router_v2", "source-dir", True, "dir"),
        ("router", "protected-source-dir", True, "dir"),
        ("PARM", "protected-source-dir", True, "dir"),
        ("Method/RAD", "protected-source-dir", True, "dir"),
        ("results/parm_taro", "results-root", True, "dir"),
        ("results/router_v2", "results-root", True, "dir"),
        ("PARM_TARO/recovery", "recovery-source", True, "dir"),
        ("results/parm_taro/recovery/reconstruction", "reconstruction-results", True, "dir"),
        ("router_v2/cache", "cache-source", True, "dir"),
        ("dataset/parm_taro/manifest.json", "dataset-manifest", True, "file"),
        ("dataset/parm_taro/train.json", "dataset-train", True, "file"),
        ("dataset/parm_taro/validation.json", "dataset-validation", True, "file"),
        ("dataset/parm_taro/test_prompt_only.json", "dataset-test", True, "file"),
        ("dataset/router_v2_cache", "feature-cache", True, "dir"),
        ("dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz", "safe-rlhf-training-data", True, "file"),
        ("results/parm_taro/evaluation/protocol/protocol_lock.json", "protocol-lock", True, "file"),
        ("models/tulu-2-7b", "base-model-tokenizer", True, "dir"),
        ("models/beaver-7b-v1.0-reward", "reward-evaluator", True, "dir"),
        ("models/beaver-7b-v1.0-cost", "cost-evaluator", True, "dir"),
        ("models/safe-rlhf-source", "evaluator-source", True, "dir"),
        ("results/parm_taro/checkpoints/parm_pku_pblora", "pblora-checkpoint", True, "dir"),
        ("results/parm_taro/training/taro/best.pt", "stage9-checkpoint", True, "file"),
        ("results/parm_taro/training/v2_no_alpha/best.pt", "stage9-checkpoint", True, "file"),
        ("results/parm_taro/training/v2_alpha_preference/final.pt", "stage9-checkpoint", True, "file"),
        ("results/parm_taro/evaluation/full/stage10_report.json", "stage10-artifact", True, "file"),
        ("models/gpt2-medium", "router-v2-base-model", True, "dir"),
        ("models/gpt2-large", "router-v1-base-model", True, "dir"),
        ("models/gpt2-small", "router-v1-reward-base", True, "dir"),
        ("models/sentiment-roberta-large-english/pytorch_model.bin", "sentiment-evaluator", True, "file"),
        ("models/rad_rm_sentiment/pytorch_model.bin", "router-v1-reward-checkpoint", True, "file"),
        ("results/router_v2/sentiment_guide/final_adapter/adapter_model.safetensors", "router-v2-guide-checkpoint", True, "file"),
        ("results/router_v2/training/taro/best.pt", "router-v2-checkpoint", True, "file"),
        ("results/router_v2/training/state/best.pt", "router-v2-checkpoint", True, "file"),
        ("results/router_v2/training/history/best.pt", "router-v2-checkpoint", True, "file"),
        ("results/router_v2/training/alpha_parm/best.pt", "parm-adaptive-router-checkpoint", True, "file"),
        ("PARM_TARO/recovery/scripts/train_pblora_repro.sh", "training-wrapper", True, "file"),
        ("PARM_TARO/recovery/scripts/train_taro_repro.sh", "training-wrapper", True, "file"),
        ("PARM_TARO/recovery/scripts/train_v2_no_alpha_repro.sh", "training-wrapper", True, "file"),
        ("PARM_TARO/recovery/scripts/train_v2_full_alpha_old_repro.sh", "training-wrapper", True, "file"),
        ("PARM_TARO/recovery/scripts/train_v2_recovered.sh", "training-wrapper", True, "file"),
        ("PARM_TARO/recovery/scripts/evaluate_recovery_stage.sh", "evaluation-wrapper", True, "file"),
        # These remain deliberately gated until reproduction/diagnosis reaches
        # the corresponding stage. Their absence is not a relocation failure.
        ("PARM_TARO/recovery/configs/train_v2_full_alpha_old_repro.json", "future-wrapper-config", False, "file"),
        ("PARM_TARO/recovery/configs/train_v2_recovered.json", "future-wrapper-config", False, "file"),
        ("PARM_TARO/recovery/configs/evaluate_val200.json", "future-evaluation-config", False, "file"),
        ("PARM_TARO/recovery/configs/evaluate_val500.json", "future-evaluation-config", False, "file"),
        ("PARM_TARO/recovery/configs/evaluate_test1500.json", "future-evaluation-config", False, "file"),
    ]
    entries = [inspect_path(path, kind, "canonical audit inventory", required=required, expected_kind=expected)
               for path, kind, required, expected in specs]
    for manifest in sorted((ROOT / "dataset/router_v2_cache").glob("**/manifest.json")):
        try:
            shards = json.loads(manifest.read_text(encoding="utf-8")).get("shards", [])
        except (OSError, json.JSONDecodeError):
            continue
        for shard in shards:
            relative = (manifest.parent / str(shard["path"])).relative_to(ROOT).as_posix()
            entries.append(inspect_path(
                relative,
                "feature-cache-shard",
                manifest.relative_to(ROOT).as_posix(),
                required=True,
                expected_kind="file",
            ))
    return entries


def reference_entries(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for ref in references:
        historical = all(
            source.startswith(("Method/PARM_TARO/results/", "Method/PARM_TARO/reports/", "Method/Router_Apdative/router_v2/reports/"))
            for source in ref["sources"]
        )
        source = ", ".join(ref["sources"][:4])
        if len(ref["sources"]) > 4:
            source += f" (+{len(ref['sources']) - 4} more)"
        entry = inspect_path(ref["path"], ref["reference_type"], source, required=False)
        entry["historical_record"] = historical
        entries.append(entry)
    return entries


def markdown(entries: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# PARM-TARO path audit", "",
        f"Generated: `{summary['generated_at']}`", "",
        f"Final gate: **{summary['gate']}**. Required broken paths: **{summary['required_broken']}**.", "",
        "Host GPU is available (RTX 5090); agent GPU access is unavailable by design. No GPU check was used by this audit.", "",
        "Compatibility links restore the canonical project-root imports and output locations while their physical data remains under `Method/`.", "",
        "| PATH | TYPE | EXISTS | READABLE | WRITABLE | SOURCE_OF_PATH | STATUS | REQUIRED | RESOLVED_PATH |",
        "|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for item in entries:
        values = [
            item["path"], item["type"], str(item["exists"]), str(item["readable"]),
            str(item["writable"]), item["source_of_path"], item["status"],
            str(item["required"]), item["resolved_path"],
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    lines.extend(["", "Historical reports/protocol locks remain immutable. Old `/Reward Decoding/...` values found only in retained evidence are recorded but are not rewritten.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = source_files()
    current_fingerprint = fingerprint(paths)
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    if args.resume and state.get("fingerprint") == current_fingerprint and CACHE.is_file():
        references = json.loads(CACHE.read_text())
        scan_status = "resumed"
    else:
        references = scan(paths)
        atomic_json(CACHE, references)
        scan_status = "scanned"
    entries = canonical_entries() + reference_entries(references)
    required_broken = [item for item in entries if item["required"] and item["status"] == "BROKEN"]
    counts: dict[str, int] = defaultdict(int)
    for item in entries:
        counts[item["status"]] += 1
    summary = {
        "project_root": str(ROOT),
        "generated_at": utc_now(),
        "gate": "PATH AUDIT PASS" if not required_broken else "PATH AUDIT FAIL",
        "required_broken": len(required_broken),
        "required_broken_paths": [item["path"] for item in required_broken],
        "status_counts": dict(sorted(counts.items())),
        "files_scanned": len(paths),
        "references_scanned": len(references),
        "scan_status": scan_status,
        "host_gpu": "AVAILABLE",
        "agent_gpu_access": "UNAVAILABLE_BY_DESIGN",
        "compatibility_links": {
            str(path): os.readlink(path)
            for path in (
                ROOT / "PARM", ROOT / "PARM_TARO", ROOT / "router", ROOT / "router_v2",
                ROOT / "results/parm_taro", ROOT / "results/router_v2", OLD_ROOT,
            )
            if path.is_symlink()
        },
        "entries": entries,
    }
    atomic_json(OUT / "path_audit.json", summary)
    report = OUT / "path_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(entries, summary), encoding="utf-8")
    atomic_json(STATE, {"fingerprint": current_fingerprint, "status": "done", "updated_at": utc_now()})
    print("PATH\tTYPE\tEXISTS\tREADABLE\tWRITABLE\tSOURCE_OF_PATH\tSTATUS")
    for item in entries:
        print("\t".join(str(item[key]) for key in ("path", "type", "exists", "readable", "writable", "source_of_path", "status")))
    print(summary["gate"])
    return 0 if not required_broken else 2


if __name__ == "__main__":
    raise SystemExit(main())
