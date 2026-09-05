"""Audit local Stage 7 dependencies without loading large model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.adapters.parm_adapter import (
    activate_vendored_parm_runtime,
    inspect_parm_adapter_config,
    load_original_parm_generation_module,
)
from PARM_TARO.config import ParmTaroConfig
from router_v2.smart_checkpoint import validate_smart_checkpoint_payload
from router_v2.smart_config import SmartRouterConfig
from router_v2.training.audit import hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO" / "configs" / "parm_taro_tulu2.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "PARM_TARO" / "reports" / "stage7_prerequisite_audit.json"
)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return record
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    record.update(
        {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "first_record_keys": sorted(first),
        }
    )
    return record


def audit(config: ParmTaroConfig) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        runtime = activate_vendored_parm_runtime()
        details["vendored_origins"] = runtime.origins
        checks["vendored_dependencies_resolve"] = True
    except Exception as error:
        details["vendored_import_error"] = f"{type(error).__name__}: {error}"
        checks["vendored_dependencies_resolve"] = False
    try:
        original = load_original_parm_generation_module()
        details["original_parm_generation_source"] = str(
            Path(original.__file__).resolve()
        )
        checks["original_parm_generation_module_resolves"] = True
    except Exception as error:
        details["original_parm_import_error"] = (
            f"{type(error).__name__}: {error}"
        )
        checks["original_parm_generation_module_resolves"] = False

    base_path = _resolve(config.base_model_path)
    tokenizer_path = _resolve(config.tokenizer_path)
    adapter_path = _resolve(config.parm_adapter_path)
    router_path = _resolve(config.router_checkpoint_path)
    checks["base_model_present"] = (base_path / "config.json").is_file()
    checks["tokenizer_present"] = tokenizer_path.exists()
    details["paths"] = {
        "base_model": str(base_path),
        "tokenizer": str(tokenizer_path),
        "parm_adapter": str(adapter_path),
        "router_checkpoint": str(router_path),
    }
    try:
        adapter_config = inspect_parm_adapter_config(
            adapter_path,
            expected_preference_dim=config.preference_dim,
        )
        checks["preference_aware_parm_adapter_ready"] = True
        details["parm_adapter"] = {
            "peft_type": adapter_config["peft_type"],
            "obj_num": adapter_config["obj_num"],
            "base_model_name_or_path": adapter_config["base_model_name_or_path"],
        }
    except Exception as error:
        checks["preference_aware_parm_adapter_ready"] = False
        details["parm_adapter_error"] = f"{type(error).__name__}: {error}"

    try:
        payload = torch.load(router_path, map_location="cpu", weights_only=False)
        validate_smart_checkpoint_payload(payload)
        router_config = SmartRouterConfig.from_dict(payload["config"])
        if not router_config.use_preference:
            raise ValueError("Router checkpoint has use_preference=false")
        if router_config.preference_dim != config.preference_dim:
            raise ValueError("Router checkpoint preference_dim mismatch")
        if (
            payload["metadata"].get("tokenizer_semantic_sha256")
            != config.router_tokenizer_semantic_sha256
        ):
            raise ValueError("Router checkpoint tokenizer hash mismatch")
        checks["parm_compatible_router_ready"] = True
        details["router"] = {
            "variant": router_config.variant,
            "vocab_size": router_config.vocab_size,
            "top_k": router_config.top_k,
            "preference_dim": router_config.preference_dim,
        }
    except Exception as error:
        checks["parm_compatible_router_ready"] = False
        details["router_error"] = f"{type(error).__name__}: {error}"

    dataset_path = (
        PROJECT_ROOT
        / "dataset"
        / "GenARM"
        / "PKU-SafeRLHF-10K"
        / "round0"
        / "train.jsonl.xz"
    )
    details["dataset"] = _dataset_record(dataset_path)
    checks["local_pku_source_present"] = bool(details["dataset"]["exists"])
    details["parm_tree"] = hash_tree(PROJECT_ROOT / "PARM")
    checks["parm_tree_matches_baseline"] = (
        details["parm_tree"]["tree_sha256"]
        == "dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a"
    )
    return {
        "schema_version": 1,
        "stage": 7,
        "method_label": "PARM_TARO",
        "checks": checks,
        "details": details,
        "ready": all(checks.values()),
        "status": "READY" if all(checks.values()) else "PREREQUISITES_REQUIRED",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(ParmTaroConfig.load_json(args.config))
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
