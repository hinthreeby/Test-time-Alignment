#!/usr/bin/env python3
"""Resume-safe entry point for the token-level headroom experiment.

The ``all`` phase deliberately starts each model-bearing phase in a fresh
process.  Consequently Tulu/PBLoRA, Beaver reward, and Beaver cost can never
coexist in one Python process or CUDA allocator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _insert_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "Method/PARM_TARO").exists():
            sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Cannot locate Test-time-Alignment project root")


ROOT = _insert_project_root()

from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import (  # noqa: E402
    bootstrap_import_paths,
    subprocess_environment,
)

SOURCE_PATHS = bootstrap_import_paths(ROOT)

from PARM_TARO.adaptive_parm.token_headroom.config import (  # noqa: E402
    ADAPTER,
    BASE,
    COST,
    OUT,
    PHASE09,
    REWARD,
    SAFE_RLHF,
    SOURCE_MANIFEST,
    WEIGHTS,
)
from PARM_TARO.adaptive_parm.token_headroom.io import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    read_jsonl,
)


ALPHAS = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))


def build_manifest(num_prompts: int) -> list[dict[str, Any]]:
    source = read_jsonl(SOURCE_MANIFEST)
    if not source:
        raise RuntimeError(f"Frozen Feasibility60 manifest missing/empty: {SOURCE_MANIFEST}")
    prompt_ids = list(dict.fromkeys(str(row["sample_id"]) for row in source))[:num_prompts]
    if len(prompt_ids) != num_prompts:
        raise RuntimeError(f"Requested {num_prompts} prompts, source has {len(prompt_ids)}")
    selected_set = set(prompt_ids)
    selected = [dict(row) for row in source if str(row["sample_id"]) in selected_set]
    order = {sample_id: index for index, sample_id in enumerate(prompt_ids)}
    alpha_order = {alpha: index for index, alpha in enumerate(ALPHAS)}
    selected.sort(key=lambda row: (order[str(row["sample_id"])], alpha_order[tuple(float(x) for x in row["requested_alpha"])]))

    counts = Counter(str(row["sample_id"]) for row in selected)
    alpha_counts = Counter(tuple(float(x) for x in row["requested_alpha"]) for row in selected)
    if len(selected) != num_prompts * len(ALPHAS):
        raise RuntimeError(f"Expected {num_prompts * len(ALPHAS)} cases, got {len(selected)}")
    if set(counts.values()) != {len(ALPHAS)} or set(alpha_counts) != set(ALPHAS):
        raise RuntimeError(f"Manifest is not a full prompt x alpha grid: {counts=} {alpha_counts=}")
    if any(row.get("alpha_order") != ["helpfulness", "harmlessness"] for row in selected):
        raise RuntimeError("Manifest alpha order is not [helpfulness, harmlessness]")

    target = OUT / "manifest.jsonl"
    if target.exists():
        previous = read_jsonl(target)
        if previous != selected:
            raise RuntimeError(f"Frozen token-headroom manifest differs from requested selection: {target}")
    else:
        atomic_jsonl(target, selected)
    return selected


def _artifact_check(path: Path, *, directory: bool = True) -> dict[str, Any]:
    exists = path.is_dir() if directory else path.is_file()
    return {"path": str(path), "exists": exists, "readable": exists and os.access(path, os.R_OK)}


def dry_run(num_prompts: int) -> dict[str, Any]:
    manifest = build_manifest(num_prompts)
    try:
        import router_v2
        from router_v2.device import resolve_device

        router_import = {
            "ready": True,
            "module_file": str(Path(router_v2.__file__).resolve()),
            "resolve_device_module": resolve_device.__module__,
            "source_paths": list(SOURCE_PATHS),
        }
    except Exception as error:
        router_import = {"ready": False, "error": repr(error), "source_paths": list(SOURCE_PATHS)}
    checks = {
        "base_model": _artifact_check(BASE),
        "pblora_adapter": _artifact_check(ADAPTER),
        "adapter_weights": _artifact_check(ADAPTER / "adapter_model.safetensors", directory=False),
        "reward_model": _artifact_check(REWARD),
        "cost_model": _artifact_check(COST),
        "safe_rlhf_source": _artifact_check(SAFE_RLHF),
        "phase09_normalization": _artifact_check(PHASE09, directory=False),
        "source_manifest": _artifact_check(SOURCE_MANIFEST, directory=False),
    }
    try:
        from PARM_TARO.recovery.low_vram_suite.common import resolve_safe_rlhf_source

        safe_resolution = resolve_safe_rlhf_source()
    except Exception as error:  # preflight must report rather than hide import errors
        safe_resolution = {"ready": False, "error": repr(error)}
    expected_router_parent = (ROOT / "Method/Router_Apdative").resolve()
    router_location_ok = router_import.get("ready", False) and expected_router_parent in Path(router_import["module_file"]).parents
    router_import["canonical_location"] = bool(router_location_ok)
    ready = all(item["exists"] and item["readable"] for item in checks.values()) and bool(safe_resolution.get("ready")) and router_location_ok
    result = {
        "status": "PASS" if ready else "FAIL",
        "loads_large_models": False,
        "num_prompts": num_prompts,
        "num_cases": len(manifest),
        "max_states": len(manifest) * 5,
        "max_action_rollouts": len(manifest) * 5 * len(WEIGHTS),
        "weight_grid": list(WEIGHTS),
        "checks": checks,
        "router_v2_import": router_import,
        "safe_rlhf_resolution": safe_resolution,
    }
    atomic_json(OUT / "preflight.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not ready:
        raise RuntimeError("Token-headroom preflight failed; inspect preflight.json")
    return result


def _run_child(phase: str, args: argparse.Namespace) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--num-prompts", str(args.num_prompts)]
    if args.resume:
        command.append("--resume")
    command.extend(("--min-free-mib", str(args.min_free_mib)))
    print("Launching isolated phase:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=subprocess_environment(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "generate", "reward", "cost", "analyze"), default="all")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=int(os.environ.get("MIN_FREE_MIB", "6000")))
    args = parser.parse_args()
    if args.num_prompts != 10:
        raise SystemExit("This locked diagnostic requires exactly --num-prompts 10")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.num_prompts)
    if args.dry_run:
        dry_run(args.num_prompts)
        return 0
    if args.phase == "all":
        dry_run(args.num_prompts)
        for phase in ("generate", "reward", "cost", "analyze"):
            _run_child(phase, args)
        return 0
    if args.phase == "generate":
        from PARM_TARO.adaptive_parm.token_headroom.generation import run_generation

        run_generation(manifest, resume=args.resume, min_free_mib=args.min_free_mib)
    elif args.phase in ("reward", "cost"):
        from PARM_TARO.adaptive_parm.token_headroom.scoring import run_scoring

        run_scoring(args.phase, resume=args.resume, min_free_mib=args.min_free_mib)
    else:
        from PARM_TARO.adaptive_parm.token_headroom.analysis import run_analysis

        result = run_analysis()
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
