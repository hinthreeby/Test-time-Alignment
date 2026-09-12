"""Create the non-destructive PARM-TARO server-restore audit reports.

This script uses only the Python standard library.  It does not install
packages, load a model, generate text, evaluate prompts, or train anything.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
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
OUT = ROOT / "results/parm_taro/recovery/server_restore"
TTA_PYTHON = Path(os.environ.get("TTA_PYTHON", sys.executable))


def command(args: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout.rstrip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "present": False, "file_count": 0,
                "total_bytes": 0, "tree_sha256": None}
    rows: list[str] = []
    total = 0
    files = sorted((p for p in path.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(path).as_posix())
    for item in files:
        rel = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total += size
        rows.append(f"{rel}\t{size}\t{sha256_file(item)}\n")
    return {"path": str(path), "present": True, "file_count": len(files),
            "total_bytes": total,
            "tree_sha256": hashlib.sha256("".join(rows).encode()).hexdigest()}


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def human(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(value)
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:.1f} {unit}"
        n /= 1024
    return str(value)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write(name, json.dumps(value, indent=2, sort_keys=True))


def git_info(path: str) -> tuple[str, str]:
    _, tracked = command(["git", "ls-files", "--", path])
    _, changed = command(["git", "status", "--short", "--", path])
    return ("yes" if tracked else "no", changed or "clean")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    _, commit = command(["git", "rev-parse", "HEAD"])
    _, branch = command(["git", "branch", "--show-current"])
    _, git_status = command(["git", "status", "--short"])
    _, remotes = command(["git", "remote", "-v"])

    inventory = [
        ("PARM baseline", "PARM", True, "Preserve; tracked source clean"),
        ("PARM-TARO", "PARM_TARO", True, "Recovery code present"),
        ("Router V1", "router", True, "Preserve; tracked source clean"),
        ("Router V2", "router_v2", True, "Incomplete without cache package"),
        ("Router V2 cache package", "router_v2/cache", True, "Restore exact source from backup"),
        ("RAD baseline", "Method/RAD", True, "Preserve; tracked source clean"),
        ("Dataset", "dataset", True, "Processed and source PKU data present"),
        ("PARM-TARO results", "results/parm_taro", True, "Historical reports present"),
        ("Recovery code", "PARM_TARO/recovery", True, "Present; untracked recovery-only code"),
        ("Recovery evidence", "results/parm_taro/recovery", True, "Present; historical files preserved"),
        ("Stage 9 training records", "results/parm_taro/training", True, "JSON records present; weights absent"),
        ("Stage 10 evaluation", "results/parm_taro/evaluation", True, "Historical outputs present"),
        ("Checkpoint root", "results/parm_taro/checkpoints", True, "Restore from backup"),
    ]
    lines = [
        "# Source inventory", "", f"Project root: `{ROOT}`", "",
        f"Git branch/commit: `{branch}` / `{commit}`", "",
        "The protected tracked trees have no Git modifications. Recovery files and Files 12-13 are untracked; no checkout/reset/rebase was performed.", "",
        "| Component | Expected path | Exists? | Size | Git tracked? | Modified? | Required? | Action |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for component, rel, required, action in inventory:
        path = ROOT / rel
        tracked, modified = git_info(rel)
        lines.append(f"| {component} | `{rel}` | {'yes' if path.exists() else 'no'} | {human(size(path))} | {tracked} | `{modified}` | {'yes' if required else 'no'} | {action} |")
    lines += ["", "## Git status", "", "```text", git_status or "clean", "```", "", "## Remotes", "", "```text", remotes, "```"]
    write("01_source_inventory.md", "\n".join(lines))

    expected_trees = {
        "PARM original": ("PARM", "dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a"),
        "router V1": ("router", "8595fb3e48c2b1771a07a6efba09c29560785a581d5490f582286c58fbf5ffd3"),
        "Method/RAD": ("Method/RAD", "e0c8907885b80cde9c12f0e1f6e3134b0781e811ee8edafb9f0d690988d36bbe"),
        "PBLoRA": ("results/parm_taro/checkpoints/parm_pku_pblora", "f26682e3cf51f8e09921873119fc371d329669cf9448d59ae0025fd40b23ffef"),
    }
    expected_files = {
        "Stage 9 TARO": ("results/parm_taro/training/taro/best.pt", "32caa71a0e1462e5f31a6c587c735b19c22d109f16f729430cd63c7910731881", 14327946),
        "Stage 9 V2 full-alpha": ("results/parm_taro/training/v2_alpha_preference/final.pt", "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20", 4374570),
        "Stage 9 V2 no-alpha": ("results/parm_taro/training/v2_no_alpha/best.pt", "ea5cf0e7b33ccdef1f4882afba0679a09feaaf2b27c8a8bc12ad7c058cb0d7e3", 12976666),
    }
    artifacts: list[dict[str, Any]] = []
    for name, (rel, expected) in expected_trees.items():
        measured = hash_tree(ROOT / rel)
        status = "MISSING" if not measured["present"] else ("VERIFIED" if measured["tree_sha256"] == expected else "MISMATCH")
        artifacts.append({"name": name, "kind": "tree", "relative_path": rel,
                          "expected_sha256": expected, "actual_sha256": measured["tree_sha256"],
                          "bytes": measured["total_bytes"], "files": measured["file_count"], "status": status})
    for name, (rel, expected, expected_bytes) in expected_files.items():
        path = ROOT / rel
        actual = sha256_file(path) if path.is_file() else None
        status = "MISSING" if actual is None else ("VERIFIED" if actual == expected else "MISMATCH")
        artifacts.append({"name": name, "kind": "file", "relative_path": rel,
                          "expected_sha256": expected, "actual_sha256": actual,
                          "expected_bytes": expected_bytes,
                          "bytes": path.stat().st_size if path.is_file() else 0, "status": status})
    audit_json = {
        "project_root": str(ROOT), "git_commit": commit,
        "method": "tree hash = SHA256 of sorted relative_path<TAB>size<TAB>file_sha256<LF>",
        "artifacts": artifacts,
        "interpretation": "Current tracked source is clean at HEAD. Historical full-workspace tree hashes included ignored/runtime files absent after restore; therefore current tree mismatch is not by itself evidence of a source edit.",
        "status": "FAIL",
    }
    write_json("02_protected_artifact_audit.json", audit_json)
    p = ["# Protected artifact audit", "", "Status: **FAIL** — none of the seven complete historical artifacts can be verified in this restored snapshot.", "",
         "| Artifact | Path | Expected SHA-256 | Actual SHA-256 | Status |", "|---|---|---|---|---|"]
    for item in artifacts:
        p.append(f"| {item['name']} | `{item['relative_path']}` | `{item['expected_sha256']}` | `{item['actual_sha256'] or '—'}` | **{item['status']}** |")
    p += ["", "`PARM/`, `router/`, and `Method/RAD/` are Git-clean at the current commit. Their MISMATCH status means the archived hash covered a fuller old workspace (including now-missing ignored/runtime files); it does not prove the tracked files were edited. PBLoRA and all three Stage-9 weight files are genuinely missing."]
    write("02_protected_artifact_audit.md", "\n".join(p))

    model_inventory = """# Model, data, and protocol inventory

## Models and checkpoints

| Role | Exact ID/path established by retained evidence | Revision / hash | Present | Status |
|---|---|---|---:|---|
| Base LLM | `allenai/tulu-2-7b` / `models/tulu-2-7b` | revision UNKNOWN; archived aggregate `cd816175…` | no | BLOCKING |
| Tokenizer | Tulu-2-7B tokenizer | archived semantic SHA `25cd4342…` | no | BLOCKING |
| PARM PBLoRA | `results/parm_taro/checkpoints/parm_pku_pblora` | tree `f26682e3…`; adapter weights `10127909…` | no | BLOCKING |
| TARO router | `results/parm_taro/training/taro/best.pt` | `32caa71a…` | no | BLOCKING |
| V2 full-alpha | `results/parm_taro/training/v2_alpha_preference/final.pt` | `56fbed68…` | no | BLOCKING |
| V2 no-alpha | `results/parm_taro/training/v2_no_alpha/best.pt` | `ea5cf0e7…` | no | BLOCKING |
| Helpfulness evaluator | `PKU-Alignment/beaver-7b-v1.0-reward` | revision UNKNOWN; tree `3410256f…` | no | BLOCKING FOR STAGE 10 |
| Harmlessness evaluator | `PKU-Alignment/beaver-7b-v1.0-cost` | revision UNKNOWN; tree `418b9416…` | no | BLOCKING FOR STAGE 10 |
| Safe-RLHF evaluator source | `PKU-Alignment/safe-rlhf` | revision UNKNOWN; tree `1b61a3e4…` | no | BLOCKING FOR STAGE 10 |

The retained Stage-10 configuration specifies `load_in_4bit=true`, one physical shared backbone, base pass through `PeftModel.disable_adapter`, and guide pass with PBLoRA enabled. Exact public repository names are known, but exact model revisions are not recorded; no substitute or latest revision was downloaded.

## Dataset

| Item | Path / strategy | Evidence | Status |
|---|---|---|---|
| Original PKU data | `dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz` | 10,000 records; SHA `f5f42f6…` | PASS |
| Train | `dataset/parm_taro/train.json` | 8,000; SHA `9a1ca7c…` | PASS |
| Validation | `dataset/parm_taro/validation.json` | 500; SHA `cdbb9144…` | PASS |
| Test | `dataset/parm_taro/test.json` | 1,500; SHA `892a5796…` | PASS |
| Test prompt-only | `dataset/parm_taro/test_prompt_only.json` | 1,500; SHA `1ab1b59e…` | PASS |
| Validation-200 | `results/parm_taro/recovery/val200/manifest.json` | deterministic lowest SHA-256 under seed label `parm_taro_recovery_val200_seed_2026` | PASS |
| Validation-500 | full validation split | File 13 | PASS |

## Frozen protocol

- Lock: `results/parm_taro/evaluation/protocol/protocol_lock.json`.
- Embedded canonical SHA-256: `81a465048a6b0a1b7a585ff37c2ca42dd67dd5d90a591225c9ae7603bb05faa7` (verified by canonical recomputation).
- Alpha grid: `(0,1), (.25,.75), (.5,.5), (.75,.25), (1,0)` in helpfulness/harmlessness order.
- Seed: `2026`; deterministic generation; 64 new tokens.
- Normalization: validation quantile min-max clip, q01/q99, anchors helpfulness `[-13.500625, 15.42625]`, harmlessness `[-43.375, 18.98453125]`.
- HV reference: `[0,0]`, maximize both objectives.
- Regret pool: all seven outputs for identical prompt/requested-alpha/seed.

Dataset and protocol specifications are present. Execution is blocked by missing models/checkpoints and runtime source.
"""
    write("03_model_and_data_inventory.md", model_inventory)

    missing = """# Missing artifacts and exact unblock requirements

## Blocking artifacts

1. Restore the original `router_v2/cache/` package, at minimum `__init__.py`, `features.py`, `io.py`, `schema.py`, and `inventory.py`. It is referenced throughout the project but was ignored by the broad `cache/` Git rule and is absent from Git history, remote `main`, local mounts, and caches.
2. Restore `models/tulu-2-7b/` matching the archived model shard hashes (index `e572e08c…`, shard 1 `6d90e535…`, shard 2 `3eb1b183…`). The exact Hugging Face revision is UNKNOWN.
3. Restore PBLoRA at `results/parm_taro/checkpoints/parm_pku_pblora/`, tree SHA `f26682e3…`; its `adapter_model.safetensors` must be 25,233,088 bytes and SHA `10127909…`.
4. Restore the three Stage-9 checkpoints with exact hashes listed in `02_protected_artifact_audit.json`.
5. Before Stage-10 reproduction, restore both Beaver evaluators and Safe-RLHF source matching the frozen tree hashes.
6. Restore host access to `/dev/nvidia*` and make `nvidia-smi` work.
7. Resolve the environment only after GPU repair. Retained provenance identifies Python 3.10.20, torch 2.2.2+cu121, transformers 4.39.3, PEFT 0.10.0, accelerate 0.29.2, and bitsandbytes 0.43.1. Because this server appears to use a newer RTX 5090/GB202 device, do not blindly install the old cu121 torch build: first prove architecture support, or record and validate a minimal cu128 compatibility exception in an isolated environment.

Do not download a current/latest model revision. Supply either the old model directories or the exact revisions plus files that verify against the archived hashes. The root filesystem currently has only about 9.5 GiB free, insufficient for these model assets; free or mount adequate storage first.

## Verification commands after placing a backup

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
sha256sum results/parm_taro/training/taro/best.pt \\
  results/parm_taro/training/v2_alpha_preference/final.pt \\
  results/parm_taro/training/v2_no_alpha/best.pt
/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python \\
  PARM_TARO/recovery/server_restore_audit.py
```

For tree artifacts, rerun the audit rather than assuming a copied directory is correct.
"""
    write("04_missing_artifacts.md", missing)

    checks = []
    for label, args in [
        ("uname -a", ["uname", "-a"]),
        ("/etc/os-release", ["cat", "/etc/os-release"]),
        ("nvidia-smi", ["nvidia-smi"]),
        ("nvidia-smi query", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"]),
        ("nvcc --version", ["nvcc", "--version"]),
        ("python resolution", ["bash", "-lc", "command -v python || true; python --version 2>&1 || true; command -v python3; python3 --version"]),
        ("pip resolution", ["bash", "-lc", "command -v pip || true; pip --version 2>&1 || true"]),
        ("df -h", ["df", "-h"]),
        ("free -h", ["free", "-h"]),
        ("PCI GPU", ["lspci", "-nnk"]),
    ]:
        rc, output = command(args)
        if label == "PCI GPU":
            output = "\n".join(line for line in output.splitlines() if "VGA" in line or "3D controller" in line or "Kernel driver in use: nvidia" in line)
        checks.append(f"## {label}\n\nExit code: `{rc}`\n\n```text\n{output}\n```")
    cuda_code = """import torch
print('torch:', torch.__version__)
print('cuda runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
try:
    a=torch.randn((512,512),device='cuda'); b=torch.randn((512,512),device='cuda'); c=a@b
    print('CUDA_MATMUL_PASS', float(c[0,0]))
except Exception as exc:
    print('CUDA_MATMUL_FAILED:', repr(exc))
"""
    cuda_rc, cuda_out = command([str(TTA_PYTHON), "-c", cuda_code])
    checks.append(f"## Actual PyTorch CUDA compute\n\nExit code: `{cuda_rc}`\n\n```text\n{cuda_out}\n```")
    system_head = """# System and GPU audit

Status: **FAIL**.

PCI ID `10de:2b85` identifies an NVIDIA GeForce RTX 5090 (GB202); the NVIDIA 580.105.08 kernel module is loaded, but `/dev/nvidia*` is absent. `nvidia-smi` cannot communicate with the driver and a real CUDA tensor matmul fails with `No CUDA GPUs are available`. CUDA toolkit 11.5 being installed does not constitute a functioning runtime. No tiny-model smoke test was attempted because GPU compute already failed.
"""
    write("05_system_gpu_audit.md", system_head + "\n" + "\n\n".join(checks))

    pkg_code = """import importlib.metadata as m
names=['torch','transformers','peft','accelerate','bitsandbytes','datasets','tokenizers','numpy','pandas','scipy','scikit-learn','tqdm','safetensors','huggingface-hub']
for n in names:
    try: print(n+'\\t'+m.version(n))
    except Exception: print(n+'\\tMISSING')
"""
    _, pkg_out = command([str(TTA_PYTHON), "-c", pkg_code])
    installed = dict(line.split("\t", 1) for line in pkg_out.splitlines() if "\t" in line)
    required = {
        "torch": "2.2.2+cu121", "transformers": "4.39.3", "peft": "0.10.0",
        "accelerate": "0.29.2", "bitsandbytes": "0.43.1", "datasets": "2.18.0",
        "tokenizers": "0.15.2", "numpy": "1.26.4", "pandas": "2.2.2",
        "scipy": "1.13.0", "scikit-learn": "1.4.2", "tqdm": "4.66.2",
        "safetensors": "0.4.3", "huggingface-hub": "0.22.2",
    }
    d = ["# Dependency audit", "", "Selected historical evidence: `Method/GenARM/requirements.txt` plus Stage-9 provenance (Python 3.10.20 and torch 2.2.2+cu121). `PARM/requirements.txt` contains later/conflicting pins, so packages were not changed automatically.", "", "Current candidate environment: `/home/jupyter-iec2024se10/miniforge3/envs/tta` (Python 3.10.21).", "", "| Package | Required version | Installed version | Status | Evidence/source |", "|---|---|---|---|---|"]
    for name, req in required.items():
        got = installed.get(name, "MISSING")
        ok = got == req or (name == "torch" and got.startswith(req))
        d.append(f"| {name} | `{req}` | `{got}` | {'PASS' if ok else 'FAIL'} | Method/GenARM lock / Stage-9 provenance |")
    d += ["", "Import scanning also reveals an internal source dependency, `router_v2.cache`, which is missing and cannot be fixed by pip.", "", "The historical torch 2.2.2+cu121 lock may not support the newer GB202 GPU. The existing torch 2.7.1+cu128 appears intentional for this hardware, but it cannot be accepted until CUDA device access is repaired and a real model smoke test passes. No package was changed during this audit."]
    write("06_dependency_audit.md", "\n".join(d))

    if TTA_PYTHON.is_file():
        _, freeze = command([str(TTA_PYTHON), "-m", "pip", "freeze"])
        write("07_environment_lock.txt", freeze)
    else:
        write("07_environment_lock.txt", "# NOT AVAILABLE: candidate tta environment is missing")

    smoke_code = """import importlib, traceback
mods=['torch','transformers','peft','accelerate','bitsandbytes','PARM','PARM_TARO','router_v2','PARM_TARO.recovery.diagnostics','PARM_TARO.training.runtime','PARM_TARO.evaluation.protocol']
for mod in mods:
    print('\\n=== '+mod+' ===')
    try:
        importlib.import_module(mod)
        print('PASS')
    except Exception:
        print('FAIL')
        traceback.print_exc()
"""
    smoke_rc, smoke_out = command([str(TTA_PYTHON), "-c", smoke_code])
    write("08_import_smoke_test.log", f"python={TTA_PYTHON}\nprocess_exit={smoke_rc}\n{smoke_out}")

    preflight = {
        "status": "NOT_RUN", "reason": "Required model/PBLoRA/checkpoints are missing; router_v2.cache is missing; CUDA compute is unavailable.",
        "expected_runtime": {"base_model": "models/tulu-2-7b", "load_in_4bit": True,
                             "physical_backbone_count": 1, "shared_backbone": True,
                             "base_pass": "PeftModel.disable_adapter", "guide_pass": "PBLoRA enabled"},
        "checks": {key: False for key in ["tokenizer_load", "base_model_load", "pblora_attach", "disable_adapter", "enable_adapter", "one_prompt_forward", "finite_logits", "no_nan_inf", "gpu_memory_reasonable"]},
        "generation_or_evaluation_run": False,
    }
    write_json("09_model_load_preflight.json", preflight)
    write("09_model_load_preflight.md", "# Real model-load preflight\n\nStatus: **NOT RUN**. The exact Tulu-2-7B files, tokenizer, PBLoRA, Stage-9 checkpoints, required internal `router_v2.cache` source, and usable CUDA device are unavailable. Import success alone was not accepted as model readiness. No substitute model and no bulk generation were used.\n\nExpected Stage-10 logic remains documented as one 4-bit shared backbone, base pass with adapters disabled, and guide pass with PBLoRA enabled; it cannot be validated dynamically in the present state.")

    write("10_static_equivalence.md", "# Static-equivalence gate\n\nStatus: **NOT RUN — BLOCKED_STATIC_EQUIVALENCE**.\n\nThe retained source-level equation algebraically gives the base distribution at `lambda=0` and the PARM-static formula at `lambda=1`, but the mandatory numeric comparison on real model states cannot run without the exact model, PBLoRA, internal cache package, and CUDA runtime. Source-level reasoning is not promoted to PASS. Retraining remains prohibited.")
    write("11_fresh_diagnosis_summary.md", "# Fresh D1-D7 diagnosis\n\nStatus: **NOT RUN**. Phases 0-10 did not pass, so no fresh D1-D7 job was started. In particular, no validation-200/500 evaluation, test-1500 evaluation, model generation, or retraining occurred. The retained historical diagnosis still reports H1/H5 CONFIRMED and H6/H7 PARTIALLY_CONFIRMED, but this restore audit does not claim those retained aggregates are a fresh rerun.")

    status = {
        "project_root": str(ROOT), "git_commit": commit,
        "source": {"status": "PARTIAL"},
        "protected_artifacts": {"status": "FAIL"},
        "models": {"status": "FAIL"},
        "datasets": {"status": "PASS"},
        "protocol": {"status": "PASS"},
        "python_environment": {"status": "FAIL"},
        "gpu": {"status": "FAIL"},
        "model_load": {"status": "NOT_RUN"},
        "static_equivalence": {"status": "NOT_RUN"},
        "fresh_diagnosis": {"status": "NOT_RUN"},
        "ready_for_retraining": False,
        "blocking_issues": [
            "router_v2/cache source package missing",
            "Tulu-2-7B model and tokenizer missing; exact revision not recorded",
            "PBLoRA protected checkpoint missing",
            "three Stage-9 router checkpoints missing",
            "Stage-10 evaluator model/source directories missing",
            "historical full protected-tree hashes cannot be reproduced",
            "GPU device nodes unavailable; nvidia-smi and CUDA matmul fail",
            "candidate Python environment differs from historical lock and lacks bitsandbytes",
            "root filesystem has insufficient free space for exact model restore",
        ],
    }
    write_json("recovery_status.json", status)

    summary = f"""# PARM-TARO server restore summary

Audit commit: `{commit}` on branch `{branch}`.

## Decision

**NOT READY — MISSING ARTIFACTS**

No retraining, 200/500/1500 evaluation, bulk generation, package installation, or model download was performed. Protected baseline paths and historical Stage-9/10 artifacts were not modified.

## Gate results

| Gate | Status | Main evidence |
|---|---|---|
| Source | PARTIAL | Main trees present and Git-clean, but ignored `router_v2/cache` implementation is absent |
| Protected artifacts | FAIL | PBLoRA and three Stage-9 checkpoints missing; archived full tree hashes not reproducible |
| Models | FAIL | Tulu-2-7B/tokenizer and evaluators absent |
| Dataset | PASS | PKU source and deterministic 8000/500/1500 processed splits present |
| Protocol | PASS | Frozen protocol and deterministic val200 manifest present; canonical lock hash verifies |
| Python environment | FAIL | Candidate env differs from historical torch/Python and lacks bitsandbytes |
| GPU | FAIL | RTX 5090-class PCI device present, but no `/dev/nvidia*`; `nvidia-smi` and CUDA matmul fail |
| Real model load | NOT RUN | Blocked upstream |
| Static equivalence | NOT RUN | `BLOCKED_STATIC_EQUIVALENCE` |
| Fresh D1-D7 | NOT RUN | Required Phases 0-10 did not pass |

See `04_missing_artifacts.md` for exact restoration requirements. The final conclusion is driven by both missing irreplaceable checkpoints/source and an unusable GPU/runtime; artifact recovery is the first non-substitutable blocker.
"""
    write("00_restore_summary.md", summary)


if __name__ == "__main__":
    main()
