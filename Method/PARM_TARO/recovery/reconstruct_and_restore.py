"""State-driven, CPU-only PARM-TARO source/public-artifact recovery runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
WORK = ROOT / "results/parm_taro/recovery/reconstruction"
STATE_PATH = WORK / "state/recovery_state.json"
PYTHON = Path(os.environ.get("TTA_PYTHON", sys.executable))
PHASES = (
    "inventory", "reconstruct_cache", "discover_artifacts", "download_public",
    "verify_source", "import_test", "finalize",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def initial_state() -> dict[str, Any]:
    return {"schema_version": 1, "device": "cpu", "updated_at": utc_now(),
            "phases": {name: {"status": "pending"} for name in PHASES}}


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return initial_state()
    value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    for phase in PHASES:
        value.setdefault("phases", {}).setdefault(phase, {"status": "pending"})
    return value


def run(args: list[str], log_name: str | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(args, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_name:
        log = WORK / "logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {utc_now()} =====\n{proc.stdout}")
    return proc.returncode, proc.stdout


def inventory() -> tuple[str, dict[str, Any]]:
    commands = {
        "git_status": ["git", "status", "--short"],
        "branch": ["git", "branch", "--show-current"],
        "commit": ["git", "rev-parse", "HEAD"],
        "remotes": ["git", "remote", "-v"],
    }
    result: dict[str, Any] = {"project_root": str(ROOT), "commands": {}}
    for name, command in commands.items():
        code, output = run(command)
        result["commands"][name] = {"exit_code": code, "output": output.rstrip()}
    result["required_directories"] = {
        name: (ROOT / name).is_dir()
        for name in ("PARM", "PARM_TARO", "router", "router_v2", "Method/RAD", "results/parm_taro")
    }
    atomic_json(WORK / "state/inventory.json", result)
    return ("done" if all(result["required_directories"].values()) else "failed", result)


def reconstruct_cache() -> tuple[str, dict[str, Any]]:
    required = [ROOT / "router_v2/cache" / name for name in
                ("__init__.py", "features.py", "io.py", "schema.py", "inventory.py")]
    marker = "# RECONSTRUCTED SOURCE\n# Original file was lost because router_v2/cache was gitignored."
    checks = {str(path.relative_to(ROOT)): path.is_file() and path.read_text(encoding="utf-8").startswith(marker)
              for path in required}
    return ("done" if all(checks.values()) else "failed",
            {"classification": "RECONSTRUCTED_FROM_REPO_EVIDENCE", "files": checks})


def discover_artifacts() -> tuple[str, dict[str, Any]]:
    value = {
        "tulu": {"id": "allenai/tulu-2-7b", "revision": "3c6e328ae91fabdd0daf09de16887de9615c1f66",
                 "evidence": "current revision matches both retained 7B weight SHA-256 values and checkpoint-index SHA-256"},
        "beaver_reward": {"id": "PKU-Alignment/beaver-7b-v1.0-reward", "revision": "375cd6a9f0d7e339d2199b05ba129a4a8906596d",
                          "evidence": "remote metadata reconstructs exact archived 14-file tree SHA-256 3410256f..."},
        "beaver_cost": {"id": "PKU-Alignment/beaver-7b-v1.0-cost", "revision": "c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28",
                        "evidence": "remote metadata reconstructs exact archived 14-file tree SHA-256 418b9416..."},
        "safe_rlhf": {"id": "https://github.com/PKU-Alignment/safe-rlhf.git", "revision": None,
                      "evidence": "repo ID is explicit; historical clone recorded no commit and archived tree hash included non-reproducible .git metadata"},
    }
    atomic_json(WORK / "state/public_artifacts.json", value)
    return "done", value


def download_public() -> tuple[str, dict[str, Any]]:
    script = WORK / "scripts/download_public_artifacts.sh"
    code, output = run(["bash", str(script)], "download_runner.log")
    return ("done" if code == 0 else "blocked",
            {"exit_code": code, "reason": "insufficient disk or unresolved pinned artifact" if code else None,
             "tail": output[-3000:]})


def verify_source() -> tuple[str, dict[str, Any]]:
    commands = [
        [str(PYTHON), "-m", "pytest", "tests/recovery/test_router_v2_cache_reconstruction.py", "-v"],
        [str(PYTHON), "-m", "pytest", "router_v2/tests/test_feature_cache.py", "-v"],
        [
            str(PYTHON), "-m", "pytest",
            "PARM_TARO/tests/test_parm_taro.py::StaticEquationTests::test_static_equation_matches_vendored_operator_raw_sum",
            "PARM_TARO/tests/test_parm_taro.py::VendoredAndProtectedTreeTests::test_original_parm_generation_module_loads_read_only",
            "PARM_TARO/tests/test_parm_taro.py::VendoredAndProtectedTreeTests::test_vendored_parm_dependencies_resolve_in_proven_environment",
            "-v",
        ],
    ]
    results = []
    for index, command in enumerate(commands):
        code, output = run(command, "source_verification.log")
        results.append({"command": command, "exit_code": code, "tail": output[-2000:]})
    return ("done" if all(item["exit_code"] == 0 for item in results) else "failed",
            {"tests": results})


CPU_PREFLIGHT = r'''import json
import torch
import transformers, accelerate
import router_v2, PARM_TARO
from router_v2.cache import schema, io, features, inventory
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import build_smart_router
from PARM_TARO.recovery.objectives import ResidualLambdaConfig, residual_lambda
from PARM_TARO.recovery.diagnostics import static_equivalence
from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime

assert torch.cuda.is_available() is False
vendored=activate_vendored_parm_runtime()
import peft
assert '/PARM/peft/src/peft/' in vendored.origins['peft']
assert '/PARM/language-model-arithmetic/src/model_arithmetic/' in vendored.origins['model_arithmetic']
config=SmartRouterConfig.load_json('PARM_TARO/configs/router_v2_alpha_tulu2.json')
model=build_smart_router(config).cpu().eval()
base=torch.randn(1,2,config.vocab_size)
guide=torch.randn(1,2,config.vocab_size)
out=model.forward_from_full_logits(base,guide,position=torch.tensor([[0,1]]),selected_score=torch.zeros(1,2),preference=torch.tensor([[0.5,0.5]]))
assert torch.isfinite(out.lambda_t).all()
assert torch.all(out.gate >= config.lambda_eps) and torch.all(out.gate <= 1-config.lambda_eps)
raw=torch.tensor([-2.0,0.0,2.0])
lam=residual_lambda(raw,ResidualLambdaConfig(lambda_0=1.0,rho=0.25))
assert torch.isfinite(lam).all() and lam[1].item()==1.0
eq=static_equivalence(torch.randn(4,19),torch.randn(4,19),torch.zeros(4,1),torch.ones(4,1))
assert eq.selected_token_match_rate_lambda_0==1.0 and eq.selected_token_match_rate_lambda_1==1.0
print('torch',torch.__version__)
print('cuda visible:',torch.cuda.is_available())
print('lambda range',float(out.lambda_t.min()),float(out.lambda_t.max()))
print('residual',lam.tolist())
print('IMPORT_PASS')
'''


def import_test() -> tuple[str, dict[str, Any]]:
    code, output = run([str(PYTHON), "-c", CPU_PREFLIGHT], "import_tests.log")
    return ("done" if code == 0 and "IMPORT_PASS" in output else "failed",
            {"exit_code": code, "output": output[-5000:], "cuda_expected": False})


def finalize() -> tuple[str, dict[str, Any]]:
    state = load_state()
    phase = state["phases"]
    downloads = run([str(PYTHON), str(WORK / "scripts/check_downloads.py"), "--all"])[1]
    blockers = []
    if phase["reconstruct_cache"]["status"] != "done" or phase["verify_source"]["status"] != "done":
        blockers.append("cache source reconstruction or tests incomplete")
    if phase["download_public"]["status"] != "done":
        blockers.append("public artifacts incomplete: disk capacity and Safe-RLHF revision")
    if phase["import_test"]["status"] != "done":
        blockers.append("CPU imports/static preflight incomplete")
    blockers.append("PBLoRA seed/base provenance conflict requires explicit reproduction protocol approval")
    status = {
        "host_gpu": {"status": "AVAILABLE", "model": "NVIDIA GeForce RTX 5090",
                     "driver": "580.105.08", "host_cuda": "13.0", "agent_access": False,
                     "agent_gpu_access": "UNAVAILABLE_BY_DESIGN",
                     "note": "GPU works on host but agent is intentionally CPU-only"},
        "source_reconstruction": {"status": "PASS" if phase["verify_source"]["status"] == "done" else "PARTIAL"},
        "public_artifacts": {"status": "PASS" if phase["download_public"]["status"] == "done" else "PARTIAL"},
        "custom_checkpoints": {"status": "NEEDS_REPRODUCTION"},
        "cpu_environment": {"status": "PASS" if phase["import_test"]["status"] == "done" else "FAIL"},
        "imports": {"status": "PASS" if phase["import_test"]["status"] == "done" else "FAIL"},
        "ready_for_gpu_reproduction": not blockers,
        "blocking_issues": blockers,
        "public_download_check": json.loads(downloads) if downloads.strip().startswith("[") else downloads,
        "updated_at": utc_now(),
    }
    atomic_json(WORK / "00_status.json", status)
    return "done", {"status_path": str(WORK / "00_status.json"), "ready": status["ready_for_gpu_reproduction"]}


ACTIONS: dict[str, Callable[[], tuple[str, dict[str, Any]]]] = {
    "inventory": inventory, "reconstruct_cache": reconstruct_cache,
    "discover_artifacts": discover_artifacts, "download_public": download_public,
    "verify_source": verify_source, "import_test": import_test, "finalize": finalize,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", *PHASES), default="all")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-phase", choices=PHASES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = list(PHASES) if args.phase == "all" else [args.phase]
    if args.force_phase and args.force_phase not in selected:
        selected = [args.force_phase]
    state = load_state()
    failures = False
    for name in selected:
        previous = state["phases"][name]["status"]
        forced = args.force_phase == name
        if args.resume and previous == "done" and not forced:
            print(f"SKIP done phase: {name}")
            continue
        if args.dry_run:
            print(f"DRY-RUN phase={name} previous={previous} forced={forced}")
            continue
        state["phases"][name] = {"status": "running", "started_at": utc_now()}
        state["updated_at"] = utc_now()
        atomic_json(STATE_PATH, state)
        try:
            status, detail = ACTIONS[name]()
        except Exception as exc:
            status, detail = "failed", {"error": repr(exc)}
        state = load_state()
        state["phases"][name] = {"status": status, "finished_at": utc_now(), "detail": detail}
        state["updated_at"] = utc_now()
        atomic_json(STATE_PATH, state)
        print(f"{name}: {status}")
        failures |= status in {"failed", "blocked"}
        if status == "failed" and args.phase != "all":
            break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
