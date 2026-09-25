#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
def root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p/".git").exists() and (p/"PARM_TARO").exists(): sys.path.insert(0, str(p)); return p
    raise RuntimeError("project root not found")
ROOT = root()
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import bootstrap_import_paths, subprocess_environment
bootstrap_import_paths(ROOT)
from PARM_TARO.pareto_scalarization_probe.config import ADAPTER, BASE, COST, DENSE_OUT, MAX_NEW_TOKENS, MIN_FREE_MIB, NUM_GRADIENT_BATCHES, OUTPUT_ROOT, PROTOCOL, REWARD, TRAIN
from PARM_TARO.pareto_scalarization_probe.io import atomic_json
from PARM_TARO.pareto_scalarization_probe.manifest import build_manifest
from PARM_TARO.pareto_scalarization_probe.tiny_training import prepare

def preflight() -> dict:
    cases = build_manifest(); prep = prepare(); checks = {"base": (BASE/"config.json").is_file(), "adapter": (ADAPTER/"adapter_model.safetensors").is_file(),
        "reward": REWARD.is_dir(), "cost": COST.is_dir(), "protocol": PROTOCOL.is_file(), "train": TRAIN.is_file()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "loads_model": False, "fresh_prompts": 20, "alpha_grid": 21,
              "expected_phase_a_generations": len(cases), "gradient_batches": NUM_GRADIENT_BATCHES, "phase_c": prep["status"], "checks": checks}
    atomic_json(OUTPUT_ROOT/"preflight.json", result); return result
def child(phase: str, args: argparse.Namespace) -> None:
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--resume", "--min-free-mib", str(args.min_free_mib),
                    "--max-new-tokens", str(args.max_new_tokens)], cwd=ROOT, env=subprocess_environment(ROOT), check=True)
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--phase", choices=("all","generate","reward","cost","analyze","gradients","decide","prepare-c"), default="all")
    p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--min-free-mib", type=int, default=int(os.environ.get("MIN_FREE_MIB", MIN_FREE_MIB)))
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS); args=p.parse_args(); audit=preflight()
    if args.dry_run: print(json.dumps(audit, indent=2, sort_keys=True)); return 0 if audit["status"]=="PASS" else 2
    if audit["status"] != "PASS": raise SystemExit("Preflight failed")
    if args.phase == "all":
        for phase in ("generate","reward","cost","analyze","gradients","decide"): child(phase,args)
    elif args.phase == "generate":
        from PARM_TARO.pareto_scalarization_probe.generation import run; print(json.dumps(run(build_manifest(), resume=args.resume, min_free_mib=args.min_free_mib, max_new_tokens=args.max_new_tokens), indent=2))
    elif args.phase in ("reward","cost"):
        from PARM_TARO.pareto_scalarization_probe.scoring import run; print(json.dumps(run(args.phase, resume=args.resume, min_free_mib=args.min_free_mib), indent=2))
    elif args.phase == "analyze":
        from PARM_TARO.pareto_scalarization_probe.analysis import run; print(json.dumps(run(), indent=2))
    elif args.phase == "gradients":
        from PARM_TARO.pareto_scalarization_probe.gradient_audit import run; print(json.dumps(run(resume=args.resume, min_free_mib=args.min_free_mib), indent=2))
    elif args.phase == "decide":
        from PARM_TARO.pareto_scalarization_probe.decision import run; print(json.dumps(run(), indent=2))
    else: print(json.dumps(prepare(), indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
