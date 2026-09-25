#!/usr/bin/env python3
"""Clean-child import preflight for every epoch-2 evaluation phase."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import required_python_paths,subprocess_environment

def run()->dict:
    paths=required_python_paths(ROOT);code="""
import json
from router_v2.device import resolve_device
from PARM_TARO.training.runtime import FrozenBaseModelView
from PARM_TARO.pareto_scalarization_probe.generation import generate_one
print(json.dumps({
  'router_device_module': resolve_device.__module__,
  'frozen_base_module': FrozenBaseModelView.__module__,
  'generation_module': generate_one.__module__,
}))
"""
    completed=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=subprocess_environment(ROOT),text=True,capture_output=True)
    result={"status":"PASS" if completed.returncode==0 else "FAIL","returncode":completed.returncode,
        "ordered_package_roots":[str(path) for path in paths],"stdout":completed.stdout.strip(),"stderr":completed.stderr.strip()}
    output=ROOT/"results/parm_taro/recovery/reproduced/pblora_epoch2_control/import_regression.json"
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if completed.returncode:raise RuntimeError(f"Evaluation child import regression failed: {result}")
    return result

if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
