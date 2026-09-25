#!/usr/bin/env python3
"""Shared-GPU watcher for the complete Phase A+B probe."""
from __future__ import annotations
import argparse, fcntl, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import subprocess_environment
OUT=ROOT/"results/parm_taro/pareto_scalarization_probe"; ENTRY=ROOT/"PARM_TARO/pareto_scalarization_probe/run_probe.py"
PYTHON="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"
def gpu():
    line=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=15).strip().splitlines()[0]
    return tuple(int(x.strip()) for x in line.split(","))
def log(path:Path,message:str):
    value=f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')} {message}"; print(value,flush=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(value+"\n")
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args(); threshold=int(os.environ.get("MIN_FREE_MIB","6000")); interval=int(os.environ.get("CHECK_INTERVAL","10")); needed=int(os.environ.get("STABLE_CHECKS","2"))
    command=[PYTHON,str(ENTRY),"--phase","all","--resume","--min-free-mib",str(threshold)]; OUT.mkdir(parents=True,exist_ok=True)
    if args.dry_run: print(" ".join(command)); return 0
    lock=(OUT/"watcher.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print("Another Pareto probe watcher is active",file=sys.stderr); return 2
    (OUT/"watcher.pid").write_text(str(os.getpid())+"\n"); stable=0; waitlog=OUT/"gpu_wait.log"; attempt=OUT/"last_attempt.log"
    while True:
        try:
            free,used,util=gpu(); stable=stable+1 if free>=threshold else 0; log(waitlog,f"free={free} MiB used={used} MiB util={util}% stable={stable}/{needed}")
        except Exception as error: stable=0; log(waitlog,f"GPU query failed: {error!r}")
        if stable<needed: time.sleep(interval); continue
        env=subprocess_environment(ROOT); env.update({"CUDA_VISIBLE_DEVICES":"0","TOKENIZERS_PARALLELISM":"false","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
        with attempt.open("w",encoding="utf-8") as handle: result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode==0: log(waitlog,"Phase A+B probe completed"); return 0
        if re.search(r"CUDA out of memory|OutOfMemoryError|LOW_VRAM_BLOCKED",attempt.read_text(errors="replace"),re.I): stable=0; log(waitlog,"Memory contention; waiting to resume"); time.sleep(max(30,interval)); continue
        log(waitlog,f"Non-memory failure exit={result.returncode}; stopping"); return result.returncode
if __name__=="__main__": raise SystemExit(main())
