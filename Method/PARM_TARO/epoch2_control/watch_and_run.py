#!/usr/bin/env python3
from __future__ import annotations
import fcntl,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"results/parm_taro/recovery/reproduced/pblora_epoch2_control"
PY="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"; ENTRY=ROOT/"PARM_TARO/epoch2_control/run_control.py"
def log(message:str):
    line=f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')} {message}"; print(line,flush=True)
    with (OUT/"watcher.log").open("a") as h:h.write(line+"\n")
def gpu():
    line=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().splitlines()[0]
    return tuple(int(x.strip()) for x in line.split(","))
def main()->int:
    OUT.mkdir(parents=True,exist_ok=True);threshold=int(os.environ.get("MIN_FREE_MIB","24000"));interval=int(os.environ.get("CHECK_INTERVAL","15"));stable=0
    lock=(OUT/"watcher.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print("Epoch-2 watcher already active");return 2
    while True:
        try:free,used,util=gpu();stable=stable+1 if free>=threshold else 0;log(f"free={free} used={used} util={util}% stable={stable}/2")
        except Exception as error:stable=0;log(f"GPU query failed: {error!r}")
        if stable<2:time.sleep(interval);continue
        command=[PY,str(ENTRY),"--phase","all","--resume","--min-free-mib","6000"]
        with (OUT/"last_attempt.log").open("w") as h:result=subprocess.run(command,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT)
        if result.returncode==0:log("Epoch-2 control and paired evaluation completed");return 0
        content=(OUT/"last_attempt.log").read_text(errors="replace")
        if re.search(r"CUDA out of memory|OutOfMemoryError|LOW_VRAM_BLOCKED",content,re.I):stable=0;log("GPU memory contention; will resume");time.sleep(60);continue
        log(f"Non-memory failure exit={result.returncode}; stopping");return result.returncode
if __name__=="__main__":raise SystemExit(main())
