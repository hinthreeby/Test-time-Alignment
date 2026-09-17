"""Wait for shared GPU 0, then resume prompt-router extraction and training."""

from __future__ import annotations

import fcntl
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/parm_taro/adaptive_parm/02_prompt_router"
PYTHON = Path(os.environ.get("PROMPT_ROUTER_PYTHON", "/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"))
INTERVAL = int(os.environ.get("CHECK_INTERVAL", "5"))
MIN_FREE = int(os.environ.get("MIN_FREE_MIB", "6000"))
STABLE_REQUIRED = 2


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_row() -> tuple[int, int, int]:
    command = ["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    values = [int(value.strip()) for value in subprocess.check_output(command, text=True, timeout=15).splitlines()[0].split(",")]
    return values[0], values[1], values[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), "-m", "PARM_TARO.adaptive_parm.prompt_router.run_prompt_router", "--phase", "all", "--device", "cuda", "--resume", "--min-free-mib", str(MIN_FREE)]
    if args.dry_run:
        print({"status":"DRY_RUN_PASS","python":str(PYTHON),"min_free_mib":MIN_FREE,"check_interval":INTERVAL,"stable_checks":STABLE_REQUIRED,"command":command,"training_started":False})
        return 0
    lock_handle = (OUTPUT / "watcher.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another prompt-router watcher already holds the lock", flush=True); return 2
    (OUTPUT / "watcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    wait_log = (OUTPUT / "gpu_wait.log").open("a", encoding="utf-8", buffering=1)
    stable = 0
    while True:
        try:
            free, used, utilization = gpu_row()
            stable = stable + 1 if free >= MIN_FREE else 0
            line = f"{timestamp()} free_mib={free} used_mib={used} util_pct={utilization} stable={stable}/{STABLE_REQUIRED}"
        except Exception as error:
            stable = 0; line = f"{timestamp()} GPU_QUERY_FAILED {type(error).__name__}: {error} stable=0/{STABLE_REQUIRED}"
        print(line, flush=True); wait_log.write(line + "\n")
        if stable >= STABLE_REQUIRED:
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES":"0","TOKENIZERS_PARALLELISM":"false","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
            attempt = OUTPUT / "last_attempt.log"
            with attempt.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                print(f"{timestamp()} PROMPT_ROUTER_COMPLETE", flush=True); return 0
            text = attempt.read_text(encoding="utf-8", errors="replace")
            if "CUDA out of memory" not in text and "torch.OutOfMemoryError" not in text:
                print(f"{timestamp()} NON_OOM_FAILURE exit={result.returncode}; see {attempt}", flush=True); return result.returncode or 1
            print(f"{timestamp()} CUDA_OOM; returning to wait loop", flush=True)
            stable = 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
