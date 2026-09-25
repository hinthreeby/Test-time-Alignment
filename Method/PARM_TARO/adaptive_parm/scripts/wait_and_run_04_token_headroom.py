#!/usr/bin/env python3
"""Wait for shared GPU memory, then run/resume token-headroom phase 04."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import subprocess_environment

OUT = ROOT / "results/parm_taro/adaptive_parm/04_token_headroom"
ENTRYPOINT = ROOT / "PARM_TARO/adaptive_parm/token_headroom/run_token_headroom.py"
DEFAULT_PYTHON = Path("/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10")


def gpu_status() -> tuple[int, int, int]:
    command = ["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "--id=0"]
    output = subprocess.check_output(command, text=True, timeout=15).strip().splitlines()[0]
    free, used, utilization = (int(part.strip()) for part in output.split(","))
    return free, used, utilization


def log(path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    interval = int(os.environ.get("CHECK_INTERVAL", "5"))
    threshold = int(os.environ.get("MIN_FREE_MIB", "18000"))
    stable_needed = int(os.environ.get("STABLE_CHECKS", "2"))
    python = Path(os.environ.get("GPU_PYTHON", str(DEFAULT_PYTHON)))
    OUT.mkdir(parents=True, exist_ok=True)
    command = [str(python), str(ENTRYPOINT), "--phase", "all", "--num-prompts", str(args.num_prompts), "--resume"]
    if args.dry_run:
        print("DRY RUN")
        print(f"threshold={threshold} MiB interval={interval}s stable_checks={stable_needed}")
        print("command:", " ".join(command))
        return 0
    lock_handle = (OUT / "watcher.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another token-headroom watcher already holds watcher.lock", file=sys.stderr)
        return 2
    (OUT / "watcher.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    wait_log = OUT / "gpu_wait.log"
    attempt_log = OUT / "last_attempt.log"
    stable = 0
    while True:
        try:
            free, used, utilization = gpu_status()
        except Exception as error:
            stable = 0
            log(wait_log, f"GPU query failed: {error!r}; stable=0/{stable_needed}")
            time.sleep(interval)
            continue
        stable = stable + 1 if free >= threshold else 0
        log(wait_log, f"free={free} MiB used={used} MiB util={utilization}% stable={stable}/{stable_needed}")
        if stable < stable_needed:
            time.sleep(interval)
            continue
        environment = subprocess_environment(ROOT)
        environment.update({"CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "GPU_PYTHON": str(python)})
        with attempt_log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode == 0:
            log(wait_log, "Experiment completed successfully")
            return 0
        contents = attempt_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"CUDA out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|LOW_VRAM_BLOCKED", contents, re.IGNORECASE):
            log(wait_log, f"GPU memory became unavailable (exit={process.returncode}); returning to wait loop")
            stable = 0
            time.sleep(max(interval, 30))
            continue
        log(wait_log, f"Non-OOM failure (exit={process.returncode}); stopping. See {attempt_log}")
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
