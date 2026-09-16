import os
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/jupyter-iec2024se10/Test-time-Alignment")
PYTHON = "/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"

MIN_FREE_MIB = 18000
CHECK_INTERVAL = 5

SAFE_RLHF = ROOT / "models/safe-rlhf-source"
GEN_FILE = ROOT / "results/parm_taro/recovery/low_vram_suite/08_generations.jsonl"

LOG_DIR = ROOT / "results/parm_taro/recovery/low_vram_suite"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "phase_07_09_10_auto.log"

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def gpu_free_mib():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def wait_for_gpu():
    log(
        f"Waiting for GPU0 free VRAM >= {MIN_FREE_MIB} MiB "
        f"(check every {CHECK_INTERVAL}s)"
    )

    while True:
        try:
            free = gpu_free_mib()
            log(f"GPU0 free={free} MiB")

            if free >= MIN_FREE_MIB:
                time.sleep(2)
                free2 = gpu_free_mib()

                if free2 >= MIN_FREE_MIB:
                    log(f"GPU READY: {free2} MiB free")
                    return

        except Exception as e:
            log(f"GPU check error: {e}")

        time.sleep(CHECK_INTERVAL)


def run_phase(name, args):
    log(f"START {name}")
    log("CMD: " + " ".join(args))

    with open(LOG_FILE, "a") as f:
        p = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if p.returncode != 0:
        log(f"{name} FAILED returncode={p.returncode}")
        sys.exit(p.returncode)

    log(f"{name} PASS")


# ---------- prerequisites ----------

if not SAFE_RLHF.exists():
    log(f"BLOCKED: missing {SAFE_RLHF}")
    sys.exit(2)

if not GEN_FILE.exists():
    log(f"BLOCKED: missing {GEN_FILE}")
    sys.exit(2)

with open(GEN_FILE) as f:
    n_generations = sum(1 for x in f if x.strip())

log(f"Phase-08 generations found: {n_generations}")

if n_generations < 600:
    log(f"BLOCKED: expected >=600 generations, found {n_generations}")
    sys.exit(2)


# ---------- Phase 07 ----------

wait_for_gpu()

run_phase(
    "PHASE 07 SCORER SANITY",
    [
        PYTHON,
        "PARM_TARO/recovery/low_vram_suite/07_scorer_sanity.py",
        "--limit", "10",
    ],
)

time.sleep(5)


# ---------- Phase 09 ----------

wait_for_gpu()

run_phase(
    "PHASE 09 FEASIBILITY60 SCORE",
    [
        PYTHON,
        "PARM_TARO/recovery/low_vram_suite/09_feasibility60_score.py",
    ],
)


# ---------- Phase 10 ----------

run_phase(
    "PHASE 10 DECISION REPORT",
    [
        PYTHON,
        "PARM_TARO/recovery/low_vram_suite/10_decision_report.py",
    ],
)

log("ALL DONE: PHASE 07 -> 09 -> 10 COMPLETE")
