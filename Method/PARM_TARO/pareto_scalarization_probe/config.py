from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists(): return candidate
    raise RuntimeError("Cannot locate project root")

ROOT = project_root(); OUTPUT_ROOT = ROOT / "results/parm_taro/pareto_scalarization_probe"
DENSE_OUT = OUTPUT_ROOT / "01_dense_alpha"; GRADIENT_OUT = OUTPUT_ROOT / "02_gradient_conflict"; TINY_OUT = OUTPUT_ROOT / "03_tiny_training"
VALIDATION = ROOT / "dataset/parm_taro/validation.json"; TRAIN = ROOT / "dataset/parm_taro/train.json"
BASE = ROOT / "models/tulu-2-7b"; ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
REWARD = ROOT / "models/beaver-7b-v1.0-reward"; COST = ROOT / "models/beaver-7b-v1.0-cost"
PROTOCOL = ROOT / "results/parm_taro/evaluation/protocol/protocol_lock.json"
ALPHA_GRID = tuple((round(i / 20, 2), round(1 - i / 20, 2)) for i in range(21))
CANCELLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 42; NUM_PROMPTS = 20; NUM_GRADIENT_BATCHES = 20; GRADIENT_BATCH_SIZE = 1
MAX_NEW_TOKENS = 64; MIN_FREE_MIB = 6000
