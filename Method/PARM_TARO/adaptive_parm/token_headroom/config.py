"""Frozen experiment configuration."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError("Cannot locate project root")


ROOT = project_root()
OUT = ROOT / "results/parm_taro/adaptive_parm/04_token_headroom"
SOURCE_MANIFEST = ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl"
PROTOCOL = ROOT / "results/parm_taro/evaluation/protocol/protocol_lock.json"
PHASE09 = ROOT / "results/parm_taro/recovery/low_vram_suite/09_metrics_summary.json"
BASE = ROOT / "models/tulu-2-7b"
ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
REWARD = ROOT / "models/beaver-7b-v1.0-reward"
COST = ROOT / "models/beaver-7b-v1.0-cost"
SAFE_RLHF = ROOT / "models/safe-rlhf-source"

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE_WEIGHT = 1.0
MAX_REFERENCE_TOKENS = 64
ROLLOUT_HORIZON = 32
STATE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOP_K = 10
SEED = 42
LEAKAGE_TOLERANCE = 1e-4

GENERATION_STATE = OUT / "generation_state"
LEAKAGE_STATE = OUT / "leakage_state"
SCORE_STATE = OUT / "score_state"
