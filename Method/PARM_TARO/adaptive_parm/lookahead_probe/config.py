"""Frozen paths and constants for the offline lookahead probe."""

from pathlib import Path


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError("Cannot locate project root")


ROOT = project_root()
SOURCE = ROOT / "results/parm_taro/adaptive_parm/04_token_headroom"
RICH = ROOT / "results/parm_taro/adaptive_parm/05_rich_state_probe"
OUT = ROOT / "results/parm_taro/adaptive_parm/06_lookahead_probe"
CACHE = OUT / "score_cache"
BASE = ROOT / "models/tulu-2-7b"
ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"

HORIZONS = (1, 2, 4, 8, 16, 32)
SCORE_FAMILIES = ("parm", "base", "ratio")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE_WEIGHT = 1.0
UTILITY_EPSILON = 1e-6

