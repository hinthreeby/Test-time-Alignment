from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError("Cannot locate project root")


ROOT=project_root()
SOURCE=ROOT/"results/parm_taro/adaptive_parm/04_token_headroom"
TIE=SOURCE/"tie_aware"
OUT=ROOT/"results/parm_taro/adaptive_parm/05_rich_state_probe"
CACHE=OUT/"state_cache"
ADAPTER=ROOT/"results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
WEIGHTS=(0.0,0.25,0.5,0.75,1.0)
TOP_K_VALUES=(16,32,64)
PCA_DIMS=(8,16,32)
RIDGE_ALPHAS=(0.1,1.0,10.0,100.0)
DELTA_GRID=(0.0,1e-4,5e-4,1e-3,2.5e-3,5e-3,1e-2,2e-2,5e-2)
EPSILON=1e-6
SEED=42
