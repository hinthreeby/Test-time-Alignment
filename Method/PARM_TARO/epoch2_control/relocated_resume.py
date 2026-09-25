#!/usr/bin/env python3
"""Relocation-safe delegation to the existing audited recovery entrypoint."""
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT_ROOT))
import PARM_TARO.recovery.pblora_resume_entrypoint as recovery

METHOD_ROOT=PROJECT_ROOT/"Method"
if not (METHOD_ROOT/"PARM/code/training/train_pref_arm.py").is_file():
    raise RuntimeError("Canonical Method/PARM trainer is unavailable")
recovery.ROOT=METHOD_ROOT

if __name__=="__main__":
    recovery.main()
