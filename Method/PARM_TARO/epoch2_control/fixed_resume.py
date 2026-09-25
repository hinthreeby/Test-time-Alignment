#!/usr/bin/env python3
"""Epoch-2 resume entrypoint with scoped trusted-state compatibility."""
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(PROJECT_ROOT))
import torch
import PARM_TARO.recovery.pblora_resume_entrypoint as recovery
from PARM_TARO.epoch2_control.resume_compat import TrustedTrainerStateLoadShim,install_training_audit,preflight

def option(name:str)->Path:
    try:return Path(sys.argv[sys.argv.index(name)+1]).resolve()
    except (ValueError,IndexError) as error:raise RuntimeError(f"Required option missing: {name}") from error

METHOD_ROOT=PROJECT_ROOT/"Method"
if not (METHOD_ROOT/"PARM/code/training/train_pref_arm.py").is_file():raise RuntimeError("Canonical Method/PARM trainer is unavailable")
checkpoint=option("--resume-from-checkpoint");output_dir=option("--recovery-output-dir")
recovery.ROOT=METHOD_ROOT
preflight(checkpoint,output_dir,torch)
shim=TrustedTrainerStateLoadShim(checkpoint,torch);shim.install()
install_training_audit(recovery,checkpoint,output_dir,shim)

if __name__=="__main__":recovery.main()
