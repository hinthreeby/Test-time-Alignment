from pathlib import Path

def project_root()->Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate/".git").exists() and (candidate/"PARM_TARO").exists():return candidate
    raise RuntimeError("Cannot locate project root")

ROOT=project_root();OUT=ROOT/"results/parm_taro/adaptive_parm/07_disagreement_gate"
VALIDATION=ROOT/"dataset/parm_taro/validation.json";DIAGNOSTIC_MANIFEST=ROOT/"results/parm_taro/recovery/feasibility60/manifest.jsonl"
BASE=ROOT/"models/tulu-2-7b";ADAPTER=ROOT/"results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
REWARD=ROOT/"models/beaver-7b-v1.0-reward";COST=ROOT/"models/beaver-7b-v1.0-cost";PROTOCOL=ROOT/"results/parm_taro/evaluation/protocol/protocol_lock.json"
GENERATION_STATE=OUT/"generation_state";SCORE_STATE=OUT/"score_state"
ALPHAS=((1.,0.),(.75,.25),(.5,.5),(.25,.75),(0.,1.));METHODS=("base","parm_fixed","midpoint","disagreement_gate")
SEED=42;MAX_NEW_TOKENS=64;MANIFEST_PROMPTS=200

