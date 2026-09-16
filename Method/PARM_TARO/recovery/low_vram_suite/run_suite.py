"""List suite phases, report status, or suggest (never execute) the next command."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from PARM_TARO.recovery.low_vram_suite.common import OUT
PYTHON="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"
PHASES=(
 ("00","PREFLIGHT","00_preflight.py","00_preflight.json","--dry-run"),
 ("01","ARTIFACT AUDIT","01_artifact_audit.py","01_artifact_audit.json","--dry-run"),
 ("02","GUIDE UTILITY","02_guide_utility.py","02_guide_utility_summary.json","--dry-run"),
 ("03","LAMBDA60","03_lambda60_diagnostic.py","03_summary.json","--dry-run"),
 ("04","HEADROOM","04_headroom_analysis.py","04_headroom_summary.json","--dry-run"),
 ("05","EXPOSURE SHIFT","05_exposure_shift.py","05_exposure_shift_summary.json","--dry-run"),
 ("06","GENERATION SMOKE","06_generation_smoke.py","06_generation_summary.json","--dry-run"),
 ("07","SCORER SANITY","07_scorer_sanity.py","07_scorer_summary.json","--preflight-only"),
 ("08","FULL GENERATE","08_feasibility60_generate.py","08_generation_summary.json","--dry-run"),
 ("09","FULL SCORE","09_feasibility60_score.py","09_metrics_summary.json","--dry-run"),
 ("10","DECISION","10_decision_report.py","10_decision.json","--dry-run"),
)
def status(path):
    if not path.is_file(): return "NOT RUN"
    try:
        value=json.loads(path.read_text()); raw=str(value.get("status","COMPLETE")).upper()
        return "BLOCKED" if "BLOCK" in raw or raw=="FAIL" else raw
    except Exception: return "INVALID OUTPUT"
def command(file,args=""):
    return f"CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PYTHON} PARM_TARO/recovery/low_vram_suite/{file} {args}".strip()
def main():
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(required=True); group.add_argument("--list",action="store_true"); group.add_argument("--status",action="store_true"); group.add_argument("--next",action="store_true"); args=p.parse_args()
    if args.list:
        for number,name,*_ in PHASES: print(f"{number} {name}")
    elif args.status:
        for number,name,_,output,_ in PHASES: print(f"{number} {name:20} {status(OUT/output)}")
    else:
        for number,name,file,output,dry in PHASES:
            if status(OUT/output) in ("NOT RUN","BLOCKED","INVALID OUTPUT"):
                suffix="--resume" if number in ("06","08","09") and (OUT/output).exists() else ""
                print(f"NEXT: {number} {name}\n{command(file,suffix)}"); break
        else: print(f"NEXT: review decision\n{command('10_decision_report.py')}")
    return 0
if __name__=="__main__": raise SystemExit(main())
