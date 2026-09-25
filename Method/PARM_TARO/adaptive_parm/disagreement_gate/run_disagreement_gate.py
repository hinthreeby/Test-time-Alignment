#!/usr/bin/env python3
"""Orchestrate isolated generation, sequential scoring, and CPU analysis."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
def root()->Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate/".git").exists() and (candidate/"PARM_TARO").exists():sys.path.insert(0,str(candidate));return candidate
    raise RuntimeError("Cannot locate project root")
ROOT=root()
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import bootstrap_import_paths,subprocess_environment
bootstrap_import_paths(ROOT)
from PARM_TARO.adaptive_parm.disagreement_gate.config import ADAPTER,BASE,COST,MAX_NEW_TOKENS,METHODS,OUT,PROTOCOL,REWARD
from PARM_TARO.adaptive_parm.disagreement_gate.manifest import build_manifest,select_cases
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_json
from PARM_TARO.recovery.low_vram_suite.common import resolve_safe_rlhf_source
def preflight(num_prompts:int)->dict:
    manifest=build_manifest();cases=select_cases(manifest,num_prompts);resolution=resolve_safe_rlhf_source();checks={"base":(BASE/"config.json").is_file(),"adapter":(ADAPTER/"adapter_model.safetensors").is_file(),"reward":(REWARD/"model.safetensors.index.json").is_file(),"cost":(COST/"model.safetensors.index.json").is_file(),"protocol":PROTOCOL.is_file(),"safe_rlhf":bool(resolution.get("ready"))};result={"status":"PASS" if all(checks.values()) else "FAIL","loads_model":False,"split":"validation","fresh_unique_prompts":num_prompts,"cases":len(cases),"methods":list(METHODS),"expected_generations":len(cases)*len(METHODS),"max_new_tokens":MAX_NEW_TOKENS,"diagnostic_prompts_excluded":True,"checks":checks};OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/"preflight.json",result);return result
def child(phase:str,args:argparse.Namespace)->None:
    command=[sys.executable,str(Path(__file__).resolve()),"--phase",phase,"--num-prompts",str(args.num_prompts),"--min-free-mib",str(args.min_free_mib),"--max-new-tokens",str(args.max_new_tokens),"--resume"];subprocess.run(command,cwd=ROOT,env=subprocess_environment(ROOT),check=True)
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("all","generate","reward","cost","analyze"),default="all");p.add_argument("--num-prompts",type=int,default=200);p.add_argument("--max-new-tokens",type=int,default=MAX_NEW_TOKENS);p.add_argument("--min-free-mib",type=int,default=int(os.environ.get("MIN_FREE_MIB","6000")));p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true");args=p.parse_args();audit=preflight(args.num_prompts)
    if args.dry_run:print(json.dumps(audit,indent=2,sort_keys=True));return 0 if audit["status"]=="PASS" else 2
    if audit["status"]!="PASS":raise SystemExit("Disagreement-gate preflight failed")
    if args.phase=="all":
        for phase in ("generate","reward","cost","analyze"):child(phase,args)
    elif args.phase=="generate":
        from PARM_TARO.adaptive_parm.disagreement_gate.generation import run
        print(json.dumps(run(select_cases(build_manifest(),args.num_prompts),resume=args.resume,min_free_mib=args.min_free_mib,max_new_tokens=args.max_new_tokens),indent=2))
    elif args.phase in ("reward","cost"):
        from PARM_TARO.adaptive_parm.disagreement_gate.scoring import run
        print(json.dumps(run(args.phase,resume=args.resume,min_free_mib=args.min_free_mib),indent=2))
    else:
        from PARM_TARO.adaptive_parm.disagreement_gate.analysis import run
        print(json.dumps(run(),indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())

