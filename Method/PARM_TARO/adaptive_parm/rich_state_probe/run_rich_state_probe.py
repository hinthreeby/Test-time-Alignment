#!/usr/bin/env python3
"""Resume-safe frozen rich-state extraction and grouped CPU analysis."""

from __future__ import annotations

import argparse,json,os,subprocess,sys
from pathlib import Path


def _root()->Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate/".git").exists() and (candidate/"PARM_TARO").exists():sys.path.insert(0,str(candidate));return candidate
    raise RuntimeError("Cannot locate project root")


ROOT=_root()
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import bootstrap_import_paths,subprocess_environment
bootstrap_import_paths(ROOT)
from PARM_TARO.adaptive_parm.rich_state_probe.config import ADAPTER,OUT,SOURCE
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_json


def preflight()->dict:
    import csv
    from PARM_TARO.adaptive_parm.rich_state_probe.features import validate_prefix_record
    required={"state_features":SOURCE/"state_features.csv","state_utility":SOURCE/"state_utility.csv","advantage_dataset":SOURCE/"tie_aware/advantage_dataset.csv","adapter":ADAPTER/"adapter_model.safetensors","base_model":ROOT/"models/tulu-2-7b"}
    checks={name:{"path":str(path),"exists":path.exists(),"readable":path.exists() and os.access(path,os.R_OK)} for name,path in required.items()}
    with (SOURCE/"state_features.csv").open(newline="",encoding="utf-8") as handle:states=list(csv.DictReader(handle))
    with (SOURCE/"state_utility.csv").open(newline="",encoding="utf-8") as handle:utility=list(csv.DictReader(handle))
    causal_prefixes=all(bool(validate_prefix_record(row)) for row in states)
    status="PASS" if all(row["exists"] and row["readable"] for row in checks.values()) and len(states)==250 and len(utility)==1250 and causal_prefixes else "FAIL"
    result={"status":status,"loads_model":False,"states":len(states),"utility_rows":len(utility),"unique_prompts":len({row['sample_id'] for row in states}),"causal_prefix_schema":causal_prefixes,"checks":checks,"phases":["extract frozen features on GPU","terminate extractor process","grouped CPU analysis"]};OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/"preflight.json",result);return result


def child(phase:str,args:argparse.Namespace)->None:
    command=[sys.executable,str(Path(__file__).resolve()),"--phase",phase,"--min-free-mib",str(args.min_free_mib)]
    if args.resume:command.append("--resume")
    print("Launching isolated phase:"," ".join(command),flush=True);subprocess.run(command,cwd=ROOT,env=subprocess_environment(ROOT),check=True)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--phase",choices=("all","extract","analyze"),default="all");parser.add_argument("--resume",action="store_true");parser.add_argument("--dry-run",action="store_true");parser.add_argument("--min-free-mib",type=int,default=int(os.environ.get("MIN_FREE_MIB","6000")));args=parser.parse_args();audit=preflight()
    if args.dry_run:print(json.dumps(audit,indent=2,sort_keys=True));return 0 if audit["status"]=="PASS" else 2
    if audit["status"]!="PASS":raise SystemExit("Rich-state preflight failed")
    if args.phase=="all":child("extract",args);child("analyze",args)
    elif args.phase=="extract":
        from PARM_TARO.adaptive_parm.rich_state_probe.features import extract
        print(json.dumps(extract(resume=args.resume,min_free_mib=args.min_free_mib),indent=2,sort_keys=True))
    else:
        from PARM_TARO.adaptive_parm.rich_state_probe.analysis import analyze
        print(json.dumps(analyze(),indent=2,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
