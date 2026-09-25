#!/usr/bin/env python3
"""Extract frozen internal rollout scores, then run CPU lookahead analysis."""

from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

def root()->Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate/".git").exists() and (candidate/"PARM_TARO").exists():sys.path.insert(0,str(candidate));return candidate
    raise RuntimeError("Cannot locate project root")

ROOT=root()
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import bootstrap_import_paths,subprocess_environment
bootstrap_import_paths(ROOT)
from PARM_TARO.adaptive_parm.lookahead_probe.config import ADAPTER,BASE,HORIZONS,OUT,SOURCE
from PARM_TARO.adaptive_parm.lookahead_probe.scoring import build_jobs,load_source_records
from PARM_TARO.adaptive_parm.token_headroom.io import atomic_json

def preflight()->dict:
    checks={"rollouts":SOURCE/"counterfactual_rollouts.jsonl","states":SOURCE/"state_features.csv","utility":SOURCE/"state_utility.csv","leakage_audit":SOURCE/"leakage_audit.json","rich_policy":ROOT/"results/parm_taro/adaptive_parm/05_rich_state_probe/best_policy_predictions.csv","base_config":BASE/"config.json","adapter_config":ADAPTER/"adapter_config.json","adapter_weights":ADAPTER/"adapter_model.safetensors"}
    paths={name:{"path":str(path),"ready":path.is_file() and path.stat().st_size>0} for name,path in checks.items()}
    try:
        rollouts,states,utility=load_source_records();jobs,mapping=build_jobs(rollouts,states);leakage=json.loads((SOURCE/"leakage_audit.json").read_text(encoding="utf-8"));valid=len(rollouts)==1250 and len(states)==250 and len(utility)==1250 and set(mapping)==set(utility) and leakage.get("status")=="PASS"
        details={"rollouts":len(rollouts),"states":len(states),"utility_rows":len(utility),"unique_branches":len(jobs),"actionable_states":len({row['state_id'] for row in rollouts if not row['all_weights_same_next_token']}),"horizons":list(HORIZONS),"source_leakage_audit":leakage.get("status"),"no_generation":True,"no_beaver":True}
    except Exception as error:valid=False;details={"error":f"{type(error).__name__}: {error}"}
    result={"status":"PASS" if all(row["ready"] for row in paths.values()) and valid else "FAIL","loads_model":False,"paths":paths,**details};OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/"preflight.json",result);return result

def launch(phase:str,args:argparse.Namespace)->None:
    command=[sys.executable,str(Path(__file__).resolve()),"--phase",phase,"--min-free-mib",str(args.min_free_mib)]
    if args.resume:command.append("--resume")
    subprocess.run(command,cwd=ROOT,env=subprocess_environment(ROOT),check=True)

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("all","extract","analyze"),default="all");parser.add_argument("--resume",action="store_true");parser.add_argument("--dry-run",action="store_true");parser.add_argument("--min-free-mib",type=int,default=int(os.environ.get("MIN_FREE_MIB","6000")));args=parser.parse_args();audit=preflight()
    if args.dry_run:print(json.dumps(audit,indent=2,sort_keys=True));return 0 if audit["status"]=="PASS" else 2
    if audit["status"]!="PASS":raise SystemExit("Lookahead preflight failed")
    if args.phase=="all":launch("extract",args);launch("analyze",args)
    elif args.phase=="extract":
        from PARM_TARO.adaptive_parm.lookahead_probe.scoring import extract
        print(json.dumps(extract(resume=args.resume,min_free_mib=args.min_free_mib),indent=2,sort_keys=True))
    else:
        from PARM_TARO.adaptive_parm.lookahead_probe.analysis import analyze
        print(json.dumps(analyze(),indent=2,sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
