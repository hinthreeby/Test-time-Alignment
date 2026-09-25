#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import bootstrap_import_paths,subprocess_environment
bootstrap_import_paths(ROOT)
PY="/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10"; CONTROL=ROOT/"results/parm_taro/recovery/reproduced/pblora_epoch2_control"
SOURCE=ROOT/"results/parm_taro/recovery/reproduced/pblora/checkpoint-200"; EPOCH1_ADAPTER=ROOT/"results/parm_taro/recovery/reproduced/pblora/final_checkpoint/adapter_model.safetensors"
EPOCH1_DENSE=ROOT/"results/parm_taro/pareto_scalarization_probe/01_dense_alpha"; EPOCH2_DENSE_ROOT=CONTROL/"dense_alpha"
TRAIN_SCRIPT=ROOT/"PARM_TARO/epoch2_control/train_epoch2_control.sh"
EPOCH2_ADAPTER=CONTROL/"final_checkpoint/adapter_model.safetensors"
EXPECTED_EPOCH2_SHA256="73c91f162a970cc41c3b0b8ef83ede36ebd6fcda100a6712320482b4b657100b"
def sha(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(4*1024*1024),b""): d.update(block)
    return d.hexdigest()
def write(path:Path,value:dict): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def verify_epoch2_adapter():
    actual=sha(EPOCH2_ADAPTER) if EPOCH2_ADAPTER.is_file() else None
    if actual!=EXPECTED_EPOCH2_SHA256: raise RuntimeError(f"Epoch-2 adapter SHA mismatch: {actual}")
    return {"adapter_path":str(EPOCH2_ADAPTER.parent),"adapter_sha256":actual}
def record_generation_provenance():
    path=EPOCH2_DENSE_ROOT/"01_dense_alpha/generation_summary.json"
    if path.is_file():
        payload=json.loads(path.read_text()); payload.update(verify_epoch2_adapter()); write(path,payload)
    return path
def preflight()->dict:
    required=[SOURCE/x for x in ("adapter_model.safetensors","optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json")]
    checks={"checkpoint_200_complete":all(p.is_file() for p in required),"epoch1_adapter":EPOCH1_ADAPTER.is_file(),"epoch1_dense_complete":(EPOCH1_DENSE/"dense_alpha_summary.json").is_file(),"train_script":TRAIN_SCRIPT.is_file()}
    record={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"source_checkpoint":str(SOURCE),"target_total_epochs":2,
            "epoch1_adapter_sha256":sha(EPOCH1_ADAPTER) if EPOCH1_ADAPTER.is_file() else None,"epoch1_manifest_sha256":sha(EPOCH1_DENSE/"manifest.jsonl") if (EPOCH1_DENSE/"manifest.jsonl").is_file() else None}
    old=CONTROL/"control_preflight.json"
    if old.is_file() and json.loads(old.read_text()).get("epoch1_adapter_sha256") != record["epoch1_adapter_sha256"]: raise RuntimeError("Epoch-1 adapter changed since control preparation")
    write(old,record); return record
def audit()->dict:
    final=CONTROL/"final_checkpoint/adapter_model.safetensors"
    if not final.is_file(): raise RuntimeError("Epoch-2 final adapter missing")
    import torch
    from safetensors import safe_open
    finite=True; tensors=0
    with safe_open(final,framework="pt",device="cpu") as h:
        for key in h.keys(): finite=finite and bool(torch.isfinite(h.get_tensor(key)).all()); tensors+=1
    checkpoints=sorted([p for p in CONTROL.glob("checkpoint-*") if (p/"trainer_state.json").is_file()],key=lambda p:int(p.name.split("-")[-1]))
    state=json.loads((checkpoints[-1]/"trainer_state.json").read_text()); history=state["log_history"]
    training=json.loads((CONTROL/"recovery_preflight.json").read_text()); progress=json.loads((CONTROL/"recovery_training_audit.json").read_text()); resume=json.loads((CONTROL/"resume_audit.json").read_text()); before=json.loads((CONTROL/"control_preflight.json").read_text())
    result={"status":"PASS" if finite and training.get("sampled_base_state_unchanged") and progress["global_step"]==resume["target_total_steps"] and sha(EPOCH1_ADAPTER)==before["epoch1_adapter_sha256"] else "FAIL",
        "all_adapter_weights_finite":finite,"adapter_tensor_count":tensors,"adapter_sha256":sha(final),"epoch1_adapter_unchanged":sha(EPOCH1_ADAPTER)==before["epoch1_adapter_sha256"],
        "base_model_fingerprint_before":training.get("sampled_base_fingerprint_before"),"base_model_fingerprint_after":training.get("sampled_base_fingerprint_after"),
        "base_model_fingerprint_unchanged":training.get("sampled_base_state_unchanged"),"total_optimizer_steps":progress["global_step"],"target_total_epochs":2,
        "last_train_loss":progress["losses"][-1] if progress["losses"] else None,"last_full_checkpoint_step":state["global_step"],"last_eval_loss":next((x["eval_loss"] for x in reversed(history) if "eval_loss" in x),None)}
    write(CONTROL/"training_audit.json",result)
    if result["status"]!="PASS": raise RuntimeError(f"Epoch-2 audit failed: {result}")
    return result
def eval_phase(kind:str,resume:bool,min_free:int):
    import PARM_TARO.pareto_scalarization_probe.config as c
    c.OUTPUT_ROOT=EPOCH2_DENSE_ROOT; c.DENSE_OUT=EPOCH2_DENSE_ROOT/"01_dense_alpha"; c.ADAPTER=CONTROL/"final_checkpoint"
    import PARM_TARO.adapters.parm_adapter as adapter_runtime
    adapter_runtime.PARM_ROOT=ROOT/"Method/PARM"
    adapter_runtime.PARM_PEFT_SRC=adapter_runtime.PARM_ROOT/"peft/src"
    adapter_runtime.PARM_ARITHMETIC_SRC=adapter_runtime.PARM_ROOT/"language-model-arithmetic/src"
    import PARM_TARO.recovery.low_vram_suite.common as common
    common.ADAPTER=c.ADAPTER
    from PARM_TARO.pareto_scalarization_probe.probe_io import atomic_jsonl,read_jsonl
    cases=read_jsonl(EPOCH1_DENSE/"manifest.jsonl"); c.DENSE_OUT.mkdir(parents=True,exist_ok=True)
    atomic_jsonl(c.DENSE_OUT/"manifest.jsonl",cases)
    if kind=="generate":
        import PARM_TARO.pareto_scalarization_probe.generation as m
        from PARM_TARO.pareto_scalarization_probe.probe_io import atomic_json
        m.DENSE_OUT=c.DENSE_OUT; m.ADAPTER=c.ADAPTER; m.STATE=c.DENSE_OUT/"generation_state"
        result=m.run(cases,resume=resume,min_free_mib=min_free,max_new_tokens=64)
        adapter=verify_epoch2_adapter(); result.update(adapter); atomic_json(c.DENSE_OUT/"generation_summary.json",result); return result
    if kind in ("reward","cost"):
        import PARM_TARO.pareto_scalarization_probe.scoring as m; m.DENSE_OUT=c.DENSE_OUT; return m.run(kind,resume=resume,min_free_mib=min_free)
    import PARM_TARO.pareto_scalarization_probe.analysis as m; m.DENSE_OUT=c.DENSE_OUT; return m.run()
def child(phase:str,args): subprocess.run([PY,str(Path(__file__).resolve()),"--phase",phase,"--resume","--min-free-mib",str(args.min_free_mib)],cwd=ROOT,env=subprocess_environment(ROOT),check=True)
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=("all","train","audit","generate","reward","cost","analyze","compare"),default="all"); p.add_argument("--resume",action="store_true"); p.add_argument("--min-free-mib",type=int,default=6000); p.add_argument("--dry-run",action="store_true"); args=p.parse_args(); adapter_record=verify_epoch2_adapter(); record_generation_provenance(); pf=preflight(); pf.update({"epoch2_adapter":adapter_record})
    if args.dry_run: print(json.dumps(pf,indent=2)); return 0 if pf["status"]=="PASS" else 2
    if pf["status"]!="PASS": return 2
    if args.phase=="all":
        for phase in ("train","audit","generate","reward","cost","analyze","compare"): child(phase,args)
    elif args.phase=="train": subprocess.run([str(TRAIN_SCRIPT)],cwd=ROOT,check=True)
    elif args.phase=="audit": print(json.dumps(audit(),indent=2))
    elif args.phase in ("generate","reward","cost","analyze"): print(json.dumps(eval_phase(args.phase,args.resume,args.min_free_mib),indent=2))
    else:
        from PARM_TARO.epoch2_control.compare import run; print(json.dumps(run(),indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
