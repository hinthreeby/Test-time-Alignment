"""Narrow PyTorch >=2.6 compatibility for trusted local Trainer state."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
from typing import Any

TRUSTED_STATE_NAMES=frozenset({"rng_state.pth","optimizer.pt","scheduler.pt","scaler.pt"})
REQUIRED_STATE_NAMES=("rng_state.pth","optimizer.pt","scheduler.pt","trainer_state.json")

def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()

class TrustedTrainerStateLoadShim:
    """Override only omitted ``weights_only`` for an exact trusted directory."""
    def __init__(self,checkpoint:str|Path,torch_module:Any):
        self.checkpoint=Path(checkpoint).resolve();self.torch=torch_module;self.original=torch_module.load;self.events=[];self.installed=False
    def eligible(self,value:Any)->bool:
        if not isinstance(value,(str,os.PathLike)):return False
        path=Path(value).resolve()
        return path.parent==self.checkpoint and path.name in TRUSTED_STATE_NAMES
    def load(self,*args:Any,**kwargs:Any)->Any:
        if args and self.eligible(args[0]) and "weights_only" not in kwargs:
            kwargs["weights_only"]=False;name=Path(args[0]).name;self.events.append(name)
            print(f"[resume-compat] trusted torch.load weights_only=False: {name}",flush=True)
        return self.original(*args,**kwargs)
    def install(self)->None:
        if not self.installed:self.torch.load=self.load;self.installed=True
    def uninstall(self)->None:
        if self.installed:self.torch.load=self.original;self.installed=False

def atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(temporary,path)

def preflight(checkpoint:str|Path,output_dir:str|Path,torch_module:Any)->dict[str,Any]:
    checkpoint=Path(checkpoint).resolve();output_dir=Path(output_dir).resolve()
    missing=[name for name in REQUIRED_STATE_NAMES if not (checkpoint/name).is_file()]
    scaler=checkpoint/"scaler.pt";state_path=checkpoint/"trainer_state.json"
    if missing:raise RuntimeError(f"Resume checkpoint missing required state: {missing}")
    state=json.loads(state_path.read_text(encoding="utf-8"));restored=int(state["global_step"]);epoch=float(state["epoch"])
    expected_from_name=int(checkpoint.name.rsplit("-",1)[-1])
    if restored!=expected_from_name:raise RuntimeError(f"Checkpoint name/state step mismatch: {expected_from_name} != {restored}")
    paths=[checkpoint/name for name in ("rng_state.pth","optimizer.pt","scheduler.pt")]+([scaler] if scaler.is_file() else [])
    before={path.name:sha256(path) for path in paths};shim=TrustedTrainerStateLoadShim(checkpoint,torch_module);shim.install();loaded={}
    try:
        for path in paths:
            value=torch_module.load(path,map_location="cpu");loaded[path.name]=type(value).__name__;del value
    finally:shim.uninstall()
    after={path.name:sha256(path) for path in paths}
    result={"status":"PASS" if before==after and set(shim.events)==set(before) else "FAIL","trusted_local_checkpoint":True,
        "resume_checkpoint":str(checkpoint),"global_step":restored,"expected_step_from_checkpoint_name":expected_from_name,"epoch":epoch,"required_files":list(REQUIRED_STATE_NAMES),
        "optional_scaler_present":scaler.is_file(),"test_load_types":loaded,"sha256_before":before,"sha256_after":after,
        "bytes_unchanged":before==after,"compatibility_load_events":shim.events}
    atomic_json(output_dir/"resume_compat_preflight.json",result)
    if result["status"]!="PASS":raise RuntimeError(f"Resume compatibility preflight failed: {result}")
    return result

def install_training_audit(recovery_module:Any,checkpoint:Path,output_dir:Path,shim:TrustedTrainerStateLoadShim)->None:
    original_install=recovery_module.install_trainer_hooks
    def install(output:Path,resume_checkpoint:str|None)->None:
        original_install(output,resume_checkpoint)
        from transformers import Trainer
        original_train=Trainer.train
        def audited_train(self:Any,*args:Any,**kwargs:Any)->Any:
            state=json.loads((checkpoint/"trainer_state.json").read_text(encoding="utf-8"));restored=int(state["global_step"])
            batches=len(self.get_train_dataloader());updates=max(batches//self.args.gradient_accumulation_steps,1)
            target=int(self.args.max_steps) if self.args.max_steps>0 else math.ceil(float(self.args.num_train_epochs)*updates)
            record={"resume_checkpoint":str(checkpoint),"restored_global_step":restored,"checkpoint_epoch":float(state["epoch"]),
                "train_dataloader_batches":batches,"gradient_accumulation_steps":int(self.args.gradient_accumulation_steps),
                "updates_per_epoch":updates,"target_total_steps":target,"remaining_steps":target-restored,
                "optimizer_state_restored":False,"scheduler_state_restored":False,"rng_state_restored":False}
            if target<=restored:raise RuntimeError(f"Resume target {target} does not exceed restored step {restored}")
            atomic_json(output_dir/"resume_audit.json",record)
            print("[resume-audit] "+json.dumps(record,sort_keys=True),flush=True)
            result=original_train(self,*args,**kwargs)
            events=set(shim.events);record.update({"optimizer_state_restored":"optimizer.pt" in events,
                "scheduler_state_restored":"scheduler.pt" in events,"rng_state_restored":"rng_state.pth" in events,
                "compatibility_load_events":shim.events,"completed_global_step":int(self.state.global_step)})
            record["status"]="PASS" if all(record[k] for k in ("optimizer_state_restored","scheduler_state_restored","rng_state_restored")) else "FAIL"
            atomic_json(output_dir/"resume_audit.json",record)
            if record["status"]!="PASS":raise RuntimeError(f"Trainer state restoration audit failed: {record}")
            return result
        Trainer.train=audited_train
    recovery_module.install_trainer_hooks=install
