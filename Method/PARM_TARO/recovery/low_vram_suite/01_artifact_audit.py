"""CPU-only PBLoRA and feasibility-manifest integrity audit."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import ADAPTER, OUT, ROOT, atomic_json_dump, atomic_text, load_manifest, sha256_file, validate_manifest

def tree_hash(root):
    digest=hashlib.sha256(); files=[]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel=path.relative_to(root).as_posix(); sha=sha256_file(path); size=path.stat().st_size
        files.append({"path":rel,"bytes":size,"sha256":sha}); digest.update(rel.encode()+b"\0"+sha.encode()+b"\n")
    return files,digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--full-reload",action="store_true"); args=parser.parse_args()
    if args.full_reload and args.dry_run: parser.error("--full-reload is incompatible with --dry-run")
    import torch
    from safetensors import safe_open
    config_path=ADAPTER/"adapter_config.json"; weights=ADAPTER/"adapter_model.safetensors"; training=ADAPTER.parent
    config=json.loads(config_path.read_text()) if config_path.is_file() else {}
    tensor_count=parameter_count=0; finite=True
    if weights.is_file():
        with safe_open(weights,framework="pt",device="cpu") as handle:
            for key in handle.keys():
                tensor=handle.get_tensor(key); tensor_count+=1; parameter_count+=tensor.numel(); finite=finite and bool(torch.isfinite(tensor).all())
    audit_file=training/"recovery_training_audit.json"; training_audit=json.loads(audit_file.read_text()) if audit_file.is_file() else {}
    trainer_states=[]
    for state_path in training.glob("checkpoint-*/trainer_state.json"):
        state=json.loads(state_path.read_text()); trainer_states.append(state)
    expected_steps=max((int(state.get("max_steps",0)) for state in trainer_states),default=0)
    global_step=int(training_audit.get("global_step",-1)); reconstructed_epoch=(global_step/expected_steps if expected_steps else -1.0)
    files,tree=tree_hash(ADAPTER) if ADAPTER.is_dir() else ([],None)
    manifest=validate_manifest(load_manifest())
    checks={"checkpoint_exists":ADAPTER.is_dir(),"no_zero_files":bool(files) and not any(row["bytes"]==0 for row in files),"tensors_finite":finite and tensor_count==384,"parameter_count":parameter_count==6_296_064,"config":str(config.get("peft_type","")).upper()=="PBLORA" and config.get("obj_num")==2 and config.get("r1")==4 and config.get("r2")==4 and float(config.get("lora_alpha",-1))==8 and float(config.get("lora_dropout",-1))==.05 and set(config.get("target_modules",[]))=={"q_proj","k_proj","v_proj"},"step_240":global_step==240,"epoch_1":abs(reconstructed_epoch-1.0)<1e-12,"manifest":manifest["status"]=="PASS"}
    reload_status="NOT_REQUESTED"
    if args.full_reload:
        from PARM_TARO.recovery.low_vram_suite.common import load_tulu_pblora_4bit, release_model, require_free_vram
        require_free_vram(); model,base,tokenizer=load_tulu_pblora_4bit(); reload_status="PASS"; del model,base,tokenizer; release_model()
    payload={"status":"PASS" if all(checks.values()) else "FAIL","dry_run":args.dry_run,"checks":checks,"final_checkpoint":str(ADAPTER),"tree_sha256":tree,"files":files,"tensor_count":tensor_count,"trainable_parameter_count":parameter_count,"config":config,"global_step":global_step,"expected_steps":expected_steps,"epoch":reconstructed_epoch,"epoch_evidence":"global_step / trainer_state.max_steps","manifest":manifest,"full_reload":reload_status}
    atomic_json_dump(OUT/"01_artifact_audit.json",payload)
    atomic_text(OUT/"01_artifact_audit.md",f"# Artifact audit\n\nVerdict: **ARTIFACT AUDIT {payload['status']}**\n\n- PBLoRA tensors: {tensor_count}; parameters: {parameter_count:,}; finite: {finite}\n- Step/epoch: {payload['global_step']} / {payload['epoch']}\n- Tree SHA256: `{tree}`\n- Manifest: {manifest['num_cases']} cases, {manifest['num_unique_prompts']} unique prompts\n- Full 7B reload: {reload_status}\n")
    print(f"ARTIFACT AUDIT {payload['status']}"); return 0 if payload["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
