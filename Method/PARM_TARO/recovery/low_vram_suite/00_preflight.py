"""Zero-cost environment and artifact preflight; never loads Tulu."""
import argparse, importlib.metadata, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import OUT, artifact_preflight, atomic_json_dump, get_gpu_status, load_manifest, validate_manifest

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args()
    packages={}
    for name in ("torch","transformers","peft","trl","accelerate","bitsandbytes"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]="MISSING"
    import torch
    manifest=validate_manifest(load_manifest()); artifacts=artifact_preflight(); gpu=get_gpu_status()
    checks={"python_exists":Path(sys.executable).is_file(),"torch":packages["torch"]!="MISSING","cuda_available":bool(torch.cuda.is_available()),"gpu_visible":gpu.get("gpu_name") is not None,"required_source_artifacts":all(artifacts["checks"][k] for k in ("base","adapter","manifest","validation","base_config","adapter_config","adapter_weights")),"manifest":manifest["status"]=="PASS","output_writable":os.access(OUT.parent,os.W_OK)}
    payload={"status":"PASS" if all(checks.values()) else "FAIL","dry_run":args.dry_run,"python":sys.executable,"python_version":sys.version,"packages":packages,"torch_cuda_runtime":torch.version.cuda,"gpu":gpu,"checks":checks,"manifest":manifest,"artifacts":artifacts}
    atomic_json_dump(OUT/"00_preflight.json",payload)
    for key,value in checks.items(): print(f"{key:32} {'PASS' if value else 'FAIL'}")
    print(f"PREFLIGHT {payload['status']}"); return 0 if payload["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())

