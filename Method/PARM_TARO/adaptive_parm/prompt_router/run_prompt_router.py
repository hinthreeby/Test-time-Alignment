"""Run prompt feature extraction and grouped-CV training."""

from __future__ import annotations

import argparse
import json

from .feature_extraction import ADAPTER, BASE, FEATURES, MANIFEST, OUTPUT, extract_features
from .train import CANDIDATES, ORACLE, train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("features", "train", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=6000)
    args = parser.parse_args()
    paths = {"base": BASE, "adapter": ADAPTER, "manifest": MANIFEST, "candidate_utility": CANDIDATES, "oracle": ORACLE}
    if args.dry_run:
        print(json.dumps({"status":"DRY_RUN_PASS" if all(path.exists() for path in paths.values()) else "DRY_RUN_FAIL","phase":args.phase,"device":args.device,"paths":{name:{"path":str(path),"exists":path.exists()} for name,path in paths.items()},"features_exist":FEATURES.is_file(),"model_loaded":False,"output":str(OUTPUT)},indent=2)); return 0
    if args.phase in {"features", "all"}:
        if args.device != "cuda": raise SystemExit("Prompt feature extraction requires the frozen 4-bit model on CUDA")
        print(json.dumps(extract_features(resume=args.resume,min_free_mib=args.min_free_mib),indent=2))
    if args.phase in {"train", "all"}:
        print(json.dumps(train_and_evaluate(),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
