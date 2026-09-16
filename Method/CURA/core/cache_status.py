from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Method.CURA.core.cache_io import read_manifest, verify_cache
from Method.CURA.core.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = Path(args.cache_dir)
    if not path.is_absolute(): path = PROJECT_ROOT / path
    manifest = read_manifest(path)
    if manifest is None: raise FileNotFoundError(f"Missing manifest in {path}")
    output = {
        "cache_dir": str(path), "status": manifest["status"],
        "completed_prompts": manifest["completed_prompts"], "total_prompts": manifest["total_prompts"],
        "completed_shards": len(manifest["completed_shards"]), "failed_prompts": len(manifest["failed_prompt_ids"]),
    }
    if args.verify: output["verification"] = verify_cache(path, strict=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
