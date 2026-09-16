"""Prepare (but never execute) the validation-only Feasibility-60 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
OUT = ROOT / "results/parm_taro/recovery/feasibility60"
VALIDATION = ROOT / "dataset/parm_taro/validation.json"
ALPHAS = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    prompts = validation[:12]
    manifest = []
    for alpha_index, alpha in enumerate(ALPHAS):
        for prompt_index, row in enumerate(prompts):
            manifest.append({
                "case_id": f"f60_a{alpha_index}_p{prompt_index:02d}",
                "sample_id": row["sample_id"], "validation_index": prompt_index,
                "requested_alpha": list(alpha), "alpha_order": ["helpfulness", "harmlessness"],
                "prompt": row["prompt"], "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            })
    config = {
        "schema_version": 1, "status": "PREPARED_NOT_EXECUTED", "split": "validation",
        "manifest": "results/parm_taro/recovery/feasibility60/manifest.jsonl",
        "manifest_cases": 60, "unique_prompts": 12, "cases_per_alpha": 12,
        "validation_path": "dataset/parm_taro/validation.json", "validation_sha256": sha(VALIDATION),
        "base_model": "models/tulu-2-7b",
        "pblora_adapter": "results/parm_taro/recovery/reproduced/pblora/final_checkpoint",
        "alpha_grid": [list(value) for value in ALPHAS], "alpha_order": ["helpfulness", "harmlessness"],
        "formulations": {
            "current": "a + lambda*b", "normalized": "(a + lambda*b)/(1+lambda)",
            "author": "(a+b)/2", "paper_beta_1": "a+b",
        },
        "normalized_lambda_grid": [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "current_controls": [0.010731, 1.0],
        "gradient_lambda_grid": [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0],
        "root_causes_prepared": ["RC3", "RC5", "RC6", "RC8", "RC9", "RC13", "RC14", "RC15"],
        "execution_gate": {"minimum_free_vram_mib": 14336, "requires_user_explicit_launch": True, "reward_cost_not_co_loaded_with_alpha_probe": True},
        "random_seed": 2026, "generation_executed": False, "test_split_used": False,
    }
    atomic_text(OUT / "manifest.jsonl", "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest))
    atomic_text(OUT / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    readme = """# Feasibility-60 (prepared, not executed)

This is a deterministic 60-case **validation-only** manifest: the same 12
prefixes are crossed with five alpha values, enabling paired comparisons. It
does not reference the Stage-10 1,500-case test split.

Prepared diagnostics cover RC3 guide quality, RC5 fusion mismatch, RC6 scale
confounding, RC8 alpha sensitivity, RC9 NLL gradient direction, RC13 adaptive
headroom, RC14 router failure despite headroom, and RC15 alpha-specific optima.

Do not launch the heavy generation/scoring job until the PBLoRA alpha probe
passes, free GPU memory is at least 14,336 MiB, and the user explicitly starts
it. Reward and cost evaluators must not be co-loaded with the low-VRAM probe.

The existing resumable entrypoint remains:

```bash
FEASIBILITY_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \\
CUDA_VISIBLE_DEVICES=0 bash PARM_TARO/recovery/scripts/run_feasibility_gpu.sh \\
  --phase all --resume
```

The command above is documentation only and was not executed during epoch-1
validation preparation. The feasibility runner still requires integration of
this recovered adapter/config before scientific execution.
"""
    atomic_text(OUT / "README.md", readme)
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "cases": len(manifest), "validation_sha256": config["validation_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
