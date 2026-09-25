"""Deterministic prompt selection with prompt-hash leakage exclusion."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Iterable
from .config import ALPHA_GRID, DENSE_OUT, NUM_PROMPTS, ROOT, VALIDATION
from .io import atomic_json, atomic_jsonl, read_jsonl

PRIOR_SOURCES = (
    ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl",
    ROOT / "results/parm_taro/adaptive_parm/04_token_headroom/manifest.jsonl",
    ROOT / "results/parm_taro/adaptive_parm/05_rich_state_probe/feature_manifest.jsonl",
    ROOT / "results/parm_taro/adaptive_parm/06_lookahead_probe/manifest.jsonl",
    ROOT / "results/parm_taro/adaptive_parm/07_disagreement_gate/manifest.jsonl",
)

def prompt_hash(prompt: str) -> str: return hashlib.sha256(prompt.encode()).hexdigest()
def _rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl": return read_jsonl(path)
    payload = json.loads(path.read_text()); return payload if isinstance(payload, list) else payload.get("rows", [])

def excluded_prompt_hashes() -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set(); provenance = []
    for path in PRIOR_SOURCES:
        if not path.is_file():
            provenance.append({"path": str(path), "present": False, "prompt_hashes": 0}); continue
        found = {str(row["prompt_sha256"]) if row.get("prompt_sha256") else prompt_hash(str(row["prompt"]))
                 for row in _rows(path) if row.get("prompt_sha256") or row.get("prompt") is not None}
        excluded.update(found); provenance.append({"path": str(path), "present": True, "prompt_hashes": len(found)})
    return excluded, provenance

def build_manifest(num_prompts: int = NUM_PROMPTS) -> list[dict[str, Any]]:
    if num_prompts != NUM_PROMPTS: raise ValueError("The frozen protocol requires exactly 20 prompts")
    validation = json.loads(VALIDATION.read_text()); excluded, provenance = excluded_prompt_hashes(); selected = []; seen = set()
    for index, row in enumerate(validation):
        digest = prompt_hash(str(row["prompt"]))
        if digest in excluded or digest in seen: continue
        seen.add(digest); selected.append((index, row, digest))
        if len(selected) == num_prompts: break
    if len(selected) != num_prompts: raise RuntimeError(f"Only {len(selected)} fresh validation prompts are available")
    records = [{"case_id": f"dense_p{pi:02d}_a{ai:02d}", "sample_id": str(row["sample_id"]),
                "validation_index": vi, "prompt": str(row["prompt"]), "prompt_sha256": digest,
                "requested_alpha": list(alpha), "alpha_order": ["helpfulness", "harmlessness"]}
               for pi, (vi, row, digest) in enumerate(selected) for ai, alpha in enumerate(ALPHA_GRID)]
    path = DENSE_OUT / "manifest.jsonl"
    if path.is_file() and read_jsonl(path) != records: raise RuntimeError("Existing frozen dense-alpha manifest differs")
    atomic_jsonl(path, records); atomic_json(DENSE_OUT / "manifest_summary.json", {
        "status": "PASS", "split": "validation", "test_set_used": False, "fresh_prompts": num_prompts,
        "cases": len(records), "prompt_hashes": [x[2] for x in selected], "alphas": [list(x) for x in ALPHA_GRID],
        "excluded_unique_prompt_hashes": len(excluded), "exclusion_sources": provenance})
    return records
