"""Sequential, response-cached official Beaver scoring."""
from __future__ import annotations
import gc, json, math
import torch
from PARM_TARO.adaptive_parm.token_headroom.scoring import _load
from PARM_TARO.recovery.low_vram_suite.common import get_gpu_status, require_free_vram, resolve_safe_rlhf_source
from .config import COST, DENSE_OUT, REWARD
from .io import atomic_csv, atomic_json, read_jsonl

def run(kind: str, *, resume: bool, min_free_mib: int) -> dict:
    if kind not in ("reward", "cost"): raise ValueError(kind)
    resolution = resolve_safe_rlhf_source()
    if not resolution.get("ready"): raise RuntimeError(f"Safe-RLHF runtime unavailable: {resolution}")
    rows = read_jsonl(DENSE_OUT / "generations.jsonl")
    unique = {r["response_key"]: {"prompt": r["prompt"], "response": r["response"]} for r in rows}
    state = DENSE_OUT / "score_state" / kind; state.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in state.glob("*.json")}; missing = [k for k in sorted(unique) if k not in existing]
    if existing and not resume: raise RuntimeError(f"{kind} progress exists; pass --resume")
    model = tokenizer = None; memory = {"before": get_gpu_status()}
    if missing:
        require_free_vram(min_free_mib); model, tokenizer = _load(REWARD if kind == "reward" else COST, resolution); memory["loaded"] = get_gpu_status()
        try:
            with torch.inference_mode():
                for index, key in enumerate(missing, 1):
                    item = unique[key]; text = f"BEGINNING OF CONVERSATION: USER: {item['prompt']} ASSISTANT:{item['response']}"
                    encoded = {n: v.cuda() for n, v in tokenizer(text, return_tensors="pt", truncation=True, max_length=512).items()}
                    output = model(**encoded); end = output["end_scores"] if isinstance(output, dict) else output.end_scores
                    value = float(end.detach().float().reshape(-1)[0].cpu())
                    if not math.isfinite(value): raise FloatingPointError(f"Non-finite {kind} score")
                    atomic_json(state / f"{key}.json", {"response_key": key, "score": value}); print(f"[{index}/{len(missing)}] {kind} {key[:12]}", flush=True)
        finally:
            if model is not None: del model, tokenizer
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); memory["after_release"] = get_gpu_status()
    scores = {key: float(json.loads((state / f"{key}.json").read_text())["score"]) for key in unique}
    output = [{"case_id": row["case_id"], "response_key": row["response_key"], f"{kind}_score": scores[row["response_key"]]} for row in rows]
    atomic_csv(DENSE_OUT / f"{kind}_scores.csv", output)
    summary = {"status": "PASS", "kind": kind, "records": len(rows), "unique_responses": len(unique), "new_scores": len(missing),
               "model_path": str(REWARD if kind == "reward" else COST), "safe_rlhf_source": resolution.get("path"), "memory": memory}
    atomic_json(DENSE_OUT / f"{kind}_scoring_summary.json", summary); return summary
