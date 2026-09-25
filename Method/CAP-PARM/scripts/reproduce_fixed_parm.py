#!/usr/bin/env python3
"""
CAP-PARM Phase 1: Fixed PARM inference.
Outputs results/cap_parm.json (same schema as other methods).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
CAP_PARM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP_PARM_DIR))

_PEFT_SRC = ROOT_DIR / "Method" / "PARM" / "peft" / "src"
if str(_PEFT_SRC) not in sys.path:
    sys.path.insert(0, str(_PEFT_SRC))
for _k in list(sys.modules.keys()):
    if _k == "peft" or _k.startswith("peft."):
        del sys.modules[_k]

import torch

from src.models.base_adapter import BaseModelAdapter
from src.models.parm_adapter import PARMAdapter
from src.decoding.fusion import fuse_scores

DEFAULT_BASE = str(ROOT_DIR / "models" / "tulu-2-7b")
DEFAULT_PARM = str(ROOT_DIR / "Method" / "PARM_TARO" / "results" / "parm_taro"
                   / "recovery" / "reproduced" / "pblora" / "final_checkpoint")
DEFAULT_DATASET = str(ROOT_DIR / "dataset" / "rad_benchmark" / "all.jsonl")
DEFAULT_OUTPUT  = str(ROOT_DIR / "results" / "cap_parm.json")


def parse_args():
    p = argparse.ArgumentParser(description="CAP-PARM fixed-w inference → results/cap_parm.json")
    p.add_argument("--base-model",   default=DEFAULT_BASE)
    p.add_argument("--parm-adapter", default=DEFAULT_PARM)
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--output-path",  default=DEFAULT_OUTPUT)
    p.add_argument("--num-prompts",  type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--w",            type=float, default=0.5,
                   help="Fixed PARM weight w_t ∈ [0,1]")
    p.add_argument("--alpha-help",   type=float, default=0.5)
    p.add_argument("--alpha-harm",   type=float, default=0.5)
    p.add_argument("--fusion-mode",  choices=["convex", "parm_product"], default="convex")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()
    assert abs(args.alpha_help + args.alpha_harm - 1.0) < 1e-5, "alpha must sum to 1"
    return args


def load_samples(path: str, limit: int) -> list[dict]:
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        rows = json.loads(content)
        rows = rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in content.splitlines() if l.strip()]
    return [r for r in rows if isinstance(r, dict)][:limit]


def save_results(records: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


def text_value(v) -> str:
    return str(v.get("text", "")) if isinstance(v, dict) else str(v or "")


def format_prompt(raw: str) -> str:
    return f"BEGINNING OF CONVERSATION: USER: {raw} ASSISTANT:"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    samples = load_samples(args.dataset_path, args.num_prompts)
    # Resume: load existing output
    out_path = Path(args.output_path)
    existing = []
    if out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    done_ids = {r.get("id") for r in existing if r.get("status") == "success"}

    pending = [(i, s) for i, s in enumerate(samples) if i not in done_ids]
    print(f"CAP-PARM  w={args.w}  alpha=[{args.alpha_help},{args.alpha_harm}]  fusion={args.fusion_mode}")
    print(f"Dataset  : {args.dataset_path}")
    print(f"Output   : {args.output_path}")
    print(f"Target   : {len(samples)} | Done: {len(done_ids)} | Remaining: {len(pending)}")
    if not pending:
        print("All done.")
        return

    alpha_user = [args.alpha_help, args.alpha_harm]

    print("\n[1/2] Loading Base model (this may take ~10s)...")
    base = BaseModelAdapter(args.base_model)
    print("[2/2] Loading PARM adapter (this may take ~10s)...")
    parm = PARMAdapter(args.base_model, args.parm_adapter)
    print("Models loaded. Starting generation...\n")

    by_id = {r["id"]: r for r in existing}
    eos_id = base.tokenizer.eos_token_id

    for idx, sample in tqdm(pending, desc="CAP-PARM", unit="prompt", ncols=80):
        prompt = text_value(sample.get("prompt")).strip()
        if not prompt:
            continue
        t0 = time.perf_counter()
        try:
            prompt_text = format_prompt(prompt)
            input_ids = base.tokenizer.encode(prompt_text, return_tensors="pt").to(base.device)
            base_state = base.prefill(input_ids)
            parm_state = parm.prefill(input_ids, alpha_user) if args.w > 0 else None

            output_tokens = []
            for _ in range(args.max_new_tokens):
                bs = base_state.next_scores
                if args.w > 0 and parm_state is not None:
                    ps = parm_state.next_scores
                    fused = fuse_scores(bs, ps, args.w, mode=args.fusion_mode)
                else:
                    fused = bs
                next_tok = torch.argmax(fused, dim=-1)
                tok = next_tok.item()
                output_tokens.append(tok)
                if tok == eos_id:
                    break
                base_state = base.step(next_tok, base_state.cache)
                if args.w > 0 and parm_state is not None:
                    parm_state = parm.step(next_tok, alpha_user, parm_state.cache)

            response = base.tokenizer.decode(output_tokens, skip_special_tokens=True)
            if not response.strip():
                raise RuntimeError("Empty response")
            status, error = "success", None
        except Exception as exc:
            response, status, error = "", "failed", f"{type(exc).__name__}: {exc}"

        by_id[idx] = {
            "id": sample.get("id", idx),
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": text_value(sample.get("continuation")),
            "response": response,
            "method": "CAP-PARM",
            "fusion_mode": args.fusion_mode,
            "w": args.w,
            "alpha_helpfulness": args.alpha_help,
            "alpha_harmlessness": args.alpha_harm,
            "base_model": args.base_model,
            "parm_adapter": args.parm_adapter,
            "latency": round(time.perf_counter() - t0, 4),
            "status": status,
            "error": error,
        }
        save_results(list(by_id.values()), args.output_path)

    success = sum(1 for r in by_id.values() if r.get("status") == "success")
    print(f"\nDone: {success}/{len(samples)} | Saved → {args.output_path}")


if __name__ == "__main__":
    main()
