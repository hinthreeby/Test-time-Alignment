#!/usr/bin/env python3
"""
Oracle A: Per-objective prefix signal generation.
Generates using base model, probes one-hot PARM log-probs at every step.
Outputs a JSONL file suitable for training the ObjectiveTracker.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
CAP_DIR  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP_DIR))

_PEFT_SRC = ROOT_DIR / "Method" / "PARM" / "peft" / "src"
if str(_PEFT_SRC) not in sys.path:
    sys.path.insert(0, str(_PEFT_SRC))
for _k in list(sys.modules.keys()):
    if _k == "peft" or _k.startswith("peft."):
        del sys.modules[_k]

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.base_adapter import BaseModelAdapter
from src.models.parm_adapter import PARMAdapter

DEFAULT_BASE    = str(ROOT_DIR / "models" / "tulu-2-7b")
DEFAULT_PARM    = str(ROOT_DIR / "Method" / "PARM_TARO" / "results" / "parm_taro"
                      / "recovery" / "reproduced" / "pblora" / "final_checkpoint")
DEFAULT_DATASET = str(ROOT_DIR / "dataset" / "rad_benchmark" / "all.jsonl")
DEFAULT_OUT     = str(ROOT_DIR / "results" / "oracle_probe_train_v2.jsonl")
SCHEMA_VERSION  = 2


def parse_args():
    p = argparse.ArgumentParser(description="Build oracle probe dataset for DynaCAP-PARM.")
    p.add_argument("--base-model",    default=DEFAULT_BASE)
    p.add_argument("--parm-adapter",  default=DEFAULT_PARM)
    p.add_argument("--dataset-path",  default=DEFAULT_DATASET)
    p.add_argument("--num-prompts",   type=int, default=200)
    p.add_argument("--max-new-tokens",type=int, default=64)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out",           default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output instead of resuming it.")
    return p.parse_args()


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


def text_value(v) -> str:
    return str(v.get("text", "")) if isinstance(v, dict) else str(v or "")


def format_prompt(raw: str) -> str:
    return f"BEGINNING OF CONVERSATION: USER: {raw} ASSISTANT:"


def probe_one(prompt: str, base, parm, max_new_tokens: int, device: str) -> dict:
    obj_vecs        = [[1.0, 0.0], [0.0, 1.0]]
    eos_id          = base.tokenizer.eos_token_id

    input_ids  = base.tokenizer.encode(format_prompt(prompt), return_tensors="pt").to(base.device)
    base_state = base.prefill(input_ids)

    output_tokens = []
    trace         = []

    for step in range(max_new_tokens):
        bs          = base_state.next_scores
        next_tok    = torch.argmax(bs, dim=-1)
        tok         = next_tok.item()

        # PBLoRA KV caches depend on alpha. Replay the full current prefix for
        # each one-hot preference instead of reusing a neutral-alpha cache.
        probed      = parm.probe_prefix(input_ids, obj_vecs)
        base_lp     = F.log_softmax(bs, dim=-1)
        r_k = [(F.log_softmax(ps, dim=-1)[0, tok] - base_lp[0, tok]).item() for ps in probed]
        top_k_lp, _ = base_lp.topk(32, dim=-1)
        probabilities = base_lp.exp()
        entropy = -(probabilities * base_lp).sum(dim=-1)
        margin = top_k_lp[:, 0] - top_k_lp[:, 1]
        position = step / max(max_new_tokens - 1, 1)
        tracker_features = [
            *[float(value) for value in top_k_lp[0].tolist()],
            float(entropy[0]),
            float(margin[0]),
            float(position),
            0.5,
            0.5,
        ]

        trace.append({
            "step":       step,
            "token_id":   tok,
            "token_str":  base.tokenizer.decode([tok]),
            "base_logprob": base_lp[0, tok].item(),
            "r_help":     r_k[0],
            "r_harm":     r_k[1],
            "tracker_features": tracker_features,
            "reward_targets": r_k,
        })

        output_tokens.append(tok)
        if tok == eos_id:
            break

        base_state = base.step(next_tok, base_state.cache)
        input_ids = torch.cat([input_ids, next_tok.reshape(1, 1)], dim=1)

    response = base.tokenizer.decode(output_tokens, skip_special_tokens=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_cache_policy": "full_prefix_replay_per_alpha",
        "prompt": prompt,
        "response": response,
        "trace": trace,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    samples = load_samples(args.dataset_path, args.num_prompts)

    # Resume: skip already processed prompts
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        existing_rows = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_rows.append(json.loads(line))
            except Exception:
                pass
        incompatible = [
            row for row in existing_rows
            if row.get("schema_version") != SCHEMA_VERSION
            or row.get("probe_cache_policy") != "full_prefix_replay_per_alpha"
        ]
        if incompatible and not args.overwrite:
            raise RuntimeError(
                f"{out_path} contains legacy objective probes with alpha-unsafe cache reuse. "
                "Choose a new --out path or pass --overwrite; do not mix schemas."
            )
        if args.overwrite:
            out_path.write_text("", encoding="utf-8")
        else:
            done = {str(row["prompt"]) for row in existing_rows if row.get("prompt")}

    pending = [r for r in samples if text_value(r.get("prompt")).strip() not in done]

    print(f"Oracle probe  dataset={args.dataset_path}")
    print(f"Output        {args.out}")
    print(f"Target: {len(samples)} | Done: {len(done)} | Remaining: {len(pending)}")
    if not pending:
        print("All done.")
        return

    print("\n[1/2] Loading Base model...")
    base = BaseModelAdapter(args.base_model)
    print("[2/2] Loading PARM adapter...")
    parm = PARMAdapter(args.base_model, args.parm_adapter)
    print("Models loaded. Starting probing...\n")

    with out_path.open("a", encoding="utf-8") as f:
        for sample in tqdm(pending, desc="Oracle-probe", unit="prompt", ncols=80):
            prompt = text_value(sample.get("prompt")).strip()
            if not prompt:
                continue
            try:
                result = probe_one(prompt, base, parm, args.max_new_tokens, "cuda")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as exc:
                f.write(json.dumps({"prompt": prompt, "error": str(exc)}) + "\n")
                f.flush()

    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
