#!/usr/bin/env python3
"""
DynaCAP-PARM (CAP_PARM_MATO) joint inference.
Outputs results/dynacap_parm.json (same schema as other methods).
"""

from __future__ import annotations

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

from src.models.base_adapter import BaseModelAdapter
from src.models.parm_adapter import PARMAdapter
from src.models.objective_tracker import ObjectiveTracker
from src.models.token_router import TokenRouter
from src.controllers.dynamic_preference import DynamicPreferenceController
from src.decoding.fusion import fuse_scores

DEFAULT_BASE    = str(ROOT_DIR / "models" / "tulu-2-7b")
DEFAULT_PARM    = str(ROOT_DIR / "Method" / "PARM_TARO" / "results" / "parm_taro"
                      / "recovery" / "reproduced" / "pblora" / "final_checkpoint")
DEFAULT_DATASET = str(ROOT_DIR / "dataset" / "rad_benchmark" / "all.jsonl")
DEFAULT_OUTPUT  = str(ROOT_DIR / "results" / "dynacap_parm.json")


def parse_args():
    p = argparse.ArgumentParser(description="DynaCAP-PARM inference → results/dynacap_parm.json")
    p.add_argument("--base-model",      default=DEFAULT_BASE)
    p.add_argument("--parm-adapter",    default=DEFAULT_PARM)
    p.add_argument("--dataset-path",    default=DEFAULT_DATASET)
    p.add_argument("--output-path",     default=DEFAULT_OUTPUT)
    p.add_argument("--tracker-ckpt",    default=None,
                   help="ObjectiveTracker checkpoint. If None, uses oracle one-hot probing.")
    p.add_argument("--router-ckpt",     default=None,
                   help="TokenRouter checkpoint. If None, uses fixed w.")
    p.add_argument("--num-prompts",     type=int,   default=1000)
    p.add_argument("--max-new-tokens",  type=int,   default=64)
    p.add_argument("--alpha-help",      type=float, default=0.5)
    p.add_argument("--alpha-harm",      type=float, default=0.5)
    p.add_argument("--w-fixed",         type=float, default=1.0)
    p.add_argument("--fusion-mode",     choices=["convex", "parm_product"], default="convex")
    p.add_argument("--dynamic-alpha",   action="store_true")
    p.add_argument("--dynamic-w",       action="store_true")
    p.add_argument("--update-interval", type=int,   default=8)
    p.add_argument("--temperature",     type=float, default=0.5)
    p.add_argument("--smoothing",       type=float, default=0.25)
    p.add_argument("--kl-budget",       type=float, default=0.1)
    p.add_argument("--gate-threshold",  type=float, default=0.5)
    p.add_argument("--seed",            type=int,   default=42)
    args = p.parse_args()
    if args.num_prompts < 0 or args.max_new_tokens <= 0:
        p.error("--num-prompts must be >= 0 and --max-new-tokens must be > 0")
    if min(args.alpha_help, args.alpha_harm) < 0 or abs(args.alpha_help + args.alpha_harm - 1.0) >= 1e-5:
        p.error("alpha values must be non-negative and sum to 1")
    if not 0.0 <= args.w_fixed <= 1.0:
        p.error("--w-fixed must be in [0, 1]")
    if args.dynamic_w and not args.router_ckpt:
        p.error("--dynamic-w requires a trained --router-ckpt")
    if args.update_interval <= 0 or args.temperature <= 0:
        p.error("--update-interval and --temperature must be positive")
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


def compute_deficits(cumulative: torch.Tensor, alpha_u: torch.Tensor) -> torch.Tensor:
    # Convert arbitrary signed reward scales to a simplex before comparing
    # observed objective satisfaction with the user's requested allocation.
    share = torch.softmax(cumulative.float(), dim=-1)
    return alpha_u - share


def sample_key(sample: dict, index: int):
    if sample.get("md5_hash"):
        return "md5", str(sample["md5_hash"])
    if sample.get("id") is not None:
        return "id", str(sample["id"])
    if sample.get("sample_id") is not None:
        return "sample_id", str(sample["sample_id"])
    return "index", str(index)


def generate_one(
    prompt: str,
    base, parm,
    tracker, router, dyn_ctrl,
    alpha_user, args, device,
) -> tuple[str, dict]:
    """Generate one response with cache-correct lazy PARM synchronization."""
    alpha_t   = list(alpha_user)
    alpha_u_t = torch.tensor(alpha_user, dtype=torch.float32, device=device)
    cum_rew   = torch.zeros(len(alpha_user), device=device)
    eos_id    = base.tokenizer.eos_token_id
    obj_vecs  = [[1.0, 0.0], [0.0, 1.0]]

    prompt_text = format_prompt(prompt)
    input_ids   = base.tokenizer.encode(prompt_text, return_tensors="pt").to(base.device)
    base_state  = base.prefill(input_ids)
    output_tokens = []
    parm_state = None
    parm_alpha = None
    parm_synced_length = 0
    parm_calls = 0
    parm_processed_tokens = 0
    replay_tokens = 0
    probe_calls = 0
    guided_steps = 0
    prev_gate     = None

    for step in range(args.max_new_tokens):
        bs = base_state.next_scores

        # ── Reward signal ──────────────────────────────────────
        if args.dynamic_alpha and tracker is not None:
            with torch.no_grad():
                tok_out = tracker(bs, alpha_u_t, step, args.max_new_tokens)
            reward_vec = tok_out.rewards[0].float()
            unc = tok_out.uncertainty[0].item()
        elif args.dynamic_alpha:
            # Oracle path: every preference gets a full-prefix replay.  KV
            # caches are alpha-dependent and cannot be shared across probes.
            probed = parm.probe_prefix(input_ids, obj_vecs)
            probe_calls += len(obj_vecs)
            base_lp = F.log_softmax(bs, dim=-1)
            tok_id  = torch.argmax(bs, dim=-1).item()
            r_k = [(F.log_softmax(ps, dim=-1)[0, tok_id] - base_lp[0, tok_id]).item() for ps in probed]
            reward_vec = torch.tensor(r_k, device=device)
            unc = 0.0
        else:
            reward_vec = torch.zeros_like(cum_rew)
            unc = 0.0

        cum_rew += reward_vec

        # ── Dynamic alpha ──────────────────────────────────────
        if args.dynamic_alpha and step > 0 and step % args.update_interval == 0:
            alpha_t, _ = dyn_ctrl.update(alpha_user, alpha_t, cum_rew.tolist(),
                                         uncertainty=unc if unc > 0 else None)

        # ── Token router ───────────────────────────────────────
        if args.dynamic_w and router is not None:
            alpha_t_t = torch.tensor(alpha_t, dtype=torch.float32, device=device)
            if router.feature_schema == "compact_v1":
                # The historical Gate-E state export did not record deficits
                # or previous-gate state; match its zero-filled training schema.
                deficits = torch.zeros_like(alpha_u_t)
                router_prev_gate = None
            else:
                deficits = compute_deficits(cum_rew, alpha_u_t)
                router_prev_gate = prev_gate
            with torch.no_grad():
                r_out = router(bs, alpha_t_t, deficits, step,
                               args.max_new_tokens, router_prev_gate, args.gate_threshold)
            w_t       = r_out.w_t[0].item()
            prev_gate = (r_out.gate_prob > args.gate_threshold).float()
        else:
            w_t = args.w_fixed

        # ── Fuse & sample ──────────────────────────────────────
        if w_t > 0:
            guided_steps += 1
            alpha_changed = parm_alpha is None or any(
                abs(left - right) > 1e-8 for left, right in zip(parm_alpha, alpha_t)
            )
            if parm_state is None or alpha_changed:
                parm_state = parm.prefill(input_ids, alpha_t)
                parm_calls += 1
                parm_processed_tokens += int(input_ids.shape[1])
                replay_tokens += len(output_tokens)
            else:
                pending = input_ids[:, parm_synced_length:]
                if pending.shape[1] > 0:
                    parm_state = parm.step(pending, alpha_t, parm_state.cache)
                    parm_calls += 1
                    parm_processed_tokens += int(pending.shape[1])
                    replay_tokens += max(0, int(pending.shape[1]) - 1)
            parm_alpha = list(alpha_t)
            parm_synced_length = int(input_ids.shape[1])
            ps     = parm_state.next_scores
            fused  = fuse_scores(bs, ps, w_t, mode=args.fusion_mode)
        else:
            fused = bs

        next_tok = torch.argmax(fused, dim=-1)
        tok = next_tok.item()
        output_tokens.append(tok)
        if tok == eos_id:
            break

        base_state = base.step(next_tok, base_state.cache)
        input_ids = torch.cat([input_ids, next_tok.reshape(1, 1)], dim=1)

    response = base.tokenizer.decode(output_tokens, skip_special_tokens=True)
    generated = max(len(output_tokens), 1)
    stats = {
        "parm_calls": parm_calls,
        "parm_processed_tokens": parm_processed_tokens,
        "replay_tokens": replay_tokens,
        "probe_calls": probe_calls,
        "intervention_rate": round(guided_steps / generated, 4),
    }
    return response, stats


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_user = [args.alpha_help, args.alpha_harm]

    samples = load_samples(args.dataset_path, args.num_prompts)
    out_path = Path(args.output_path)
    existing = json.loads(out_path.read_text()) if out_path.is_file() else []
    by_key = {
        sample_key(row, index): row
        for index, row in enumerate(existing) if isinstance(row, dict)
    }
    pending = [
        (i, sample) for i, sample in enumerate(samples)
        if by_key.get(sample_key(sample, i), {}).get("status") != "success"
    ]

    # Build method tag
    mode_tag = ("dyn_alpha+" if args.dynamic_alpha else "") + ("dyn_w" if args.dynamic_w else "fixed_w")
    method   = f"DynaCAP-PARM[{mode_tag}]"

    print(f"Method : {method}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output : {args.output_path}")
    print(f"Target : {len(samples)} | Done: {len(samples) - len(pending)} | Remaining: {len(pending)}")
    if not pending:
        print("All done.")
        return

    print("\nLoading models...")
    base  = BaseModelAdapter(args.base_model)
    parm  = PARMAdapter(args.base_model, args.parm_adapter)

    tracker = None
    if args.tracker_ckpt:
        tracker = ObjectiveTracker.load(args.tracker_ckpt).to(device).eval()
        print(f"Loaded ObjectiveTracker from {args.tracker_ckpt}")

    router = None
    if args.dynamic_w:
        router = TokenRouter.load(args.router_ckpt).to(device).eval()
        print(f"Loaded TokenRouter from {args.router_ckpt}")

    dyn_ctrl = DynamicPreferenceController(
        temperature=args.temperature,
        smoothing=args.smoothing,
        kl_budget=args.kl_budget,
    )

    from tqdm import tqdm
    for idx, sample in tqdm(pending, desc=method, unit="prompt"):
        prompt = text_value(sample.get("prompt")).strip()
        if not prompt:
            continue
        t0 = time.perf_counter()
        try:
            response, generation_stats = generate_one(
                prompt, base, parm, tracker, router, dyn_ctrl,
                alpha_user, args, device,
            )
            if not response.strip():
                raise RuntimeError("Empty response")
            status, error = "success", None
        except Exception as exc:
            response = ""
            generation_stats = {
                "parm_calls": 0, "parm_processed_tokens": 0,
                "replay_tokens": 0, "probe_calls": 0,
                "intervention_rate": 0.0,
            }
            status, error = "failed", f"{type(exc).__name__}: {exc}"

        key = sample_key(sample, idx)
        by_key[key] = {
            "id":               sample.get("id", idx),
            "_record_key":      list(key),
            "md5_hash":         sample.get("md5_hash"),
            "prompt":           prompt,
            "reference":        text_value(sample.get("continuation")),
            "response":         response,
            "method":           method,
            "fusion_mode":      args.fusion_mode,
            "w_fixed":          args.w_fixed,
            "alpha_helpfulness":  args.alpha_help,
            "alpha_harmlessness": args.alpha_harm,
            "dynamic_alpha":    args.dynamic_alpha,
            "dynamic_w":        args.dynamic_w,
            **generation_stats,
            "base_model":       args.base_model,
            "parm_adapter":     args.parm_adapter,
            "latency":          round(time.perf_counter() - t0, 4),
            "status":           status,
            "error":            error,
        }
        save_results(list(by_key.values()), args.output_path)

    success = sum(1 for r in by_key.values() if r.get("status") == "success")
    print(f"\nDone: {success}/{len(samples)} | Saved → {args.output_path}")


if __name__ == "__main__":
    main()
