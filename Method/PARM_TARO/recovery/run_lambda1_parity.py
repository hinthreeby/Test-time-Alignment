"""Micro-diagnostic for original-PARM versus PARM-TARO lambda=1 parity.

This file is intentionally outside the protected PARM/ and router/ trees.  The
default source-only mode is CPU-safe.  Pass ``--run-model --device cuda`` to
collect real per-token traces after the exact PBLORA artifact is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

# Keep the lexical compatibility path. ``resolve()`` would cross the current
# PARM_TARO -> Method/PARM_TARO symlink and incorrectly select Method/ as root.
PROJECT_ROOT = Path(__file__).absolute().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PARM_TARO.adapters.parm_adapter import named_preference_to_parm, set_parm_preference
from PARM_TARO.decoding.adaptive import _model_step
from PARM_TARO.decoding.static_regression import parm_guided_logprobs
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.runtime import load_training_runtime


DEFAULT_OUTPUT = PROJECT_ROOT / "results/parm_taro/recovery/quick100"
AUTHOR_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _author_static(base_logprobs: torch.Tensor, guide_logprobs: torch.Tensor) -> torch.Tensor:
    """Exact normalized result of author formula M_base + M_reward.

    ``Sum.norm`` is two and ``Operator.normalize`` divides by that norm before
    applying logsumexp normalization.
    """

    return F.log_softmax((base_logprobs + guide_logprobs) / 2.0, dim=-1)


def _distribution_metrics(
    static: torch.Tensor, taro: torch.Tensor, *, top_k: int
) -> dict[str, Any]:
    difference = (static - taro).abs()
    static_top = torch.topk(static, min(top_k, static.shape[-1]), dim=-1).indices[0]
    taro_top = torch.topk(taro, min(top_k, taro.shape[-1]), dim=-1).indices[0]
    overlap = len(set(static_top.tolist()) & set(taro_top.tolist())) / len(static_top)
    return {
        "max_abs_logprob_diff": float(difference.max().item()),
        "mean_abs_logprob_diff": float(difference.mean().item()),
        "top1_agreement": bool(static.argmax(-1).eq(taro.argmax(-1)).all().item()),
        "topk_overlap": float(overlap),
        "static_top1_token_id": int(static.argmax(-1).item()),
        "taro_top1_token_id": int(taro.argmax(-1).item()),
        "static_topk_token_ids": static_top.tolist(),
        "taro_topk_token_ids": taro_top.tolist(),
    }


def source_only_probe(top_k: int) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    """Execute a deterministic CPU counterexample using independent equations."""

    base_logits = torch.tensor([[2.0, 0.5, -0.25, 1.25, -1.0]], dtype=torch.float64)
    guide_logits = torch.tensor([[-0.5, 1.75, 0.25, 0.75, -1.5]], dtype=torch.float64)
    base_lp = F.log_softmax(base_logits, dim=-1)
    guide_lp = F.log_softmax(guide_logits, dim=-1)
    author = _author_static(base_lp, guide_lp)
    taro = parm_guided_logprobs(base_logits, guide_logits, 1.0).to(torch.float64)
    stage10_static = parm_guided_logprobs(base_logits, guide_logits, 1.0).to(torch.float64)
    metrics = _distribution_metrics(author, taro, top_k=top_k)
    metrics["stage10_static_vs_bypass_max_abs_logprob_diff"] = float(
        (stage10_static - taro).abs().max().item()
    )
    trace = [{
        "base_logprobs": base_lp.cpu(),
        "guide_logprobs": guide_lp.cpu(),
        "author_static_fused_logprobs": author.cpu(),
        "taro_lambda1_fused_logprobs": taro.cpu(),
    }]
    return metrics, trace


def _load_prompts(path: Path, count: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) < count:
        raise ValueError(f"Need at least {count} validation records in {path}")
    rows = []
    for row in payload[:count]:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Validation record has no non-empty prompt")
        rows.append({"sample_id": str(row.get("sample_id", len(rows))), "prompt": prompt})
    return rows


def model_probe(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # The repository currently exposes PARM_TARO through a compatibility
    # symlink. Patch only module-level path anchors in this diagnostic process;
    # no project source or config is mutated.
    import PARM_TARO.adapters.parm_adapter as parm_adapter_module
    import PARM_TARO.training.runtime as runtime_module

    parm_adapter_module.PROJECT_ROOT = PROJECT_ROOT
    parm_adapter_module.PARM_ROOT = PROJECT_ROOT / "PARM"
    parm_adapter_module.PARM_PEFT_SRC = parm_adapter_module.PARM_ROOT / "peft" / "src"
    parm_adapter_module.PARM_ARITHMETIC_SRC = (
        parm_adapter_module.PARM_ROOT / "language-model-arithmetic" / "src"
    )
    parm_adapter_module.PARM_GENERATION_SOURCE = (
        parm_adapter_module.PARM_ROOT / "code" / "evaluation" / "generate_outputs.py"
    )
    runtime_module.PROJECT_ROOT = PROJECT_ROOT
    runtime_module.RESULTS_ROOT = PROJECT_ROOT / "results" / "parm_taro"

    config = ParmRouterTrainingConfig.load_json(args.runtime_config)
    config = replace(
        config,
        device=args.device,
        allow_cpu_fallback=False,
        max_validation_samples=args.num_prompts,
    )
    runtime = load_training_runtime(config)
    alpha = torch.tensor(args.alpha, dtype=torch.float32, device=runtime.device)
    set_parm_preference(runtime.guide_model, named_preference_to_parm(alpha))
    rows = _load_prompts(Path(args.validation), args.num_prompts)
    prompt_results: list[dict[str, Any]] = []
    trace_payload: list[dict[str, Any]] = []
    all_maxima: list[float] = []
    weighted_mean_sum = 0.0
    element_count = 0
    agreements: list[bool] = []
    overlaps: list[float] = []

    for row in rows:
        prompt = (
            AUTHOR_TEMPLATE.format(prompt=row["prompt"])
            if args.prompt_template == "parm_author"
            else row["prompt"]
        )
        encoded = runtime.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(runtime.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(runtime.device)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        base_input = input_ids
        guide_input = input_ids
        base_past = None
        guide_past = None
        generated_static: list[int] = []
        generated_taro: list[int] = []
        token_summaries: list[dict[str, Any]] = []
        tensors: list[dict[str, torch.Tensor]] = []

        with torch.inference_mode():
            for position in range(args.max_new_tokens):
                base_logits, base_past = _model_step(
                    runtime.base_model, base_input, attention_mask, position_ids, base_past
                )
                guide_logits, guide_past = _model_step(
                    runtime.guide_model, guide_input, attention_mask, position_ids, guide_past
                )
                runtime.alignment.validate_base_logits(base_logits)
                guide_logits = runtime.alignment.align_guide_logits(guide_logits)
                base_lp = F.log_softmax(base_logits.float(), dim=-1)
                guide_lp = F.log_softmax(guide_logits.float(), dim=-1)
                author = _author_static(base_lp, guide_lp)
                taro = parm_guided_logprobs(base_logits, guide_logits, 1.0)
                stage10_static = parm_guided_logprobs(base_logits, guide_logits, 1.0)
                step = _distribution_metrics(author, taro, top_k=args.top_k)
                step.update({
                    "position": position,
                    "stage10_static_vs_bypass_max_abs_logprob_diff": float(
                        (stage10_static - taro).abs().max().item()
                    ),
                })
                token_summaries.append(step)
                tensors.append({
                    "base_logprobs": base_lp.cpu(),
                    "guide_logprobs": guide_lp.cpu(),
                    "author_static_fused_logprobs": author.cpu(),
                    "taro_lambda1_fused_logprobs": taro.cpu(),
                })
                all_maxima.append(step["max_abs_logprob_diff"])
                weighted_mean_sum += step["mean_abs_logprob_diff"] * author.numel()
                element_count += author.numel()
                agreements.append(step["top1_agreement"])
                overlaps.append(step["topk_overlap"])
                static_token = step["static_top1_token_id"]
                taro_token = step["taro_top1_token_id"]
                generated_static.append(static_token)
                generated_taro.append(taro_token)
                if static_token != taro_token:
                    break
                next_token = torch.tensor([[static_token]], device=runtime.device)
                if static_token == runtime.tokenizer.eos_token_id:
                    break
                base_input = next_token
                guide_input = next_token
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)
                position_ids = torch.full(
                    (1, 1), attention_mask.shape[1] - 1, dtype=torch.long, device=runtime.device
                )

        identical = generated_static == generated_taro
        prompt_results.append({
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "prompt_template": args.prompt_template,
            "tokens_compared": len(token_summaries),
            "generated_output_identical": identical,
            "static_token_ids": generated_static,
            "taro_token_ids": generated_taro,
            "static_text": runtime.tokenizer.decode(generated_static, skip_special_tokens=True),
            "taro_text": runtime.tokenizer.decode(generated_taro, skip_special_tokens=True),
            "token_metrics": token_summaries,
        })
        trace_payload.append({"sample_id": row["sample_id"], "states": tensors})

    aggregate = {
        "max_abs_logprob_diff": max(all_maxima),
        "mean_abs_logprob_diff": weighted_mean_sum / element_count,
        "top1_agreement": sum(agreements) / len(agreements),
        "topk_overlap": sum(overlaps) / len(overlaps),
        "generated_output_identical_rate": sum(
            row["generated_output_identical"] for row in prompt_results
        ) / len(prompt_results),
        "states": len(agreements),
        "prompts": len(prompt_results),
        "model_load_info": runtime.model_load_info,
    }
    return {"aggregate": aggregate, "prompts": prompt_results}, trace_payload


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    aggregate = metrics.get("aggregate", metrics)
    conclusion = payload["conclusion"]
    return f"""# Lambda=1 parity micro-diagnostic

## Conclusion

**{conclusion}**

The original PARM author path and PARM-TARO are not exact normalized-distribution
equivalents at lambda=1. Author `ModelArithmetic` evaluates `M_base + M_reward`
as `log_softmax((log p_base + log p_guide) / 2)`, while PARM-TARO evaluates
`log_softmax(log p_base + log p_guide)`. The common factor 1/2 preserves ranking
and therefore normally preserves greedy tokens, but changes probabilities.

Stage-10 `parm_static` is a different comparator: it uses the PARM-TARO decoder
with `FixedLambdaProvider(1.0)`. Its parity with router-bypass lambda=1 is exact
by construction; that does not establish parity with the original PARM author
generation distribution.

## Measurements

- Mode: `{payload['mode']}`
- Real model executed: `{payload['real_model_executed']}`
- max_abs_logprob_diff: `{aggregate['max_abs_logprob_diff']}`
- mean_abs_logprob_diff: `{aggregate['mean_abs_logprob_diff']}`
- top1_agreement: `{aggregate['top1_agreement']}`
- topk_overlap: `{aggregate['topk_overlap']}`
- numerical tolerance: `{payload['tolerance']}`
- Full tensor trace: `{payload['trace_file']}`

## Gate

Router experiments must remain stopped for claims that require original-PARM
distribution parity. No protected source was changed and no retraining ran.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--runtime-config", default=str(PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"))
    parser.add_argument("--validation", default=str(PROJECT_ROOT / "dataset/parm_taro/validation.json"))
    parser.add_argument("--num-prompts", type=int, default=8, choices=range(5, 11))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--prompt-template", choices=("parm_author", "raw"), default="parm_author")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.run_model and args.device != "cuda":
        raise SystemExit("Real 7B parity is intentionally GPU-only; use --device cuda")
    if args.max_new_tokens <= 0 or args.top_k <= 0 or args.tolerance <= 0:
        raise SystemExit("max-new-tokens, top-k, and tolerance must be positive")

    if args.run_model:
        metrics, trace = model_probe(args)
        aggregate = metrics["aggregate"]
        mode = "real_validation_micro_diagnostic"
        real_model_executed = True
    else:
        aggregate, trace = source_only_probe(args.top_k)
        metrics = aggregate
        mode = "deterministic_cpu_source_equation_probe"
        real_model_executed = False

    passes = (
        aggregate["max_abs_logprob_diff"] <= args.tolerance
        and aggregate["top1_agreement"] >= 0.999
        and (
            not args.run_model
            or aggregate["generated_output_identical_rate"] >= 0.999
        )
    )
    conclusion = "LAMBDA1_PARITY_PASS" if passes else "LAMBDA1_PARITY_FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "lambda1_token_traces.pt"
    temporary_trace = trace_path.with_suffix(".pt.tmp")
    torch.save(trace, temporary_trace)
    os.replace(temporary_trace, trace_path)
    payload = {
        "schema_version": 1,
        "conclusion": conclusion,
        "mode": mode,
        "real_model_executed": real_model_executed,
        "device": args.device,
        "comparison": "original_parm_author_vs_parm_taro_fixed_lambda_1",
        "tolerance": args.tolerance,
        "alpha_named_order": ["helpfulness", "harmlessness"],
        "alpha": list(args.alpha),
        "prompt_template": args.prompt_template,
        "metrics": metrics,
        "trace_file": str(trace_path.relative_to(PROJECT_ROOT)),
        "stage10_static_note": (
            "Stage-10 parm_static and bypass lambda=1 call the same PARM-TARO "
            "fusion function and are exact by construction."
        ),
        "runtime_prerequisites": {
            "base_model_config_present": (PROJECT_ROOT / "models/tulu-2-7b/config.json").is_file(),
            "base_model_weight_index_present": (
                PROJECT_ROOT / "models/tulu-2-7b/pytorch_model.bin.index.json"
            ).is_file(),
            "tokenizer_present": (
                PROJECT_ROOT / "models/tulu-2-7b/tokenizer.model"
            ).is_file(),
            "pblora_adapter_config_present": (
                PROJECT_ROOT
                / "results/parm_taro/checkpoints/parm_pku_pblora/adapter_config.json"
            ).is_file(),
            "pblora_weights_present": any(
                path.is_file()
                for path in (
                    PROJECT_ROOT
                    / "results/parm_taro/checkpoints/parm_pku_pblora/adapter_model.safetensors",
                    PROJECT_ROOT
                    / "results/parm_taro/checkpoints/parm_pku_pblora/adapter_model.bin",
                )
            ),
        },
    }
    _atomic_text(args.output_dir / "lambda1_parity.json", json.dumps(payload, indent=2) + "\n")
    _atomic_text(args.output_dir / "lambda1_parity.md", _markdown(payload))
    print(conclusion)
    print(args.output_dir / "lambda1_parity.json")
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
