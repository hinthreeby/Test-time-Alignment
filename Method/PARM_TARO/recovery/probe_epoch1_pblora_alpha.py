"""Low-VRAM, next-token-only alpha sensitivity probe for recovered PBLoRA."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def find_project_root(source: Path) -> Path:
    """Find the checkout even when PARM_TARO is reached through a symlink."""
    start = source.absolute().parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError(f"Cannot find project root from {source}")


ROOT = find_project_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/parm_taro/recovery/pblora_repro/epoch1_validation"
BASE = ROOT / "models/tulu-2-7b"
ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
VALIDATION = ROOT / "dataset/parm_taro/validation.json"
ALPHAS = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))
TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def telemetry() -> dict[str, int | str]:
    command = ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        values = [item.strip() for item in subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip().split(",")]
        return {"gpu_name": values[0], "total_mib": int(values[1]), "free_mib": int(values[2]), "used_mib": int(values[3]), "utilization_pct": int(values[4])}
    except Exception as error:
        return {"error": repr(error)}


def memory_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "torch_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "torch_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "nvidia_smi": telemetry(),
    }


def write_blocked(reason: str, before: dict[str, Any]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["status", "reason"]
    with (OUT / "alpha_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow({"status": "NOT_RUN", "reason": reason})
    payload = {"status": "ALPHA_PROBE_BLOCKED_LOW_VRAM", "reason": reason, "gpu_before": before, "real_model_loaded": False}
    atomic_text(OUT / "alpha_probe_summary.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(OUT / "alpha_probe.md", f"# PBLoRA alpha probe\n\n**ALPHA_PROBE_BLOCKED_LOW_VRAM**\n\n{reason}\n\nNo model or evaluator was loaded.\n")
    print("ALPHA_PROBE_BLOCKED_LOW_VRAM")
    return 3


def load_model(torch: Any) -> tuple[Any, Any]:
    from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime
    vendored = activate_vendored_parm_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, load_in_4bit=True, device_map={"": 0})
    model = vendored.peft.PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, tokenizer


def distribution(model: Any, tokenizer: Any, prompt: str, alpha: tuple[float, float], max_prefix_tokens: int, torch: Any) -> tuple[Any, list[str], list[float]]:
    from PARM_TARO.adapters.parm_adapter import named_preference_to_parm, set_parm_preference
    named = torch.tensor(alpha, device="cuda", dtype=torch.float32)
    actual = named_preference_to_parm(named)
    updated = set_parm_preference(model, actual)
    readbacks = []
    seen = set()
    for collection in (model.named_parameters(), model.named_buffers()):
        for name, value in collection:
            if "pref_vec" in name and id(value) not in seen:
                seen.add(id(value)); readbacks.append(value.detach().float().cpu())
    if not readbacks or not all(torch.equal(value, actual.float().cpu()) for value in readbacks):
        raise RuntimeError("Requested alpha did not reach every PBLoRA preference tensor")
    encoded = tokenizer(TEMPLATE.format(prompt=prompt), return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prefix_tokens)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1).cpu()
    if not bool(torch.isfinite(logp).all()):
        raise FloatingPointError("PBLoRA emitted non-finite normalized log-probabilities")
    del encoded, logits
    return logp, updated, readbacks[0].tolist()


def pair_metrics(p_log: Any, q_log: Any, torch: Any) -> dict[str, Any]:
    p, q = p_log.exp(), q_log.exp()
    m = 0.5 * (p + q)
    logm = m.log()
    js = 0.5 * ((p * (p_log - logm)).sum() + (q * (q_log - logm)).sum())
    kl_pq = (p * (p_log - q_log)).sum()
    top_p = set(torch.topk(p_log, 10).indices.tolist()); top_q = set(torch.topk(q_log, 10).indices.tolist())
    return {
        "js": float(js), "kl_pq": float(kl_pq),
        "mean_abs_logprob_delta": float((p_log - q_log).abs().mean()),
        "l2_logprob_delta": float(torch.linalg.vector_norm(p_log - q_log)),
        "max_abs_logprob_delta": float((p_log - q_log).abs().max()),
        "top1_changed": int(p_log.argmax() != q_log.argmax()),
        "top10_overlap": len(top_p & top_q) / 10.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=5500)
    parser.add_argument("--num-prefixes", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--max-prefix-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--reload-min-free-mib",
        type=int,
        default=8192,
        help="Skip optional fresh reload below this post-delete free-VRAM safety margin",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    before = telemetry()
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "gpu_before": before, "adapter": str(ADAPTER), "num_prefixes": args.num_prefixes, "alphas": ALPHAS}, indent=2)); return 0
    if "free_mib" not in before or int(before["free_mib"]) < args.min_free_mib:
        return write_blocked(f"Need >= {args.min_free_mib} MiB free before load; telemetry={before}", before)

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        return write_blocked("CUDA is unavailable to this process despite host GPU status; model load refused.", before)
    torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    vram = {"before_model_load": memory_snapshot(torch)}
    rows_data = json.loads(VALIDATION.read_text(encoding="utf-8"))[: args.num_prefixes]
    model, tokenizer = load_model(torch)
    vram["after_model_load"] = memory_snapshot(torch)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if "pblora_" in n)
    traces: dict[str, Any] = {}
    csv_rows = []
    updated_count = None
    for row in rows_data:
        for alpha in ALPHAS:
            logp, updated, actual_readback = distribution(model, tokenizer, row["prompt"], alpha, args.max_prefix_tokens, torch)
            updated_count = len(updated) if updated_count is None else updated_count
            key = f"{row['sample_id']}|{alpha[0]:.2f},{alpha[1]:.2f}"
            traces[key] = logp.numpy()
            values, ids = torch.topk(logp, args.top_k)
            csv_rows.append({
                "sample_id": row["sample_id"], "requested_alpha": json.dumps(alpha),
                "actual_alpha_entering_pblora": json.dumps(actual_readback),
                "updated_pref_tensor_count": len(updated), "topk_token_ids": json.dumps(ids.tolist()),
                "topk_logprobs": json.dumps(values.tolist()), "normalized_logprobs_key": key,
            })

    # Establish a deterministic numerical-noise floor while model A is still
    # the sole GPU model. Fresh-reload equivalence is lower priority and may be
    # deferred after the primary 10x5 alpha experiment is safely on disk.
    repeat_differences = []
    for row in rows_data[:3]:
        key = f"{row['sample_id']}|0.50,0.50"
        repeated, _, _ = distribution(
            model, tokenizer, row["prompt"], (0.5, 0.5), args.max_prefix_tokens, torch
        )
        repeat_differences.append(float((torch.from_numpy(traces[key]) - repeated).abs().max()))
        del repeated

    metrics = []
    for row in rows_data:
        for left, right in itertools.combinations(ALPHAS, 2):
            left_key = f"{row['sample_id']}|{left[0]:.2f},{left[1]:.2f}"; right_key = f"{row['sample_id']}|{right[0]:.2f},{right[1]:.2f}"
            item = pair_metrics(torch.from_numpy(traces[left_key]), torch.from_numpy(traces[right_key]), torch)
            item.update({"sample_id": row["sample_id"], "left_alpha": left, "right_alpha": right, "extreme": left == ALPHAS[0] and right == ALPHAS[-1], "adjacent": ALPHAS.index(right) == ALPHAS.index(left) + 1})
            metrics.append(item)

    # Persist the primary alpha outputs before releasing model A. All arrays
    # are CPU NumPy arrays; no GPU tensor is retained in ``traces``.
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "alpha_probe_logprobs.npz", **traces)
    with (OUT / "alpha_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)

    peak_probe = torch.cuda.max_memory_allocated() / 1024**2
    vram["peak_probe"] = {
        "torch_peak_allocated_mib": peak_probe,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "nvidia_smi": telemetry(),
    }
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    vram["after_model_delete"] = memory_snapshot(torch)

    reload_status = "RELOAD_EQUIVALENCE_DEFERRED_LOW_VRAM"
    reload_max_abs = None
    reload_differences: list[float] = []
    post_delete_free = vram["after_model_delete"]["nvidia_smi"].get("free_mib")
    memory_released = (
        vram["after_model_delete"]["torch_allocated_mib"]
        < vram["after_model_load"]["torch_allocated_mib"]
    )
    if (
        memory_released
        and isinstance(post_delete_free, int)
        and post_delete_free >= args.reload_min_free_mib
    ):
        try:
            torch.cuda.reset_peak_memory_stats()
            reload_model, reload_tokenizer = load_model(torch)
            vram["after_serial_reload"] = memory_snapshot(torch)
            for row in rows_data[:3]:
                key = f"{row['sample_id']}|0.50,0.50"
                reload_logp, _, _ = distribution(
                    reload_model, reload_tokenizer, row["prompt"], (0.5, 0.5),
                    args.max_prefix_tokens, torch,
                )
                reload_differences.append(
                    float((torch.from_numpy(traces[key]) - reload_logp).abs().max())
                )
                del reload_logp
            reload_max_abs = max(reload_differences)
            reload_status = "PASS"
            del reload_model, reload_tokenizer
        except torch.cuda.OutOfMemoryError:
            reload_status = "RELOAD_EQUIVALENCE_DEFERRED_LOW_VRAM"
        finally:
            # Locals may be absent if loading failed part-way through.
            if "reload_model" in locals():
                del reload_model
            if "reload_tokenizer" in locals():
                del reload_tokenizer
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            vram["after_reload_cleanup"] = memory_snapshot(torch)

    extremes = [item for item in metrics if item["extreme"]]
    repeat_max_abs = max(repeat_differences)
    empirical_noise = reload_max_abs if reload_max_abs is not None else repeat_max_abs
    noise = max(empirical_noise, float(torch.finfo(torch.float32).eps))
    clear_prompts = sum(item["max_abs_logprob_delta"] > 10.0 * noise for item in extremes)
    all_delta = [item["mean_abs_logprob_delta"] for item in metrics]
    classification = "PASS" if clear_prompts >= max(3, math.ceil(0.6 * len(extremes))) else ("WEAK" if max(item["max_abs_logprob_delta"] for item in metrics) > noise else "FAIL")
    summary = {
        "status": "COMPLETE", "classification": classification, "gpu_before": before,
        "vram_snapshots": vram,
        "torch_peak_allocated_mib": peak_probe,
        "trainable_parameter_count_in_inference": trainable, "adapter_parameter_count": adapter_params,
        "preference_tensor_count": updated_count, "num_prefixes": len(rows_data), "num_distributions": len(csv_rows),
        "mean_js_divergence_all_pairs": sum(item["js"] for item in metrics) / len(metrics),
        "mean_extreme_alpha_js_divergence": sum(item["js"] for item in extremes) / len(extremes),
        "mean_abs_logprob_delta_all_pairs": sum(all_delta) / len(all_delta),
        "top1_change_rate_all_pairs": sum(item["top1_changed"] for item in metrics) / len(metrics),
        "mean_top10_overlap_all_pairs": sum(item["top10_overlap"] for item in metrics) / len(metrics),
        "same_model_repeat_max_abs_logprob_noise": repeat_max_abs,
        "same_model_repeat_max_abs_by_prefix": repeat_differences,
        "reload_equivalence_status": reload_status,
        "same_alpha_reload_max_abs_logprob_noise": reload_max_abs,
        "same_alpha_reload_max_abs_by_prefix": reload_differences,
        "noise_anchored_rule": "PASS if >=60% and at least 3 extreme-alpha prompts exceed 10x max(reload noise,float32 epsilon); WEAK if reproducibly above noise; otherwise FAIL.",
        "pairwise_metrics": metrics,
    }
    atomic_text(OUT / "alpha_probe_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    reload_display = "deferred (low VRAM)" if reload_max_abs is None else f"{reload_max_abs:.8g}"
    atomic_text(OUT / "alpha_probe.md", f"# PBLoRA alpha probe\n\n**{classification}**\n\n- Prefixes: {len(rows_data)}\n- Mean JS: {summary['mean_js_divergence_all_pairs']:.8g}\n- Extreme-alpha JS: {summary['mean_extreme_alpha_js_divergence']:.8g}\n- Same-model repeat max-abs noise: {repeat_max_abs:.8g}\n- Fresh-reload max-abs noise: {reload_display}\n- Reload status: {reload_status}\n- Peak torch allocation: {summary['torch_peak_allocated_mib']:.1f} MiB\n\nNo generation or reward/cost evaluator was used.\n")
    print(classification)
    return 0 if classification in {"PASS", "WEAK"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
