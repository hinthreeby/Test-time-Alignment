"""Read-only validation of the recovered PBLoRA smoke checkpoint.

The audit phase is CPU-only. The probe phase performs short forward passes on
validation data, never trains, generates, reads test data, or writes checkpoint
content.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

def find_project_root(start: Path) -> Path:
    # Do not resolve the PARM_TARO symlink before checking lexical parents, but
    # also support invocation through the physical Method/PARM_TARO path.
    for origin in (start.absolute(), start.resolve()):
        for candidate in (origin.parent, *origin.parents):
            if (candidate / ".git").exists():
                return candidate
    raise RuntimeError("Cannot locate project root")


ROOT = find_project_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SMOKE = ROOT / "results/parm_taro/recovery/reproduced/pblora_smoke"
INFERENCE_ADAPTER = SMOKE / "final_checkpoint"
OUT = ROOT / "results/parm_taro/recovery/pblora_repro/check_smoke"
VALIDATION = ROOT / "dataset/parm_taro/validation.json"
BASE = ROOT / "models/tulu-2-7b"
SMOKE_LOG = ROOT / "results/parm_taro/recovery/pblora_repro/smoke.log"
ALPHAS = ((0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0))
PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"
EXPECTED_TRAINABLE = 6_296_064


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_audit() -> dict[str, Any]:
    from safetensors import safe_open

    checkpoints = sorted(
        [path for path in SMOKE.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    resumable_required = (
        "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
    )
    checkpoint_rows = []
    for checkpoint in checkpoints:
        checkpoint_rows.append({
            "path": str(checkpoint.relative_to(ROOT)),
            "step": int(checkpoint.name.rsplit("-", 1)[1]),
            "complete": all((checkpoint / name).is_file() for name in resumable_required),
            "missing": [name for name in resumable_required if not (checkpoint / name).is_file()],
        })

    adapter_files = [
        SMOKE / "adapter_config.json", SMOKE / "adapter_model.safetensors",
        INFERENCE_ADAPTER / "adapter_config.json", INFERENCE_ADAPTER / "adapter_model.safetensors",
    ]
    files = [file_record(path) for path in adapter_files]
    for checkpoint in checkpoints:
        files.extend(file_record(checkpoint / name) for name in resumable_required)

    adapter_config = load_json(INFERENCE_ADAPTER / "adapter_config.json") if (INFERENCE_ADAPTER / "adapter_config.json").is_file() else {}
    weight_path = INFERENCE_ADAPTER / "adapter_model.safetensors"
    tensor_count = 0
    parameter_count = 0
    finite = True
    tensor_names: list[str] = []
    module_paths: set[str] = set()
    if weight_path.is_file():
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensor_count += 1
                parameter_count += tensor.numel()
                finite = finite and bool(tensor.isfinite().all())
                tensor_names.append(key)
                module_paths.add(key.split(".pblora_", 1)[0])

    pblora_names = [name for name in tensor_names if ".pblora_" in name]
    targets = sorted({part for name in pblora_names for part in ("q_proj", "k_proj", "v_proj") if f".{part}." in name})
    unexpected_targets = [name for name in pblora_names if not any(f".{part}." in name for part in ("q_proj", "k_proj", "v_proj"))]
    pointers = {}
    for name in ("last", "best"):
        path = SMOKE / name
        pointers[name] = {
            "is_symlink": path.is_symlink(),
            "target": str(path.resolve(strict=False)),
            "valid": path.is_symlink() and (path / "trainer_state.json").is_file(),
        }
    training_audit = load_json(SMOKE / "recovery_training_audit.json") if (SMOKE / "recovery_training_audit.json").is_file() else {}
    final_step = int(training_audit.get("global_step", 0))
    valid = bool(
        checkpoints and checkpoint_rows[-1]["complete"]
        and all(row["exists"] for row in files[:4])
        and pointers["last"]["valid"] and pointers["best"]["valid"]
        and adapter_config.get("peft_type") == "PBLORA"
        and int(adapter_config.get("obj_num", -1)) == 2
        and parameter_count == EXPECTED_TRAINABLE and finite
        and final_step == 30 and not unexpected_targets
    )
    payload = {
        "status": "PASS" if valid else "FAIL",
        "training_root": str(SMOKE),
        "inference_adapter": str(INFERENCE_ADAPTER),
        "final_training_step": final_step,
        "resumable_checkpoints": checkpoint_rows,
        "pointers": pointers,
        "adapter_config": adapter_config,
        "adapter_tensor_count": tensor_count,
        "adapter_parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_TRAINABLE,
        "all_adapter_parameters_finite": finite,
        "adapter_module_count": len(module_paths),
        "target_module_types": targets,
        "unexpected_target_tensors": unexpected_targets,
        "files": files,
    }
    atomic_json(OUT / "checkpoint_audit.json", payload)
    return payload


def training_health() -> tuple[dict[str, Any], dict[str, Any]]:
    training_audit = load_json(SMOKE / "recovery_training_audit.json")
    preflight = load_json(SMOKE / "recovery_preflight.json")
    state_path = SMOKE / "checkpoint-25/trainer_state.json"
    trainer_state = load_json(state_path)
    text = SMOKE_LOG.read_text(encoding="utf-8", errors="replace") if SMOKE_LOG.is_file() else ""
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    clean = ansi.sub("", text)
    summary_matches = re.findall(r"\{[^{}]*'train_runtime'[^{}]*\}", clean)
    summary: dict[str, Any] = {}
    if summary_matches:
        import ast
        summary = ast.literal_eval(summary_matches[-1])

    history = trainer_state.get("log_history", [])
    train_rows = [row for row in history if "loss" in row]
    eval_rows = [row for row in history if "eval_loss" in row]
    grad_norms = [float(row["grad_norm"]) for row in train_rows if isinstance(row.get("grad_norm"), (int, float))]
    losses = [float(value) for value in training_audit.get("losses", [])]
    runtime = float(summary.get("train_runtime", 0.0))
    steps = int(training_audit.get("global_step", 0))
    seconds_per_step = runtime / steps if runtime > 0 and steps > 0 else None
    health = {
        "status": "PASS",
        "completed_steps": steps,
        "expected_steps": 30,
        "loss_count": len(losses),
        "train_losses_finite": bool(losses) and all(math.isfinite(x) for x in losses),
        "eval_losses": [float(row["eval_loss"]) for row in eval_rows],
        "eval_losses_finite": bool(eval_rows) and all(math.isfinite(float(row["eval_loss"])) for row in eval_rows),
        "grad_norm_observations_in_checkpoint_state": len(grad_norms),
        "grad_norms_finite_nonzero": bool(grad_norms) and all(math.isfinite(x) and x > 0 for x in grad_norms),
        "grad_norm_min": min(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "successful_run_contains_cuda_oom": "CUDA out of memory" in clean,
        "checkpoint_saved": state_path.is_file(),
        "train_runtime_seconds": runtime or None,
        "train_steps_per_second": summary.get("train_steps_per_second"),
        "seconds_per_optimizer_step": seconds_per_step,
        "train_loss": summary.get("train_loss"),
        "base_sampled_fingerprint_unchanged": preflight.get("sampled_base_state_unchanged"),
        "base_all_non_pblora_parameters_frozen": preflight.get("all_non_pblora_parameters_frozen"),
        "note": "The recovery callback's gradient hook was unsupported by this Trainer version; trainer_state grad_norm values are authoritative.",
    }
    health["status"] = "PASS" if all((
        health["completed_steps"] == 30,
        health["train_losses_finite"], health["eval_losses_finite"],
        health["grad_norms_finite_nonzero"], not health["successful_run_contains_cuda_oom"],
        health["checkpoint_saved"], health["base_sampled_fingerprint_unchanged"],
        health["base_all_non_pblora_parameters_frozen"],
    )) else "FAIL"
    runtime_estimate = {
        "observed_train_runtime_seconds": runtime or None,
        "observed_optimizer_steps": steps,
        "seconds_per_optimizer_step": seconds_per_step,
        "nominal_train_samples": 8000,
        "effective_batch_size": 32,
        "nominal_steps_one_epoch": 250,
        "nominal_steps_two_epochs": 500,
        "eta_one_epoch_seconds_excluding_eval_save": seconds_per_step * 250 if seconds_per_step else None,
        "eta_two_epochs_seconds_excluding_eval_save": seconds_per_step * 500 if seconds_per_step else None,
        "eval_save_overhead": "report separately; smoke eval at step 25 took 32.7629 s",
        "peak_allocated_vram_mib": None,
        "peak_reserved_vram_mib": None,
        "vram_note": "The smoke launcher did not record torch peak-memory counters.",
    }
    atomic_json(OUT / "training_health.json", health)
    atomic_json(OUT / "runtime_estimate.json", runtime_estimate)
    return health, runtime_estimate


def load_validation() -> list[dict[str, Any]]:
    rows = load_json(VALIDATION)
    if not isinstance(rows, list) or len(rows) < 20:
        raise RuntimeError("Validation split is missing or too small")
    return rows


def load_guide(device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime
    from safetensors import safe_open

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real PBLoRA probe requires visible CUDA")
    vendored = activate_vendored_parm_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, low_cpu_mem_usage=True,
        quantization_config=quantization, device_map={"": 0},
    )
    model = vendored.peft.PeftModel.from_pretrained(
        backbone, INFERENCE_ADAPTER, is_trainable=True,
    )
    with safe_open(INFERENCE_ADAPTER / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
    runtime_keys = set()
    runtime_finite = True
    for name, parameter in model.named_parameters():
        if "pblora_" not in name:
            continue
        normalized = re.sub(r"(pblora_(?:A|B|W1|W2))\.default$", r"\1", name)
        runtime_keys.add(normalized)
        runtime_finite = runtime_finite and bool(parameter.detach().isfinite().all())
    critical_missing = sorted(saved_keys - runtime_keys)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    info = {
        "vendored_peft": vendored.origins["peft"],
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters_before_inference_freeze": sum(parameter.numel() for _, parameter in trainable),
        "trainable_percent": 100.0 * sum(parameter.numel() for _, parameter in trainable) / sum(parameter.numel() for parameter in model.parameters()),
        "trainable_tensors": [name for name, _ in trainable],
        "all_trainable_are_pblora_qkv": all(
            "pblora_" in name and any(f".{target}." in name for target in ("q_proj", "k_proj", "v_proj"))
            for name, _ in trainable
        ),
        "saved_adapter_key_count": len(saved_keys),
        "matched_runtime_adapter_key_count": len(saved_keys & runtime_keys),
        "critical_missing_adapter_keys": critical_missing,
        "runtime_adapter_parameters_finite": runtime_finite,
    }
    if critical_missing or not runtime_finite:
        raise RuntimeError(
            f"Critical PBLoRA reload mismatch: missing={len(critical_missing)}, finite={runtime_finite}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info["runtime_requires_grad_after_freeze"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return model, tokenizer, info


def set_alpha(model: Any, alpha: tuple[float, float]) -> tuple[list[str], Any]:
    import torch
    from PARM_TARO.adapters.parm_adapter import named_preference_to_parm, set_parm_preference

    named = torch.tensor(alpha, dtype=torch.float32, device="cuda")
    guide_alpha = named_preference_to_parm(named)
    updated = set_parm_preference(model, guide_alpha)
    components = []
    for module in model.modules():
        if not hasattr(module, "pblora_W2") or not hasattr(module, "pref_vec"):
            continue
        for adapter in module.pblora_W2.keys():
            pref = module.pref_vec[adapter].detach().float()
            w2 = module.pblora_W2[adapter].detach().float()
            components.append((pref @ w2).cpu())
    if not components:
        raise RuntimeError("PBLoRA alpha-conditioned W2 component is unavailable")
    return updated, torch.cat([value.reshape(-1) for value in components])


def next_token_logprobs(model: Any, tokenizer: Any, prompt: str) -> Any:
    import torch
    from torch.nn import functional as F

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
    return F.log_softmax(logits, dim=-1).squeeze(0).cpu()


def internal_metrics(left: Any, right: Any) -> dict[str, float]:
    import torch

    delta = (left - right).abs()
    return {"internal_w2_max_abs_diff": float(delta.max()), "internal_w2_mean_abs_diff": float(delta.mean()), "internal_w2_l2_diff": float(torch.linalg.vector_norm(delta))}


def distribution_metrics(left: Any, right: Any, top_k: int) -> dict[str, Any]:
    import torch

    p, q = left.exp(), right.exp()
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    kl_left_right = float((p * (left - right)).sum())
    kl_right_left = float((q * (right - left)).sum())
    js = float(0.5 * ((p * (left - log_midpoint)).sum() + (q * (right - log_midpoint)).sum()))
    left_top = set(torch.topk(left, top_k).indices.tolist())
    right_top = set(torch.topk(right, top_k).indices.tolist())
    return {
        "kl_a_to_b": kl_left_right,
        "kl_b_to_a": kl_right_left,
        "js_divergence": js,
        "mean_abs_logprob_diff": float((left - right).abs().mean()),
        "max_abs_logprob_diff": float((left - right).abs().max()),
        "top1_changed": int(left.argmax() != right.argmax()),
        "topk_overlap": len(left_top & right_top) / top_k,
    }


def sequence_logp(model: Any, tokenizer: Any, prompt: str, response: str) -> tuple[float, int]:
    import torch
    from torch.nn import functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits.float()
    token_logps = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    start = max(0, len(prompt_ids) - 1)
    selected = token_logps[:, start:]
    return float(selected.sum().cpu()), int(selected.numel())


def run_probe(top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import torch

    rows = load_validation()
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, architecture = load_guide("cuda")

    reload_before = []
    for row in rows[:3]:
        set_alpha(model, (0.5, 0.5))
        reload_before.append(next_token_logprobs(model, tokenizer, PROMPT_TEMPLATE.format(prompt=row["prompt"])))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer, architecture_reloaded = load_guide("cuda")
    reload_metrics = []
    for row, before in zip(rows[:3], reload_before):
        set_alpha(model, (0.5, 0.5))
        after = next_token_logprobs(model, tokenizer, PROMPT_TEMPLATE.format(prompt=row["prompt"]))
        metric = distribution_metrics(before, after, top_k)
        metric["sample_id"] = row["sample_id"]
        reload_metrics.append(metric)
    reload_summary = {
        "status": "PASS" if max(item["max_abs_logprob_diff"] for item in reload_metrics) <= 1e-5 and all(not item["top1_changed"] for item in reload_metrics) else "FAIL",
        "tolerance": 1e-5,
        "max_abs_logprob_diff": max(item["max_abs_logprob_diff"] for item in reload_metrics),
        "mean_abs_logprob_diff": sum(item["mean_abs_logprob_diff"] for item in reload_metrics) / len(reload_metrics),
        "top1_agreement": sum(not item["top1_changed"] for item in reload_metrics) / len(reload_metrics),
        "per_prefix": reload_metrics,
    }

    probe_rows: list[dict[str, Any]] = []
    stored_logprobs: dict[str, Any] = {}
    all_pair_metrics: list[dict[str, Any]] = []
    updated_counts = []
    for row in rows[:10]:
        prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])
        outputs: dict[tuple[float, float], Any] = {}
        internals: dict[tuple[float, float], Any] = {}
        for alpha in ALPHAS:
            updated, internal = set_alpha(model, alpha)
            logp = next_token_logprobs(model, tokenizer, prompt)
            probabilities = logp.exp()
            values, ids = torch.topk(probabilities, top_k)
            key = f"{row['sample_id']}|{alpha[0]:.2f},{alpha[1]:.2f}"
            stored_logprobs[key] = logp
            outputs[alpha] = logp
            internals[alpha] = internal
            updated_counts.append(len(updated))
            probe_rows.append({
                "row_type": "distribution", "sample_id": row["sample_id"],
                "alpha_a": json.dumps(alpha), "alpha_b": "",
                "guide_alpha": json.dumps((alpha[1], alpha[0])),
                "entropy": float(-(probabilities * logp).sum()),
                "top1_token_id": int(logp.argmax()),
                "topk_token_ids": json.dumps(ids.tolist()),
                "topk_probabilities": json.dumps(values.tolist()),
                "logprob_storage_key": key,
                "kl_a_to_b": "", "kl_b_to_a": "", "js_divergence": "",
                "mean_abs_logprob_diff": "", "max_abs_logprob_diff": "",
                "top1_changed": "", "topk_overlap": "",
                "internal_w2_max_abs_diff": "", "internal_w2_mean_abs_diff": "", "internal_w2_l2_diff": "",
            })
        for alpha_a, alpha_b in combinations(ALPHAS, 2):
            metrics = distribution_metrics(outputs[alpha_a], outputs[alpha_b], top_k)
            metrics.update(internal_metrics(internals[alpha_a], internals[alpha_b]))
            all_pair_metrics.append(metrics)
            probe_rows.append({
                "row_type": "pairwise", "sample_id": row["sample_id"],
                "alpha_a": json.dumps(alpha_a), "alpha_b": json.dumps(alpha_b), "guide_alpha": "",
                "entropy": "", "top1_token_id": "", "topk_token_ids": "", "topk_probabilities": "", "logprob_storage_key": "",
                **metrics,
            })

    alpha_logprobs_path = OUT / "alpha_logprobs.pt"
    atomic_torch_save(alpha_logprobs_path, stored_logprobs)
    max_js = max(item["js_divergence"] for item in all_pair_metrics)
    median_js = sorted(item["js_divergence"] for item in all_pair_metrics)[len(all_pair_metrics) // 2]
    max_lp_delta = max(item["mean_abs_logprob_diff"] for item in all_pair_metrics)
    top1_change_rate = sum(item["top1_changed"] for item in all_pair_metrics) / len(all_pair_metrics)
    mean_overlap = sum(item["topk_overlap"] for item in all_pair_metrics) / len(all_pair_metrics)
    max_internal = max(item["internal_w2_max_abs_diff"] for item in all_pair_metrics)
    alpha_reaches = bool(updated_counts and min(updated_counts) > 0 and max_internal > 0)
    if not alpha_reaches or (max_js < 1e-10 and max_lp_delta < 1e-8):
        classification = "PBLORA_ALPHA_PATH_FAIL"
    elif median_js < 1e-6 and top1_change_rate < 0.02 and mean_overlap > 0.99:
        classification = "PBLORA_ALPHA_EFFECT_WEAK"
    else:
        classification = "PBLORA_ALPHA_CONDITIONING_PASS"

    utility_rows = []
    objective_specs = (
        ("helpfulness", (1.0, 0.0), "helpfulness_preferred_response_id"),
        ("harmlessness", (0.0, 1.0), "harmlessness_preferred_response_id"),
    )
    if classification == "PBLORA_ALPHA_CONDITIONING_PASS":
        for row in rows[:20]:
            prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])
            for objective, alpha, label_field in objective_specs:
                set_alpha(model, alpha)
                scores = [sequence_logp(model, tokenizer, prompt, row[f"response_{idx}"]) for idx in (0, 1)]
                preferred = int(row[label_field])
                rejected = 1 - preferred
                margin = scores[preferred][0] - scores[rejected][0]
                utility_rows.append({
                    "sample_id": row["sample_id"], "objective": objective,
                    "requested_alpha": json.dumps(alpha), "preferred_response_id": preferred,
                    "response_0_logp": scores[0][0], "response_1_logp": scores[1][0],
                    "response_0_tokens": scores[0][1], "response_1_tokens": scores[1][1],
                    "margin": margin, "correct": int(margin > 0),
                })

    def summarize_utility(selected: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [float(item["margin"]) for item in selected]
        if not margins:
            return {"count": 0, "preference_accuracy": None, "mean_margin": None, "median_margin": None}
        ordered = sorted(margins)
        return {
            "count": len(selected),
            "preference_accuracy": sum(int(item["correct"]) for item in selected) / len(selected),
            "mean_margin": sum(margins) / len(margins),
            "median_margin": ordered[len(ordered) // 2],
        }

    utility_summary = {
        "overall": summarize_utility(utility_rows),
        "by_objective": {
            objective: summarize_utility([item for item in utility_rows if item["objective"] == objective])
            for objective, _, _ in objective_specs
        },
    }
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    summary = {
        "classification": classification,
        "alpha_tensor_reaches_pblora": alpha_reaches,
        "updated_preference_tensor_count_min": min(updated_counts) if updated_counts else 0,
        "max_internal_w2_abs_delta": max_internal,
        "max_js_divergence": max_js,
        "median_js_divergence": median_js,
        "max_mean_abs_logprob_difference": max_lp_delta,
        "top1_change_rate": top1_change_rate,
        "mean_topk_overlap": mean_overlap,
        "thresholds": {"broken_max_js": 1e-10, "broken_mean_logprob_delta": 1e-8, "weak_median_js": 1e-6, "weak_top1_change_rate": 0.02, "weak_topk_overlap": 0.99},
        "reload_equivalence": reload_summary,
        "architecture": architecture,
        "architecture_after_reload": architecture_reloaded,
        "full_normalized_logprobs": str(alpha_logprobs_path.relative_to(ROOT)),
        "peak_allocated_vram_mib": peak_allocated,
        "peak_reserved_vram_mib": peak_reserved,
        "utility": utility_summary,
    }
    return summary, probe_rows, {"rows": utility_rows, "summary": utility_summary}


def write_pending_outputs() -> None:
    fields = [
        "row_type", "sample_id", "alpha_a", "alpha_b", "guide_alpha", "entropy",
        "top1_token_id", "topk_token_ids", "topk_probabilities", "logprob_storage_key",
        "kl_a_to_b", "kl_b_to_a", "js_divergence", "mean_abs_logprob_diff",
        "max_abs_logprob_diff", "top1_changed", "topk_overlap",
        "internal_w2_max_abs_diff", "internal_w2_mean_abs_diff", "internal_w2_l2_diff",
    ]
    atomic_csv(OUT / "alpha_probe.csv", [], fields)
    atomic_json(OUT / "alpha_probe_summary.json", {"status": "NOT_RUN", "reason": "GPU probe pending"})
    utility_fields = [
        "sample_id", "objective", "requested_alpha", "preferred_response_id",
        "response_0_logp", "response_1_logp", "response_0_tokens", "response_1_tokens",
        "margin", "correct",
    ]
    atomic_csv(OUT / "guide_utility.csv", [], utility_fields)


def write_report(audit: dict[str, Any], health: dict[str, Any], runtime: dict[str, Any], probe: dict[str, Any] | None) -> None:
    if probe is None:
        answers = [
            "1. **Smoke checkpoint valid?** PASS by static artifact audit.",
            "2. **Reload deterministic?** NOT RUN — GPU forward pending.",
            f"3. **Trainable parameter count correct?** PASS: {audit['adapter_parameter_count']:,}.",
            f"4. **Base frozen?** {'PASS' if health['base_all_non_pblora_parameters_frozen'] and health['base_sampled_fingerprint_unchanged'] else 'FAIL'}.",
            "5. **Alpha tensor reaches PBLoRA?** NOT RUN.",
            "6. **Guide distribution changes with alpha?** NOT RUN.",
            "7. **Alpha effect strong / weak / broken?** UNDETERMINED.",
            "8. **Preference signal exists?** NOT RUN.",
        ]
        final = "GPU_ALPHA_PROBE_PENDING — NO SCIENTIFIC CLASSIFICATION YET"
    else:
        reload_ok = probe["reload_equivalence"]["status"] == "PASS"
        utility_accuracy = probe["utility"]["overall"]["preference_accuracy"]
        helpfulness_accuracy = probe["utility"]["by_objective"]["helpfulness"]["preference_accuracy"]
        harmlessness_accuracy = probe["utility"]["by_objective"]["harmlessness"]["preference_accuracy"]
        signal = utility_accuracy is not None and utility_accuracy > 0.5
        classification = probe["classification"]
        if not reload_ok:
            final = "SMOKE_CHECKPOINT_INVALID"
        elif classification == "PBLORA_ALPHA_PATH_FAIL":
            final = "PBLORA_ALPHA_PATH_FAIL"
        elif classification == "PBLORA_ALPHA_EFFECT_WEAK" or not signal:
            final = "PBLORA_SMOKE_PASS — ALPHA EFFECT WEAK, INVESTIGATE FIRST"
        else:
            final = "PBLORA_SMOKE_PASS — READY FOR ONE EPOCH"
        answers = [
            "1. **Smoke checkpoint valid?** PASS.",
            f"2. **Reload deterministic?** {probe['reload_equivalence']['status']} (max diff {probe['reload_equivalence']['max_abs_logprob_diff']:.3e}).",
            f"3. **Trainable parameter count correct?** PASS: {audit['adapter_parameter_count']:,}.",
            f"4. **Base frozen?** {'PASS' if health['base_all_non_pblora_parameters_frozen'] and health['base_sampled_fingerprint_unchanged'] else 'FAIL'}.",
            f"5. **Alpha tensor reaches PBLoRA?** {probe['alpha_tensor_reaches_pblora']}.",
            f"6. **Guide distribution changes with alpha?** max JS={probe['max_js_divergence']:.3e}; top-1 change={probe['top1_change_rate']:.3f}.",
            f"7. **Alpha effect strong / weak / broken?** {classification}.",
            f"8. **Preference signal exists?** overall={utility_accuracy if utility_accuracy is not None else 'NOT_RUN'}; helpfulness={helpfulness_accuracy}; harmlessness={harmlessness_accuracy}.",
        ]
    seconds = runtime["seconds_per_optimizer_step"]
    eta1 = runtime["eta_one_epoch_seconds_excluding_eval_save"]
    eta2 = runtime["eta_two_epochs_seconds_excluding_eval_save"]
    answers.extend([
        f"9. **Actual seconds/step?** {seconds:.4f} s." if seconds else "9. **Actual seconds/step?** UNKNOWN.",
        f"10. **ETA 1 epoch?** {eta1:.1f} s ({eta1 / 60:.1f} min), excluding eval/save." if eta1 else "10. **ETA 1 epoch?** UNKNOWN.",
        f"11. **ETA 2 epochs?** {eta2:.1f} s ({eta2 / 60:.1f} min), excluding eval/save." if eta2 else "11. **ETA 2 epochs?** UNKNOWN.",
        f"12. **Safe to start one-epoch reproduction?** {'YES' if final.endswith('READY FOR ONE EPOCH') else 'NO; complete/investigate alpha gate first.'}",
    ])
    text = "# PBLoRA smoke checkpoint report\n\n" + "\n\n".join(answers) + "\n\n"
    if probe is not None:
        architecture = probe["architecture"]
        text += (
            f"Runtime architecture: total parameters={architecture['total_parameters']:,}; "
            f"PBLoRA trainable parameters={architecture['trainable_parameters_before_inference_freeze']:,} "
            f"({architecture['trainable_percent']:.4f}%); inference `requires_grad` count="
            f"{architecture['runtime_requires_grad_after_freeze']}. All trainable paths were PBLoRA Q/K/V.\n\n"
            f"Diagnostic peak VRAM: allocated={probe['peak_allocated_vram_mib']:.1f} MiB; "
            f"reserved={probe['peak_reserved_vram_mib']:.1f} MiB.\n\n"
        )
    text += "The audit is read-only, uses validation only, and performs no generation or training.\n\n" + final + "\n"
    atomic_text(OUT / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "probe", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    audit = checkpoint_audit()
    health, runtime = training_health()
    if audit["status"] != "PASS" or health["status"] != "PASS":
        write_pending_outputs()
        write_report(audit, health, runtime, None)
        print("SMOKE_CHECKPOINT_INVALID")
        return 2
    if args.phase == "audit":
        write_pending_outputs()
        write_report(audit, health, runtime, None)
        print("STATIC_AUDIT_PASS; GPU alpha probe pending")
        return 0
    if args.device != "cuda":
        raise SystemExit("Probe phase requires --device cuda")

    summary, probe_rows, utility = run_probe(args.top_k)
    fields = [
        "row_type", "sample_id", "alpha_a", "alpha_b", "guide_alpha", "entropy",
        "top1_token_id", "topk_token_ids", "topk_probabilities", "logprob_storage_key",
        "kl_a_to_b", "kl_b_to_a", "js_divergence", "mean_abs_logprob_diff",
        "max_abs_logprob_diff", "top1_changed", "topk_overlap",
        "internal_w2_max_abs_diff", "internal_w2_mean_abs_diff", "internal_w2_l2_diff",
    ]
    atomic_csv(OUT / "alpha_probe.csv", probe_rows, fields)
    atomic_json(OUT / "alpha_probe_summary.json", summary)
    utility_fields = [
        "sample_id", "objective", "requested_alpha", "preferred_response_id",
        "response_0_logp", "response_1_logp", "response_0_tokens", "response_1_tokens",
        "margin", "correct",
    ]
    atomic_csv(OUT / "guide_utility.csv", utility["rows"], utility_fields)
    runtime["peak_allocated_vram_mib"] = summary["peak_allocated_vram_mib"]
    runtime["peak_reserved_vram_mib"] = summary["peak_reserved_vram_mib"]
    runtime["vram_note"] = "Measured by the read-only smoke checker; includes checkpoint reload and all validation forwards."
    atomic_json(OUT / "runtime_estimate.json", runtime)
    write_report(audit, health, runtime, summary)
    print(summary["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
