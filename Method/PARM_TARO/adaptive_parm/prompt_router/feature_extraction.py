"""Leakage-safe prompt-end scalar feature extraction with one frozen model."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


FEATURE_COLUMNS = (
    "alpha_helpfulness",
    "alpha_harmlessness",
    "prompt_token_length",
    "base_entropy",
    "base_top1_probability",
    "base_top1_top2_margin",
    "base_topk_mass",
    "parm_entropy",
    "parm_top1_probability",
    "parm_top1_top2_margin",
    "parm_topk_mass",
    "top1_disagreement",
    "topk_overlap",
    "js_divergence",
    "symmetric_kl",
    "logprob_correlation",
    "candidate_rank_disagreement",
)


def _root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate
    raise RuntimeError("Cannot locate project root")


ROOT = _root()
BASE = ROOT / "models/tulu-2-7b"
ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
MANIFEST = ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl"
OUTPUT = ROOT / "results/parm_taro/adaptive_parm/02_prompt_router"
STATE = OUTPUT / "feature_state"
FEATURES = OUTPUT / "prompt_features.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def gpu_status() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = [part.strip() for part in subprocess.check_output(command, text=True, timeout=15).splitlines()[0].split(",")]
        return {
            "name": values[0],
            "total_mib": int(values[1]),
            "free_mib": int(values[2]),
            "used_mib": int(values[3]),
            "utilization_pct": int(values[4]),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _rank_disagreement(a: Tensor, b: Tensor, top_k: int) -> float:
    a_ids = torch.topk(a, top_k).indices
    b_ids = torch.topk(b, top_k).indices
    candidates = torch.unique(torch.cat((a_ids, b_ids)))
    if candidates.numel() <= 1:
        return 0.0
    a_order = torch.argsort(a.index_select(0, candidates), descending=True)
    b_order = torch.argsort(b.index_select(0, candidates), descending=True)
    a_rank = torch.empty_like(a_order); a_rank[a_order] = torch.arange(a_order.numel())
    b_rank = torch.empty_like(b_order); b_rank[b_order] = torch.arange(b_order.numel())
    return float((a_rank.float() - b_rank.float()).abs().mean() / (candidates.numel() - 1))


def compute_prompt_features(
    base_logprobs: Tensor,
    parm_logprobs: Tensor,
    *,
    prompt_token_length: int,
    top_k: int = 10,
) -> dict[str, float]:
    """Compute scalar features from the distribution before generation."""
    if base_logprobs.ndim != 1 or parm_logprobs.shape != base_logprobs.shape:
        raise ValueError("Expected matching one-dimensional vocabulary log-probabilities")
    if top_k < 2 or top_k > base_logprobs.numel():
        raise ValueError("top_k must be between 2 and vocabulary size")
    a = torch.log_softmax(base_logprobs.detach().float().cpu(), dim=-1)
    b = torch.log_softmax(parm_logprobs.detach().float().cpu(), dim=-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Prompt distributions contain NaN or Inf")
    p, q = a.exp(), b.exp()
    a_top = torch.topk(p, top_k)
    b_top = torch.topk(q, top_k)
    log_midpoint = torch.logaddexp(a, b) - math.log(2.0)
    js = 0.5 * ((p * (a - log_midpoint)).sum() + (q * (b - log_midpoint)).sum())
    symmetric_kl = 0.5 * ((p * (a - b)).sum() + (q * (b - a)).sum())
    ac, bc = a - a.mean(), b - b.mean()
    denominator = ac.norm() * bc.norm()
    correlation = float((ac * bc).sum() / denominator) if float(denominator) > 0.0 else 0.0
    overlap = len(set(a_top.indices.tolist()) & set(b_top.indices.tolist())) / float(top_k)
    result = {
        "prompt_token_length": float(prompt_token_length),
        "base_entropy": float(-(p * a).sum()),
        "base_top1_probability": float(a_top.values[0]),
        "base_top1_top2_margin": float(a_top.values[0] - a_top.values[1]),
        "base_topk_mass": float(a_top.values.sum()),
        "parm_entropy": float(-(q * b).sum()),
        "parm_top1_probability": float(b_top.values[0]),
        "parm_top1_top2_margin": float(b_top.values[0] - b_top.values[1]),
        "parm_topk_mass": float(b_top.values.sum()),
        "top1_disagreement": float(a_top.indices[0] != b_top.indices[0]),
        "topk_overlap": overlap,
        "js_divergence": float(js),
        "symmetric_kl": float(symmetric_kl),
        "logprob_correlation": correlation,
        "candidate_rank_disagreement": _rank_disagreement(a, b, top_k),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite prompt feature")
    return result


def _load_manifest() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 60 or len({row["case_id"] for row in rows}) != 60:
        raise ValueError("Expected the frozen 60-case manifest")
    if any(row.get("alpha_order") != ["helpfulness", "harmlessness"] for row in rows):
        raise ValueError("Unexpected alpha order")
    return rows


def _load_model() -> tuple[Any, Any, Any]:
    from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime, freeze_for_inference
    from PARM_TARO.training.runtime import FrozenBaseModelView
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    vendored = activate_vendored_parm_runtime()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    guide = vendored.peft.PeftModel.from_pretrained(backbone, ADAPTER, is_trainable=False)
    freeze_for_inference(guide)
    if any(parameter.requires_grad for parameter in guide.parameters()):
        raise RuntimeError("PBLoRA parameters are not frozen")
    return guide.eval(), FrozenBaseModelView(guide).eval(), tokenizer


def _set_alpha(model: Any, alpha: list[float]) -> None:
    from PARM_TARO.adapters.parm_adapter import named_preference_to_parm, set_parm_preference

    named = torch.tensor(alpha, dtype=torch.float32)
    set_parm_preference(model, named_preference_to_parm(named))


def _next_logprobs(model: Any, encoded: Mapping[str, Tensor]) -> Tensor:
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).cuda()
    position_ids = attention_mask.cumsum(-1) - 1
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        result = torch.log_softmax(output.logits[0, -1].float(), dim=-1).cpu().clone()
    del input_ids, attention_mask, position_ids, output
    return result


def extract_features(*, resume: bool, min_free_mib: int = 6000, top_k: int = 10) -> dict[str, Any]:
    """Run prompt-only forwards; never score or generate a continuation."""
    rows = _load_manifest()
    expected_adapter_hash = sha256_file(ADAPTER / "adapter_model.safetensors")
    existing = {path.stem for path in STATE.glob("*.json")} if STATE.is_dir() else set()
    if existing and not resume:
        raise RuntimeError(f"Feature progress exists in {STATE}; use --resume")
    missing = [row for row in rows if row["case_id"] not in existing]
    before = gpu_status()
    if missing:
        if "free_mib" not in before or before["free_mib"] < min_free_mib:
            raise RuntimeError(f"GPU preflight requires {min_free_mib} MiB free: {before}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the selected Python environment")
        torch.cuda.reset_peak_memory_stats()
        model = base_view = tokenizer = None
        base_cache: dict[str, tuple[Tensor, int]] = {}
        try:
            model, base_view, tokenizer = _load_model()
            for index, row in enumerate(missing, 1):
                encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
                prompt_length = int(encoded["input_ids"].shape[-1])
                cached = base_cache.get(row["sample_id"])
                if cached is None:
                    base = _next_logprobs(base_view, encoded)
                    base_cache[row["sample_id"]] = (base, prompt_length)
                else:
                    base, cached_length = cached
                    if cached_length != prompt_length:
                        raise RuntimeError("Prompt token length changed across alpha")
                alpha = [float(value) for value in row["requested_alpha"]]
                _set_alpha(model, alpha)
                guide = _next_logprobs(model, encoded)
                features = compute_prompt_features(
                    base, guide, prompt_token_length=prompt_length, top_k=top_k
                )
                payload = {
                    "case_id": row["case_id"],
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "alpha_helpfulness": alpha[0],
                    "alpha_harmlessness": alpha[1],
                    **features,
                }
                atomic_json(STATE / f"{row['case_id']}.json", payload)
                del guide, encoded
                print(f"[{index}/{len(missing)}] extracted {row['case_id']}", flush=True)
        finally:
            base_cache.clear()
            if model is not None:
                del model, base_view, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    completed = []
    for row in rows:
        path = STATE / f"{row['case_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing completed feature case: {path}")
        completed.append(json.loads(path.read_text(encoding="utf-8")))
    fields = ["case_id", "sample_id", "prompt", *FEATURE_COLUMNS]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(completed)
    atomic_text(FEATURES, buffer.getvalue())
    after_hash = sha256_file(ADAPTER / "adapter_model.safetensors")
    if after_hash != expected_adapter_hash:
        raise RuntimeError("Frozen PBLoRA checkpoint hash changed")
    status = {
        "status": "PASS",
        "num_cases": len(completed),
        "num_unique_prompts": len({row["sample_id"] for row in completed}),
        "top_k": top_k,
        "feature_columns": list(FEATURE_COLUMNS),
        "prompt_format": "raw prompt, add_special_tokens=False (matches Phase 08 generation)",
        "time_of_feature": "after complete prompt and before first generated token",
        "pblora_frozen": True,
        "adapter_path": str(ADAPTER),
        "adapter_sha256_before": expected_adapter_hash,
        "adapter_sha256_after": after_hash,
        "model_copies": 1,
        "generation_performed": False,
        "scorer_features_used": False,
        "gpu_before": before,
        "gpu_after": gpu_status(),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None,
    }
    atomic_json(OUTPUT / "feature_extraction_status.json", status)
    return status
