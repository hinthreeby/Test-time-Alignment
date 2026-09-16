"""Low-VRAM teacher-forced diagnostic for PARM-TARO lambda feasibility.

This is deliberately not an alignment evaluation.  It loads one quantized
Tulu/PBLoRA model, caches frozen base/guide log-probabilities on CPU, releases
the model, and evaluates lambda-only NLL proxies on CPU.  It never generates
text and never loads Beaver reward/cost evaluators.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def find_project_root(source: Path) -> Path:
    """Resolve the checkout even when this file is reached through a symlink."""

    start = source.absolute().parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot locate project root from {source}")


PROJECT_ROOT = find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL = PROJECT_ROOT / "models/tulu-2-7b"
ADAPTER = (
    PROJECT_ROOT
    / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
)
MANIFEST = PROJECT_ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl"
VALIDATION = PROJECT_ROOT / "dataset/parm_taro/validation.json"
OUTPUT_DIR = PROJECT_ROOT / "results/parm_taro/recovery/low_vram_lambda"

ALPHAS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
)
LAMBDA_GRID: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0)
CACHE_SCHEMA_VERSION = 1
SEED = 42


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


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(torch: Any, path: Path, value: Any) -> None:
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


def get_gpu_status() -> dict[str, Any]:
    """Query GPU 0 without importing torch or requiring pynvml."""

    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip().splitlines()[0]
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        return {
            "gpu_name": values[0],
            "total_mib": int(values[1]),
            "free_mib": int(values[2]),
            "used_mib": int(values[3]),
            "utilization_pct": int(values[4]),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def torch_memory(torch: Any) -> dict[str, Any]:
    return {
        "torch_allocated_mib": float(torch.cuda.memory_allocated() / 1024**2),
        "torch_reserved_mib": float(torch.cuda.memory_reserved() / 1024**2),
        "torch_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024**2),
        "nvidia_smi": get_gpu_status(),
    }


def alpha_key(alpha: Sequence[float]) -> str:
    return f"{float(alpha[0]):.2f},{float(alpha[1]):.2f}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_cases() -> list[dict[str, Any]]:
    """Take the first four existing validation cases in each alpha bucket."""

    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Required validation manifest is absent: {MANIFEST}. Refusing to "
            "sample from Stage-10 or fabricate cases."
        )
    rows: list[dict[str, Any]] = []
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("alpha_order") != ["helpfulness", "harmlessness"]:
                raise ValueError(f"Invalid alpha order on manifest line {line_number}")
            rows.append(row)

    selected: list[dict[str, Any]] = []
    for expected in ALPHAS:
        bucket = [
            row
            for row in rows
            if tuple(float(x) for x in row.get("requested_alpha", [])) == expected
        ]
        if len(bucket) < 4:
            raise ValueError(
                f"Manifest has only {len(bucket)} cases for alpha {expected}; need 4"
            )
        selected.extend(bucket[:4])
    if len(selected) != 20 or len({row["case_id"] for row in selected}) != 20:
        raise ValueError("Expected exactly 20 distinct manifest case IDs")

    validation_rows = json.loads(VALIDATION.read_text(encoding="utf-8"))
    by_id = {str(row["sample_id"]): row for row in validation_rows}
    if len(by_id) != len(validation_rows):
        raise ValueError("Validation dataset contains duplicate sample IDs")
    enriched = []
    for row in selected:
        sample_id = str(row["sample_id"])
        data = by_id.get(sample_id)
        if data is None:
            raise KeyError(f"Manifest sample is absent from validation: {sample_id}")
        if str(data["prompt"]) != str(row["prompt"]):
            raise ValueError(f"Prompt mismatch for {sample_id}")
        prompt_hash = hashlib.sha256(str(data["prompt"]).encode("utf-8")).hexdigest()
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"Prompt SHA256 mismatch for {sample_id}")
        required = (
            "response_0",
            "response_1",
            "better_response_id",
            "safer_response_id",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Validation target fields missing for {sample_id}: {missing}")
        enriched.append({**row, "validation_row": data})
    return enriched


def validate_artifacts() -> dict[str, Any]:
    required = {
        "base_config": BASE_MODEL / "config.json",
        "tokenizer_config": BASE_MODEL / "tokenizer_config.json",
        "adapter_config": ADAPTER / "adapter_config.json",
        "manifest": MANIFEST,
        "validation": VALIDATION,
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty {name}: {path}")
    weights = [
        path
        for path in (ADAPTER / "adapter_model.safetensors", ADAPTER / "adapter_model.bin")
        if path.is_file() and path.stat().st_size > 0
    ]
    if len(weights) != 1:
        raise FileNotFoundError(
            f"Expected exactly one non-empty adapter weight file in {ADAPTER}; got {weights}"
        )
    config = json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
    expected = {
        "peft_type": "PBLORA",
        "obj_num": 2,
        "r1": 4,
        "r2": 4,
        "lora_alpha": 8.0,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj"}:
        mismatches["target_modules"] = {
            "expected": ["q_proj", "k_proj", "v_proj"],
            "actual": config.get("target_modules"),
        }
    if mismatches:
        raise ValueError(f"PBLoRA adapter config mismatch: {mismatches}")
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    if configured_base != BASE_MODEL.resolve():
        raise ValueError(
            f"Adapter base path {configured_base} does not match {BASE_MODEL.resolve()}"
        )
    return {
        "paths": {name: str(path) for name, path in required.items()},
        "adapter_weights": str(weights[0]),
        "adapter_weights_size": weights[0].stat().st_size,
        "adapter_weights_sha256": sha256_file(weights[0]),
        "adapter_config": config,
    }


def load_pblora_model(torch: Any) -> tuple[Any, Any, Any]:
    """Load exactly one physical 4-bit Tulu backbone with PBLoRA attached."""

    from PARM_TARO.adapters.parm_adapter import (
        activate_vendored_parm_runtime,
        freeze_for_inference,
    )
    from PARM_TARO.training.runtime import FrozenBaseModelView

    vendored = activate_vendored_parm_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Project tokenization requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    if not bool(getattr(backbone, "is_loaded_in_4bit", False)):
        raise RuntimeError("Tulu backbone did not load in required 4-bit mode")
    guide_model = vendored.peft.PeftModel.from_pretrained(
        backbone, ADAPTER, is_trainable=False
    )
    freeze_for_inference(guide_model)
    if any(parameter.requires_grad for parameter in guide_model.parameters()):
        raise RuntimeError("Model parameters were not fully frozen")
    base_view = FrozenBaseModelView(guide_model).eval()
    return guide_model, base_view, tokenizer


def _selected_logprobs(
    torch: Any,
    model: Any,
    tokenized: Any,
    *,
    device: Any,
) -> Any:
    """Run one frozen forward and immediately return selected CPU fp16 log-p."""

    with torch.inference_mode():
        output = model(
            input_ids=tokenized.input_ids.to(device),
            attention_mask=tokenized.attention_mask.to(device),
            position_ids=tokenized.position_ids.to(device),
            use_cache=False,
        )
        selected = output.logits[0].index_select(
            0, tokenized.logit_positions.to(output.logits.device)
        )
        logprobs = torch.log_softmax(selected.float(), dim=-1)
        if not bool(torch.isfinite(logprobs).all()):
            raise FloatingPointError("Model emitted non-finite selected log-probabilities")
        cpu = logprobs.detach().to(device="cpu", dtype=torch.float16)
        del output, selected, logprobs
    return cpu


def _tokenize_pair(tokenizer: Any, case: dict[str, Any], args: Any) -> tuple[Any, Any]:
    from PARM_TARO.training.data import MultiObjectiveExample, tokenize_response

    row = case["validation_row"]
    example = MultiObjectiveExample(
        sample_id=str(row["sample_id"]),
        source_index=int(row["source_index"]),
        prompt=str(row["prompt"]),
        responses=(str(row["response_0"]), str(row["response_1"])),
        better_response_id=int(row["better_response_id"]),
        safer_response_id=int(row["safer_response_id"]),
    )
    return tuple(
        tokenize_response(
            tokenizer,
            example,
            response_index,
            max_length=args.max_length,
            max_continuation_tokens=args.max_continuation_tokens,
            include_eos_target=True,
        )
        for response_index in (0, 1)
    )


def _response_weights(torch: Any, case: dict[str, Any]) -> Any:
    from PARM_TARO.training.alpha import response_objective_weights

    row = case["validation_row"]
    alpha = torch.tensor(case["requested_alpha"], dtype=torch.float32)
    return response_objective_weights(
        alpha,
        better_response_id=int(row["better_response_id"]),
        safer_response_id=int(row["safer_response_id"]),
    ).cpu()


def _cache_metadata(args: Any) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "base_model": str(BASE_MODEL.resolve()),
        "adapter": str(ADAPTER.resolve()),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256_file(MANIFEST),
        "validation_sha256": sha256_file(VALIDATION),
        "max_length": args.max_length,
        "max_continuation_tokens": args.max_continuation_tokens,
        "include_eos_target": True,
        "alpha_order": ["helpfulness", "harmlessness"],
        "objective": "alpha_weighted_response_mean_nll",
        "base_quantity": "normalized log-probability",
        "guide_quantity": "normalized log-probability",
        "cache_dtype": "float16",
    }


def _load_existing_cache(torch: Any, args: Any) -> dict[str, Any]:
    path = OUTPUT_DIR / "cached_cases.pt"
    expected = _cache_metadata(args)
    if not path.is_file():
        return {"metadata": expected, "cases": []}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("metadata") != expected:
        raise RuntimeError(
            "Existing cached_cases.pt metadata differs from this run. Move it aside "
            "rather than silently mixing model/data/tokenization settings."
        )
    if not isinstance(payload.get("cases"), list):
        raise ValueError("Existing cache has invalid cases payload")
    return payload


def build_case_cache(
    torch: Any,
    cases: Sequence[dict[str, Any]],
    guide_model: Any,
    base_view: Any,
    tokenizer: Any,
    args: Any,
) -> dict[str, Any]:
    """Cache base/guide distributions, resuming atomically after each case."""

    from PARM_TARO.adapters.parm_adapter import (
        named_preference_to_parm,
        set_parm_preference,
    )

    payload = _load_existing_cache(torch, args)
    completed = {str(item["case_id"]) for item in payload["cases"]}
    device = torch.device("cuda:0")
    # The same four samples occur under five alpha values. Base distributions
    # do not depend on alpha, so reuse them within this run without a second GPU
    # forward. Resumed cases also seed this transient lookup.
    base_lookup: dict[tuple[str, int], Any] = {}
    for cached in payload["cases"]:
        for response in cached["responses"]:
            base_lookup[(str(cached["sample_id"]), int(response["response_index"]))] = (
                response["base_logprobs"]
            )

    for index, case in enumerate(cases, 1):
        case_id = str(case["case_id"])
        if case_id in completed:
            print(f"[{index}/20] cached sample {case_id} (resume skip)", flush=True)
            continue
        tokenized_pair = _tokenize_pair(tokenizer, case, args)
        named_alpha = torch.tensor(case["requested_alpha"], dtype=torch.float32)
        actual_alpha = named_preference_to_parm(named_alpha)
        updated = set_parm_preference(guide_model, actual_alpha)
        if not updated:
            raise RuntimeError("No PBLoRA preference tensors were updated")
        cached_responses = []
        for response_index, tokenized in enumerate(tokenized_pair):
            lookup_key = (str(case["sample_id"]), response_index)
            base_logprobs = base_lookup.get(lookup_key)
            if base_logprobs is None:
                base_logprobs = _selected_logprobs(
                    torch, base_view, tokenized, device=device
                )
                base_lookup[lookup_key] = base_logprobs
            guide_logprobs = _selected_logprobs(
                torch, guide_model, tokenized, device=device
            )
            gold = tokenized.gold_token_ids.detach().cpu().long()
            if base_logprobs.shape != guide_logprobs.shape:
                raise ValueError(f"Base/guide shape mismatch for {case_id}/r{response_index}")
            if base_logprobs.shape[0] != gold.numel():
                raise ValueError(f"Target length mismatch for {case_id}/r{response_index}")
            cached_responses.append(
                {
                    "response_index": response_index,
                    "base_logprobs": base_logprobs,
                    "guide_logprobs": guide_logprobs,
                    "gold_token_ids": gold,
                    "num_target_tokens": int(gold.numel()),
                    "truncated": bool(tokenized.truncated),
                }
            )
            del guide_logprobs, gold
        row = case["validation_row"]
        payload["cases"].append(
            {
                "case_id": case_id,
                "sample_id": str(case["sample_id"]),
                "requested_alpha": [float(x) for x in case["requested_alpha"]],
                "actual_pblora_alpha": [float(x) for x in actual_alpha.tolist()],
                "response_weights": _response_weights(torch, case),
                "better_response_id": int(row["better_response_id"]),
                "safer_response_id": int(row["safer_response_id"]),
                "responses": cached_responses,
            }
        )
        atomic_torch_save(torch, OUTPUT_DIR / "cached_cases.pt", payload)
        completed.add(case_id)
        print(f"[{index}/20] cached sample {case_id}", flush=True)
        del tokenized_pair, cached_responses
        gc.collect()
    if len(payload["cases"]) != 20:
        raise RuntimeError(f"Cache contains {len(payload['cases'])} cases, expected 20")
    order = {str(case["case_id"]): index for index, case in enumerate(cases)}
    payload["cases"].sort(key=lambda item: order[str(item["case_id"])])
    atomic_torch_save(torch, OUTPUT_DIR / "cached_cases.pt", payload)
    return payload


def current_fusion(a: Any, b: Any, lambda_value: Any) -> Any:
    """Current project equation: log pi_base + lambda * log pi_guide."""

    return a + lambda_value * b


def normalized_fusion(a: Any, b: Any, lambda_value: Any) -> Any:
    """Scale-controlled diagnostic equation."""

    return (a + lambda_value * b) / (1.0 + lambda_value)


def continuation_nll(torch: Any, scores: Any, gold: Any) -> tuple[Any, Any, Any, Any]:
    logprobs = torch.log_softmax(scores, dim=-1)
    nll = -logprobs.gather(-1, gold.long().unsqueeze(-1)).squeeze(-1).mean()
    probabilities = logprobs.exp()
    entropy = -(probabilities * logprobs).sum(dim=-1).mean()
    max_probability = probabilities.max(dim=-1).values.mean()
    logit_std = scores.std(dim=-1, unbiased=False).mean()
    return nll, entropy, max_probability, logit_std


def lambda_gradient(
    torch: Any,
    case: dict[str, Any],
    lambda_value: float,
    fusion_name: str,
) -> dict[str, float]:
    lam = torch.tensor(float(lambda_value), dtype=torch.float32, requires_grad=True)
    weights = case["response_weights"].float()
    response_metrics = []
    for response in case["responses"]:
        a = response["base_logprobs"].float()
        b = response["guide_logprobs"].float()
        scores = (
            current_fusion(a, b, lam)
            if fusion_name == "current"
            else normalized_fusion(a, b, lam)
        )
        response_metrics.append(
            continuation_nll(torch, scores, response["gold_token_ids"])
        )
    nll = sum(weights[i] * response_metrics[i][0] for i in (0, 1))
    gradient = torch.autograd.grad(nll, lam, retain_graph=False)[0]
    entropy = sum(weights[i] * response_metrics[i][1] for i in (0, 1))
    max_probability = sum(weights[i] * response_metrics[i][2] for i in (0, 1))
    logit_std = sum(weights[i] * response_metrics[i][3] for i in (0, 1))
    return {
        "nll": float(nll.detach()),
        "gradient": float(gradient.detach()),
        "entropy": float(entropy.detach()),
        "max_probability": float(max_probability.detach()),
        "logit_std": float(logit_std.detach()),
    }


def verify_fusion_parity(torch: Any, cases: Sequence[dict[str, Any]]) -> dict[str, float]:
    author_logprob = author_probability = 0.0
    paper_logprob = paper_probability = 0.0
    with torch.no_grad():
        for case in cases:
            for response in case["responses"]:
                a = response["base_logprobs"].float()
                b = response["guide_logprobs"].float()
                norm = torch.log_softmax(normalized_fusion(a, b, 1.0), dim=-1)
                author = torch.log_softmax((a + b) / 2.0, dim=-1)
                current = torch.log_softmax(current_fusion(a, b, 1.0), dim=-1)
                paper = torch.log_softmax(a + b, dim=-1)
                author_logprob = max(author_logprob, float((norm - author).abs().max()))
                author_probability = max(
                    author_probability, float((norm.exp() - author.exp()).abs().max())
                )
                paper_logprob = max(paper_logprob, float((current - paper).abs().max()))
                paper_probability = max(
                    paper_probability, float((current.exp() - paper.exp()).abs().max())
                )
    return {
        "author_logprob_max_abs_diff": author_logprob,
        "author_probability_max_abs_diff": author_probability,
        "paper_logprob_max_abs_diff": paper_logprob,
        "paper_probability_max_abs_diff": paper_probability,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def compute_global_optimum(rows: Sequence[dict[str, Any]], fusion: str) -> dict[str, Any]:
    candidates = []
    for value in LAMBDA_GRID:
        subset = [row for row in rows if row["fusion"] == fusion and row["lambda"] == value]
        candidates.append({"lambda": value, "mean_nll": _mean(row["nll"] for row in subset)})
    return min(candidates, key=lambda item: (item["mean_nll"], item["lambda"]))


def compute_per_alpha_optima(
    rows: Sequence[dict[str, Any]], fusion: str
) -> list[dict[str, Any]]:
    output = []
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        candidates = []
        for value in LAMBDA_GRID:
            subset = [
                row
                for row in rows
                if row["fusion"] == fusion
                and row["alpha"] == key
                and row["lambda"] == value
            ]
            candidates.append(
                {"lambda": value, "mean_nll": _mean(row["nll"] for row in subset)}
            )
        best = min(candidates, key=lambda item: (item["mean_nll"], item["lambda"]))
        output.append({"fusion": fusion, "alpha": key, **best})
    return output


def compute_oracle_headroom(
    rows: Sequence[dict[str, Any]], fusion: str, global_best: dict[str, Any]
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fusion"] == fusion:
            by_case[str(row["case_id"])].append(row)
    selected = {
        case_id: min(items, key=lambda item: (item["nll"], item["lambda"]))
        for case_id, items in by_case.items()
    }
    oracle_nll = _mean(item["nll"] for item in selected.values())
    global_nll = float(global_best["mean_nll"])
    gain = global_nll - oracle_nll
    distribution = Counter(str(item["lambda"]) for item in selected.values())
    differs = _mean(
        float(item["lambda"] != global_best["lambda"]) for item in selected.values()
    )
    per_alpha = []
    for alpha in ALPHAS:
        key = alpha_key(alpha)
        alpha_items = [item for item in selected.values() if item["alpha"] == key]
        fixed_items = [
            row
            for row in rows
            if row["fusion"] == fusion
            and row["alpha"] == key
            and row["lambda"] == global_best["lambda"]
        ]
        fixed_nll = _mean(item["nll"] for item in fixed_items)
        alpha_oracle = _mean(item["nll"] for item in alpha_items)
        per_alpha.append(
            {
                "alpha": key,
                "global_fixed_nll": fixed_nll,
                "oracle_nll": alpha_oracle,
                "absolute_gain": fixed_nll - alpha_oracle,
            }
        )
    return {
        "best_global_lambda": global_best["lambda"],
        "best_global_nll": global_nll,
        "oracle_nll": oracle_nll,
        "oracle_absolute_gain": gain,
        "oracle_relative_gain_pct": 100.0 * gain / global_nll,
        "oracle_lambda_distribution": dict(sorted(distribution.items())),
        "fraction_oracle_differs_from_global": differs,
        "per_alpha": per_alpha,
    }


def _group_summary(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, items in sorted(groups.items(), key=lambda item: str(item[0])):
        gradients = [float(item["gradient"]) for item in items]
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "num_cases": len(items),
                "mean_nll": _mean(item["nll"] for item in items),
                "mean_gradient": _mean(gradients),
                "median_gradient": float(statistics.median(gradients)),
                "fraction_positive_gradient": _mean(float(x > 0.0) for x in gradients),
                "mean_entropy": _mean(item["entropy"] for item in items),
                "mean_max_probability": _mean(item["max_probability"] for item in items),
                "mean_logit_std": _mean(item["logit_std"] for item in items),
            }
        )
        output.append(row)
    return output


def _alpha_variation(per_alpha: Sequence[dict[str, Any]], fusion: str) -> dict[str, Any]:
    values = [float(row["lambda"]) for row in per_alpha if row["fusion"] == fusion]
    return {
        "distinct": len(set(values)),
        "std": float(statistics.pstdev(values)),
        "range": max(values) - min(values),
        "extremes_differ": values[0] != values[-1],
        "vary": len(set(values)) > 1,
    }


def run_cpu_sweep(torch: Any, payload: dict[str, Any]) -> dict[str, Any]:
    print("GPU cache phase complete", flush=True)
    print("CPU lambda sweep starting", flush=True)
    rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        for fusion in ("current", "normalized"):
            for value in LAMBDA_GRID:
                metrics = lambda_gradient(torch, case, value, fusion)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "sample_id": case["sample_id"],
                        "alpha": alpha_key(case["requested_alpha"]),
                        "fusion": fusion,
                        "lambda": value,
                        **metrics,
                    }
                )

    per_case_fields = (
        "case_id", "sample_id", "alpha", "fusion", "lambda", "nll",
        "gradient", "entropy", "max_probability", "logit_std",
    )
    atomic_csv(OUTPUT_DIR / "per_case_lambda_metrics.csv", rows, per_case_fields)

    gradient = _group_summary(rows, ("fusion", "lambda")) + _group_summary(
        rows, ("fusion", "lambda", "alpha")
    )
    gradient_fields = (
        "fusion", "lambda", "alpha", "num_cases", "mean_nll", "mean_gradient",
        "median_gradient", "fraction_positive_gradient", "mean_entropy",
        "mean_max_probability", "mean_logit_std",
    )
    # Global rows have no alpha key; normalize columns for DictWriter.
    for row in gradient:
        row.setdefault("alpha", "ALL")
    atomic_csv(OUTPUT_DIR / "gradient_summary.csv", gradient, gradient_fields)

    scale_rows = _group_summary(rows, ("fusion", "lambda"))
    scale_fields = (
        "fusion", "lambda", "num_cases", "mean_nll", "mean_gradient",
        "median_gradient", "fraction_positive_gradient", "mean_entropy",
        "mean_max_probability", "mean_logit_std",
    )
    atomic_csv(OUTPUT_DIR / "fusion_scale_summary.csv", scale_rows, scale_fields)

    globals_out = []
    per_alpha_out = []
    oracle: dict[str, Any] = {}
    for fusion in ("current", "normalized"):
        global_best = compute_global_optimum(rows, fusion)
        globals_out.extend(
            {
                "fusion": fusion,
                "lambda": value,
                "mean_nll": _mean(
                    row["nll"]
                    for row in rows
                    if row["fusion"] == fusion and row["lambda"] == value
                ),
                "is_best": value == global_best["lambda"],
            }
            for value in LAMBDA_GRID
        )
        per_alpha_out.extend(compute_per_alpha_optima(rows, fusion))
        oracle[fusion] = compute_oracle_headroom(rows, fusion, global_best)
    atomic_csv(
        OUTPUT_DIR / "global_lambda_summary.csv",
        globals_out,
        ("fusion", "lambda", "mean_nll", "is_best"),
    )
    atomic_csv(
        OUTPUT_DIR / "per_alpha_lambda_summary.csv",
        per_alpha_out,
        ("fusion", "alpha", "lambda", "mean_nll"),
    )
    atomic_json(OUTPUT_DIR / "oracle_summary.json", oracle)
    return {
        "rows": rows,
        "gradient": gradient,
        "scale": scale_rows,
        "global": globals_out,
        "per_alpha": per_alpha_out,
        "oracle": oracle,
        "parity": verify_fusion_parity(torch, payload["cases"]),
    }


def _gradient_at(results: dict[str, Any], fusion: str, value: float) -> dict[str, float]:
    row = next(
        item
        for item in results["gradient"]
        if item["fusion"] == fusion
        and item["lambda"] == value
        and item["alpha"] == "ALL"
    )
    return {
        "mean": row["mean_gradient"],
        "median": row["median_gradient"],
        "fraction_positive": row["fraction_positive_gradient"],
    }


def _choose_verdict(summary: dict[str, Any]) -> str:
    current = summary["current"]
    normalized = summary["normalized"]
    gradient = summary["gradient_at_0.01"]
    # Explicitly heuristic: this proxy can support continued investigation or
    # flag objective collapse, but can never establish a scientific NO-GO.
    if (
        gradient["current_fraction_positive"] >= 0.75
        and float(current["best_global_lambda"]) <= 0.01
    ):
        return "LOW-VRAM DIAGNOSTIC SUGGESTS LAMBDA COLLAPSE OBJECTIVE ISSUE"
    headroom = max(
        float(current["oracle_relative_gain_pct"]),
        float(normalized["oracle_relative_gain_pct"]),
    )
    varied = (
        summary["alpha_optima_vary_current"]
        or summary["alpha_optima_vary_normalized"]
        or max(
            float(current["fraction_oracle_differs_from_global"]),
            float(normalized["fraction_oracle_differs_from_global"]),
        )
        >= 0.5
    )
    if headroom >= 1.0 and varied:
        return "LOW-VRAM DIAGNOSTIC SUPPORTS ADAPTIVE LAMBDA"
    return "LOW-VRAM DIAGNOSTIC SHOWS WEAK ADAPTIVE HEADROOM"


def write_report(
    results: dict[str, Any], runtime: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    variations = {
        fusion: _alpha_variation(results["per_alpha"], fusion)
        for fusion in ("current", "normalized")
    }
    current = results["oracle"]["current"]
    normalized = results["oracle"]["normalized"]
    g_current = _gradient_at(results, "current", 0.01)
    g_normalized = _gradient_at(results, "normalized", 0.01)
    summary = {
        "status": "COMPLETE",
        "num_cases": len(payload["cases"]),
        "lambda_grid": list(LAMBDA_GRID),
        "current": current,
        "normalized": normalized,
        "gradient_at_0.01": {
            "current_mean": g_current["mean"],
            "current_median": g_current["median"],
            "current_fraction_positive": g_current["fraction_positive"],
            "normalized_mean": g_normalized["mean"],
            "normalized_median": g_normalized["median"],
            "normalized_fraction_positive": g_normalized["fraction_positive"],
        },
        "author_parity_max_abs_diff": results["parity"][
            "author_logprob_max_abs_diff"
        ],
        "paper_parity_max_abs_diff": results["parity"][
            "paper_logprob_max_abs_diff"
        ],
        "fusion_parity": results["parity"],
        "alpha_optima_vary_current": variations["current"]["vary"],
        "alpha_optima_vary_normalized": variations["normalized"]["vary"],
        "alpha_optima_variation": variations,
        "runtime": runtime,
        "proxy_objective": "alpha_weighted_response_mean_nll",
        "fusion_semantics": {
            "a": "log_softmax(base model logits)",
            "b": "log_softmax(PBLoRA guide logits)",
            "current": "log_softmax(a + lambda*b)",
            "normalized": "log_softmax((a + lambda*b)/(1+lambda))",
            "author": "log_softmax((a+b)/2)",
            "paper_beta_1": "log_softmax(a+b)",
        },
    }
    summary["preliminary_verdict"] = _choose_verdict(summary)
    atomic_json(OUTPUT_DIR / "diagnostic_summary.json", summary)

    alpha_lines = []
    for row in results["per_alpha"]:
        alpha_lines.append(
            f"| {row['fusion']} | {row['alpha']} | {row['lambda']} | "
            f"{row['mean_nll']:.6f} |"
        )
    report = f"""# Low-VRAM lambda diagnostic

## Facts

- Validation cases: {summary['num_cases']} (first four existing feasibility60 cases per alpha; no Stage-10 test samples).
- Proxy objective: alpha-weighted mean continuation NLL over both labelled responses, matching Stage-9 source semantics.
- Base quantity `a`: normalized base log-probability; guide quantity `b`: normalized PBLoRA log-probability.
- Current best global lambda: {current['best_global_lambda']} (NLL {current['best_global_nll']:.6f}).
- Normalized best global lambda: {normalized['best_global_lambda']} (NLL {normalized['best_global_nll']:.6f}).
- Current oracle: NLL {current['oracle_nll']:.6f}; absolute gain {current['oracle_absolute_gain']:.6f}; relative gain {current['oracle_relative_gain_pct']:.3f}%.
- Normalized oracle: NLL {normalized['oracle_nll']:.6f}; absolute gain {normalized['oracle_absolute_gain']:.6f}; relative gain {normalized['oracle_relative_gain_pct']:.3f}%.
- Current gradient at lambda=0.01: mean {g_current['mean']:.6g}, median {g_current['median']:.6g}, positive fraction {g_current['fraction_positive']:.3f}.
- Normalized gradient at lambda=0.01: mean {g_normalized['mean']:.6g}, median {g_normalized['median']:.6g}, positive fraction {g_normalized['fraction_positive']:.3f}.
- Author parity max absolute log-probability difference: {summary['author_parity_max_abs_diff']:.8g}.
- Paper parity max absolute log-probability difference: {summary['paper_parity_max_abs_diff']:.8g}.
- Peak process allocation during cache build: {runtime['vram']['after_cache_build']['torch_peak_allocated_mib']:.1f} MiB.

### NLL-proxy optimum by alpha

| Fusion | Alpha (help,safe) | Best lambda | Mean NLL |
|---|---:|---:|---:|
{chr(10).join(alpha_lines)}

## Interpretation

- Under gradient descent, a positive `dNLL/dlambda` pushes lambda downward. The full lambda/alpha breakdown is in `gradient_summary.csv`.
- Changes in entropy, maximum probability, and score standard deviation in `fusion_scale_summary.csv` quantify the scale/temperature confound; normalized fusion is only a diagnostic control.
- Oracle gain is a finite-grid, per-case upper bound under teacher forcing. Lambda diversity and per-alpha optima indicate proxy headroom, not a deployable routing result.
- Preliminary classification: **{summary['preliminary_verdict']}**.

## Limitations

- Only 20 validation cases.
- Teacher-forced prefixes and continuation NLL only.
- NLL is a cheap proxy, not final preference-alignment evidence.
- No autoregressive generation was performed.
- No Beaver reward/cost evaluator was loaded.
- This result alone must not be used for a final scientific GO/NO-GO decision.

{summary['preliminary_verdict']}
"""
    atomic_text(OUTPUT_DIR / "diagnostic_report.md", report)
    return summary


def write_blocked(reason: str, gpu: dict[str, Any] | None = None) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "LOW_VRAM_DIAGNOSTIC_BLOCKED",
        "reason": reason,
        "gpu_preflight": gpu,
        "model_loaded": False,
    }
    atomic_json(OUTPUT_DIR / "status.json", status)
    atomic_text(
        OUTPUT_DIR / "diagnostic_report.md",
        "# Low-VRAM lambda diagnostic\n\n"
        f"Blocked before model load: {reason}\n\n"
        "LOW-VRAM DIAGNOSTIC BLOCKED\n",
    )
    print("LOW_VRAM_DIAGNOSTIC_BLOCKED")
    print(reason)
    return 0


def run_synthetic_checks(torch: Any) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    a = torch.log_softmax(torch.randn(3, 17, generator=generator), dim=-1)
    b = torch.log_softmax(torch.randn(3, 17, generator=generator), dim=-1)
    gold = torch.tensor([1, 2, 3])
    lam = torch.tensor(0.25, requires_grad=True)
    nll, entropy, max_probability, score_std = continuation_nll(
        torch, current_fusion(a, b, lam), gold
    )
    gradient = torch.autograd.grad(nll, lam)[0]
    author = torch.log_softmax((a + b) / 2.0, dim=-1)
    normalized = torch.log_softmax(normalized_fusion(a, b, 1.0), dim=-1)
    paper = torch.log_softmax(a + b, dim=-1)
    current = torch.log_softmax(current_fusion(a, b, 1.0), dim=-1)
    values = (nll, entropy, max_probability, score_std, gradient)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("Synthetic diagnostic emitted non-finite values")
    if not torch.equal(author, normalized) or not torch.equal(paper, current):
        raise AssertionError("Synthetic fusion parity failed")
    return {
        "status": "PASS",
        "author_parity_max_abs_diff": float((author - normalized).abs().max()),
        "paper_parity_max_abs_diff": float((paper - current).abs().max()),
        "lambda_gradient_finite": True,
    }


def dry_run(args: Any) -> int:
    artifacts = validate_artifacts()
    cases = load_manifest_cases()
    package_versions = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Required package is absent: {name}") from error
    import torch

    # Exercise the exact local tokenizer, dual-response target construction,
    # alpha scalarization, and vendored PARM imports without loading any model
    # weights. This catches schema/import/boundary failures before scarce GPU
    # memory is reserved.
    from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime
    from transformers import AutoTokenizer

    vendored = activate_vendored_parm_runtime()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, use_fast=True
    )
    token_counts = []
    response_weights = []
    for case in cases:
        pair = _tokenize_pair(tokenizer, case, args)
        token_counts.extend(int(item.gold_token_ids.numel()) for item in pair)
        response_weights.append(_response_weights(torch, case).tolist())
    if not token_counts or min(token_counts) <= 0:
        raise ValueError("Dry-run produced an empty continuation target")

    synthetic = run_synthetic_checks(torch)
    counts = Counter(alpha_key(case["requested_alpha"]) for case in cases)
    result = {
        "status": "DRY_RUN_PASS",
        "model_loaded": False,
        "device_used": "cpu synthetic tensors only",
        "project_root": str(PROJECT_ROOT),
        "num_cases": len(cases),
        "cases_per_alpha": dict(sorted(counts.items())),
        "lambda_grid": list(LAMBDA_GRID),
        "artifacts": artifacts,
        "packages": package_versions,
        "source_imports": {
            "status": "PASS",
            "vendored_peft": vendored.origins["peft"],
            "vendored_model_arithmetic": vendored.origins["model_arithmetic"],
        },
        "target_preflight": {
            "num_dual_response_targets": len(token_counts),
            "min_target_tokens": min(token_counts),
            "max_target_tokens": max(token_counts),
            "all_response_weights_sum_to_one": all(
                math.isclose(sum(weights), 1.0, abs_tol=1e-6)
                for weights in response_weights
            ),
        },
        "synthetic_checks": synthetic,
        "planned_output_dir": str(OUTPUT_DIR),
        "max_length": args.max_length,
        "max_continuation_tokens": args.max_continuation_tokens,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=5500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-continuation-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.min_free_mib < 5500:
        parser.error("--min-free-mib may not be lower than the required 5500 MiB")
    if args.max_length <= 0 or args.max_continuation_tokens <= 0:
        parser.error("Token limits must be positive")
    return args


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.dry_run:
        return dry_run(args)

    validate_artifacts()
    cases = load_manifest_cases()
    gpu_before = get_gpu_status()
    import torch

    existing = _load_existing_cache(torch, args)
    expected_ids = {str(case["case_id"]) for case in cases}
    cached_ids = {str(case["case_id"]) for case in existing["cases"]}
    if not cached_ids <= expected_ids:
        raise RuntimeError(
            "Existing cache contains cases outside the selected validation manifest: "
            f"{sorted(cached_ids - expected_ids)}"
        )
    cache_complete = cached_ids == expected_ids and len(existing["cases"]) == 20
    if not cache_complete and "free_mib" not in gpu_before:
        return write_blocked(
            f"Cannot verify GPU 0 free VRAM with nvidia-smi: {gpu_before}", gpu_before
        )
    if (
        not cache_complete
        and int(gpu_before["free_mib"]) < args.min_free_mib
    ):
        return write_blocked(
            f"Need >= {args.min_free_mib} MiB free before model load; "
            f"observed {gpu_before['free_mib']} MiB",
            gpu_before,
        )

    if cache_complete:
        runtime = {
            "gpu_preflight": gpu_before,
            "vram": {
                "before_model_load": {"model_load_skipped": True},
                "after_cache_build": {
                    "torch_peak_allocated_mib": 0.0,
                    "model_load_skipped": True,
                },
                "after_model_delete": {"model_load_skipped": True},
            },
            "physical_model_copies": 0,
            "quantization": "cached CPU distributions; model load skipped",
            "batch_size": 1,
            "resumed_from_complete_cache": True,
        }
        payload = existing
        model_loaded = False
        print("[20/20] cached cases complete; GPU model load skipped", flush=True)
    else:
        if not torch.cuda.is_available():
            return write_blocked(
                "CUDA is unavailable to this process; refusing to load the 7B model",
                gpu_before,
            )
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        runtime = {
            "gpu_preflight": gpu_before,
            "vram": {"before_model_load": torch_memory(torch)},
            "physical_model_copies": 1,
            "quantization": "4-bit NF4, double quantization, float16 compute",
            "batch_size": 1,
            "resumed_from_complete_cache": False,
        }
        guide_model = base_view = tokenizer = None
        try:
            guide_model, base_view, tokenizer = load_pblora_model(torch)
            runtime["vram"]["after_model_load"] = torch_memory(torch)
            payload = build_case_cache(
                torch, cases, guide_model, base_view, tokenizer, args
            )
            runtime["vram"]["after_cache_build"] = torch_memory(torch)
        finally:
            del guide_model, base_view, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            runtime["vram"]["after_model_delete"] = torch_memory(torch)
        model_loaded = True

    results = run_cpu_sweep(torch, payload)
    summary = write_report(results, runtime, payload)
    status = {
        "status": "COMPLETE",
        "preliminary_verdict": summary["preliminary_verdict"],
        "num_cases": len(payload["cases"]),
        "model_loaded": model_loaded,
        "model_released_before_cpu_sweep": True,
        "no_generation": True,
        "no_evaluators": True,
        "runtime": runtime,
    }
    atomic_json(OUTPUT_DIR / "status.json", status)
    print(summary["preliminary_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
