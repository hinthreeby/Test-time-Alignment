"""Shared, fail-closed utilities for the low-VRAM recovery suite."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ALPHAS = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))
ALPHA_ORDER = ("helpfulness", "harmlessness")
SEED = 42


def resolve_project_root(source: Path | None = None) -> Path:
    start = (source or Path(__file__)).absolute().parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "PARM_TARO").exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot locate project root from {start}")


ROOT = resolve_project_root()
OUT = ROOT / "results/parm_taro/recovery/low_vram_suite"
BASE = ROOT / "models/tulu-2-7b"
ADAPTER = ROOT / "results/parm_taro/recovery/reproduced/pblora/final_checkpoint"
MANIFEST = ROOT / "results/parm_taro/recovery/feasibility60/manifest.jsonl"
VALIDATION = ROOT / "dataset/parm_taro/validation.json"
REWARD = ROOT / "models/beaver-7b-v1.0-reward"
COST = ROOT / "models/beaver-7b-v1.0-cost"
SAFE_RLHF = ROOT / "models/safe-rlhf-source"
HISTORICAL_SAFE_RLHF = Path(
    "/home/jupyter-iec2024se10/Reward Decoding/models/safe-rlhf-source"
)
HISTORICAL_SAFE_RLHF_TREE_SHA256 = (
    "1b61a3e47a4c9ad78184262132fad1b9edbb043bf1cdf593e66a87e02b985533"
)
PROTOCOL_LOCK = ROOT / "results/parm_taro/evaluation/protocol/protocol_lock.json"


def ensure_project_on_sys_path() -> Path:
    value = str(ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return ROOT


ensure_project_on_sys_path()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_gpu_status() -> dict[str, Any]:
    command = ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip().splitlines()[0]
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        return {"gpu_name": values[0], "total_mib": int(values[1]), "free_mib": int(values[2]), "used_mib": int(values[3]), "utilization_pct": int(values[4])}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def require_free_vram(min_free_mib: int = 5500) -> dict[str, Any]:
    status = get_gpu_status()
    if "free_mib" not in status:
        raise RuntimeError(f"Cannot verify GPU memory: {status}")
    if int(status["free_mib"]) < min_free_mib:
        raise RuntimeError(f"LOW_VRAM_BLOCKED: need >= {min_free_mib} MiB, observed {status['free_mib']} MiB")
    return status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(path: Path) -> str:
    """Hash scorer source files, excluding mutable Git/provenance metadata."""
    records = []
    source_root = path / "safe_rlhf"
    for item in sorted(source_root.rglob("*.py")):
        if item.is_file():
            relative = item.relative_to(path).as_posix()
            records.append(f"{relative}\t{item.stat().st_size}\t{sha256_file(item)}\n")
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return None


def resolve_safe_rlhf_source() -> dict[str, Any]:
    """Resolve only provenance-approved Safe-RLHF source trees."""
    canonical = SAFE_RLHF.resolve()
    historical = HISTORICAL_SAFE_RLHF.resolve()
    # A compatibility symlink from the old project root to the current root is
    # not evidence that the historical source snapshot survived.
    candidates: list[tuple[str, Path]] = []
    if historical != canonical:
        candidates.append(("historical_local_path", HISTORICAL_SAFE_RLHF))
    candidates.append(("canonical_recovery_path", SAFE_RLHF))
    configured = os.environ.get("SAFE_RLHF_SOURCE")
    if configured:
        candidates.append(("SAFE_RLHF_SOURCE", Path(configured).expanduser()))
    seen: set[Path] = set()
    checked = []
    required = (
        "safe_rlhf/models/score_model/__init__.py",
        "safe_rlhf/models/score_model/llama/modeling_llama.py",
        "safe_rlhf/models/normalizer.py",
    )
    for source, raw_path in candidates:
        path = raw_path.resolve()
        if path in seen:
            continue
        seen.add(path)
        missing = [name for name in required if not (path / name).is_file()]
        checked.append({"source": source, "path": str(path), "missing": missing})
        if not missing:
            commit = _git_commit(path)
            provenance_status = (
                "EXACT_COMMIT_RECOVERED"
                if source == "historical_local_path"
                else "UPSTREAM_COMPATIBLE_BUT_HISTORICAL_COMMIT_UNKNOWN"
            )
            return {
                "ready": True,
                "path": str(path),
                "source_of_path": source,
                "git_commit": commit,
                "tree_hash": sha256_source_tree(path),
                "historical_tree_hash": HISTORICAL_SAFE_RLHF_TREE_SHA256,
                "provenance_status": provenance_status,
                "checked": checked,
            }
    return {
        "ready": False,
        "path": None,
        "source_of_path": None,
        "git_commit": None,
        "tree_hash": None,
        "historical_tree_hash": HISTORICAL_SAFE_RLHF_TREE_SHA256,
        "provenance_status": "NO_PROVENANCE_FOUND",
        "checked": checked,
    }


def activate_safe_rlhf_score_runtime(source: Path) -> Any:
    """Import official scorer code without importing training-only DeepSpeed.

    Upstream ``safe_rlhf.__init__`` eagerly imports its complete trainer stack.
    Phase 07/09 need only the pinned upstream score-model implementation.  The
    namespace bootstrap bypasses unrelated trainers while executing upstream
    ``score_model`` and ``normalizer`` files unchanged.
    """
    source = source.resolve()
    package_root = source / "safe_rlhf"
    models_root = package_root / "models"
    loaded = sys.modules.get("safe_rlhf.models.score_model")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", "")).resolve()
        if source not in origin.parents:
            raise RuntimeError(f"safe_rlhf score runtime already loaded from {origin}")
        return loaded.AutoModelForScore

    safe_package = types.ModuleType("safe_rlhf")
    safe_package.__file__ = str(package_root / "__init__.py")
    safe_package.__path__ = [str(package_root)]
    safe_package.__package__ = "safe_rlhf"
    models_package = types.ModuleType("safe_rlhf.models")
    models_package.__file__ = str(models_root / "__init__.py")
    models_package.__path__ = [str(models_root)]
    models_package.__package__ = "safe_rlhf.models"
    sys.modules["safe_rlhf"] = safe_package
    sys.modules["safe_rlhf.models"] = models_package
    try:
        from safe_rlhf.models.score_model import AutoModelForScore, ScoreModelOutput
    except Exception:
        sys.modules.pop("safe_rlhf.models", None)
        sys.modules.pop("safe_rlhf", None)
        raise
    models_package.AutoModelForScore = AutoModelForScore
    models_package.ScoreModelOutput = ScoreModelOutput
    models_package.__all__ = ["AutoModelForScore", "ScoreModelOutput"]
    return AutoModelForScore


def atomic_json_dump(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields)); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle: os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def append_jsonl_atomic(path: Path, row: Mapping[str, Any]) -> None:
    """Append one fsynced JSON record; valid preceding records survive interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as error: raise ValueError(f"Invalid JSONL {path}:{number}") from error
    return rows


def atomic_jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))


def alpha_key(alpha: Sequence[float]) -> str:
    return f"{float(alpha[0]):.2f},{float(alpha[1]):.2f}"


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.is_file(): raise FileNotFoundError(MANIFEST)
    rows = read_jsonl(MANIFEST)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    by_id = {str(row["sample_id"]): row for row in validation}
    enriched = []
    for row in rows:
        data = by_id.get(str(row.get("sample_id")))
        if data is None: raise KeyError(f"Manifest sample absent from validation: {row.get('sample_id')}")
        if row.get("prompt") != data.get("prompt"): raise ValueError(f"Prompt mismatch: {row.get('sample_id')}")
        enriched.append({**row, "validation_row": data})
    return enriched


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    case_ids = [str(row["case_id"]) for row in rows]
    prompts = [str(row["sample_id"]) for row in rows]
    counts = Counter(alpha_key(row["requested_alpha"]) for row in rows)
    expected = {alpha_key(alpha): 12 for alpha in ALPHAS}
    prompt_alpha = Counter((str(row["sample_id"]), alpha_key(row["requested_alpha"])) for row in rows)
    checks = {
        "cases_60": len(rows) == 60,
        "case_ids_unique": len(set(case_ids)) == 60,
        "unique_prompts_12": len(set(prompts)) == 12,
        "five_alpha_12_each": dict(counts) == expected,
        "each_prompt_has_each_alpha_once": len(prompt_alpha) == 60 and all(value == 1 for value in prompt_alpha.values()),
        "alpha_order": all(row.get("alpha_order") == list(ALPHA_ORDER) for row in rows),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "num_cases": len(rows), "num_unique_prompts": len(set(prompts)), "alpha_counts": dict(sorted(counts.items()))}


def select_prompt_subset(rows: Sequence[dict[str, Any]], num_prompts: int) -> list[dict[str, Any]]:
    order = []
    for row in rows:
        if row["sample_id"] not in order: order.append(row["sample_id"])
    selected = set(order[:num_prompts])
    return [row for row in rows if row["sample_id"] in selected]


def load_tulu_tokenizer() -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    if not getattr(tokenizer, "is_fast", False): raise ValueError("A fast tokenizer is required")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tulu_pblora_4bit() -> tuple[Any, Any, Any]:
    """Return guide, adapter-disabled base view, tokenizer: one physical model."""
    import torch
    from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime, freeze_for_inference
    from PARM_TARO.training.runtime import FrozenBaseModelView
    vendored = activate_vendored_parm_runtime()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = load_tulu_tokenizer()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    backbone = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, quantization_config=quant, device_map={"": 0})
    if not bool(getattr(backbone, "is_loaded_in_4bit", False)): raise RuntimeError("Backbone is not 4-bit")
    guide = vendored.peft.PeftModel.from_pretrained(backbone, ADAPTER, is_trainable=False)
    freeze_for_inference(guide)
    if any(parameter.requires_grad for parameter in guide.parameters()): raise RuntimeError("Model parameters remain trainable")
    return guide, FrozenBaseModelView(guide).eval(), tokenizer


def set_requested_alpha(model: Any, alpha: Sequence[float]) -> list[str]:
    import torch
    from PARM_TARO.adapters.parm_adapter import named_preference_to_parm, set_parm_preference
    named = torch.tensor(alpha, dtype=torch.float32)
    return set_parm_preference(model, named_preference_to_parm(named))


def cuda_memory_snapshot() -> dict[str, Any]:
    import torch
    return {"allocated_mib": float(torch.cuda.memory_allocated()/1024**2), "reserved_mib": float(torch.cuda.memory_reserved()/1024**2), "peak_allocated_mib": float(torch.cuda.max_memory_allocated()/1024**2), "peak_reserved_mib": float(torch.cuda.max_memory_reserved()/1024**2), "nvidia_smi": get_gpu_status()}


def release_model(*objects: Any, expected_max_allocated_mib: float = 256.0) -> dict[str, Any]:
    """Drop caller-owned containers before calling; report allocator truthfully."""
    import torch
    # Deleting function arguments only releases these local references. Callers
    # must clear their own references first; the returned status catches leaks.
    objects = ()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    snapshot = cuda_memory_snapshot()
    snapshot["status"] = "PASS" if snapshot["allocated_mib"] <= expected_max_allocated_mib else "GPU_RELEASE_INCOMPLETE"
    return snapshot


def continuation_mask(prompt_length: int, total_length: int) -> list[bool]:
    if not 0 <= prompt_length <= total_length: raise ValueError("Invalid continuation boundary")
    return [False] * prompt_length + [True] * (total_length - prompt_length)


def sequence_mean_logprob(logprobs: Any, gold: Any) -> Any:
    return logprobs.gather(-1, gold.long().unsqueeze(-1)).squeeze(-1).mean()


def continuation_mean_nll(logprobs: Any, gold: Any) -> Any:
    return -sequence_mean_logprob(logprobs, gold)


def current_fusion(a: Any, b: Any, lam: Any) -> Any:
    import torch
    return torch.log_softmax(a + lam*b, dim=-1)


def normalized_fusion(a: Any, b: Any, lam: Any) -> Any:
    import torch
    return torch.log_softmax((a + lam*b)/(1.0+lam), dim=-1)


def author_fusion(a: Any, b: Any) -> Any:
    import torch
    return torch.log_softmax((a+b)/2.0, dim=-1)


def paper_beta1_fusion(a: Any, b: Any) -> Any:
    import torch
    return torch.log_softmax(a+b, dim=-1)


def tokenize_response_pair(tokenizer: Any, case: Mapping[str, Any], max_length: int = 512, max_continuation_tokens: int = 128) -> tuple[Any, Any]:
    from PARM_TARO.training.data import MultiObjectiveExample, tokenize_response
    row = case["validation_row"]
    example = MultiObjectiveExample(str(row["sample_id"]), int(row["source_index"]), str(row["prompt"]), (str(row["response_0"]), str(row["response_1"])), int(row["better_response_id"]), int(row["safer_response_id"]))
    return tuple(tokenize_response(tokenizer, example, idx, max_length=max_length, max_continuation_tokens=max_continuation_tokens, include_eos_target=True) for idx in (0, 1))


def selected_logprobs(model: Any, tokenized: Any) -> Any:
    """One inference forward, returning an owning float32 CPU clone only."""
    import torch
    with torch.inference_mode():
        output = model(input_ids=tokenized.input_ids.cuda(), attention_mask=tokenized.attention_mask.cuda(), position_ids=tokenized.position_ids.cuda(), use_cache=False)
        selected = output.logits[0].index_select(0, tokenized.logit_positions.to(output.logits.device))
        gpu_logprobs = torch.log_softmax(selected.float(), -1)
        cpu = gpu_logprobs.detach().float().cpu().clone()
        del output, selected, gpu_logprobs
    if not bool(torch.isfinite(cpu).all()): raise FloatingPointError("Non-finite model distribution")
    return cpu


def response_weights(case: Mapping[str, Any]) -> Any:
    import torch
    from PARM_TARO.training.alpha import response_objective_weights
    row = case["validation_row"]
    return response_objective_weights(torch.tensor(case["requested_alpha"], dtype=torch.float32), better_response_id=int(row["better_response_id"]), safer_response_id=int(row["safer_response_id"])).cpu()


def _next_logprobs(model: Any, input_ids: Any, attention: Any, position: Any, past: Any) -> tuple[Any, Any]:
    import torch
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention, position_ids=position, past_key_values=past, use_cache=True)
        logits = output.logits[:, -1].float()
        logp = torch.log_softmax(logits, -1)
        new_past = output.past_key_values
        del output, logits
    return logp, new_past


def generate_fixed(model: Any, base_view: Any, tokenizer: Any, prompt: str, alpha: Sequence[float], *, fusion: str, lambda_value: float | None, max_new_tokens: int, seed: int) -> dict[str, Any]:
    """Deterministic shared-backbone generation; no evaluator or second model."""
    import torch
    set_requested_alpha(model, alpha); set_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].cuda(); attention = encoded.get("attention_mask", torch.ones_like(input_ids)).cuda()
    position = attention.cumsum(-1)-1; base_input = guide_input = input_ids
    base_past = guide_past = None; ids=[]; base_scores=[]; disagreements=[]
    started=time.perf_counter()
    for _ in range(max_new_tokens):
        a, base_past = _next_logprobs(base_view, base_input, attention, position, base_past)
        b, guide_past = _next_logprobs(model, guide_input, attention, position, guide_past)
        disagreements.append(float((a.argmax(-1) != b.argmax(-1)).float().mean()))
        if fusion == "base": fused=a
        elif fusion == "guide_only": fused=b
        elif fusion == "current": fused=current_fusion(a,b,float(lambda_value))
        elif fusion == "normalized": fused=normalized_fusion(a,b,float(lambda_value))
        else: raise ValueError(f"Unknown fusion: {fusion}")
        token=fused.argmax(-1,keepdim=True); token_id=int(token.item()); ids.append(token_id); base_scores.append(float(a[0,token_id]))
        del a,b,fused
        if token_id == tokenizer.eos_token_id: break
        base_input=guide_input=token; attention=torch.cat((attention,torch.ones((1,1),dtype=attention.dtype,device=attention.device)),-1); position=torch.full((1,1),attention.shape[1]-1,dtype=torch.long,device=attention.device)
    elapsed=time.perf_counter()-started
    if not ids: raise RuntimeError("Empty generation")
    text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    result={"response":text,"token_ids":ids,"length":len(ids),"eos":ids[-1]==tokenizer.eos_token_id,"latency_seconds":elapsed,"base_conditional_perplexity":math.exp(-sum(base_scores)/len(base_scores)),"base_guide_top1_disagreement":sum(disagreements)/len(disagreements)}
    del input_ids,attention,position,base_input,guide_input,base_past,guide_past
    return result


def generation_key(row: Mapping[str, Any]) -> str:
    return "|".join((str(row["case_id"]), str(row["fusion"]), str(row.get("lambda")), str(row["seed"])))


def artifact_preflight() -> dict[str, Any]:
    resolution=resolve_safe_rlhf_source()
    resolved_safe=Path(resolution["path"]) if resolution["ready"] else SAFE_RLHF
    paths={"base":BASE,"adapter":ADAPTER,"manifest":MANIFEST,"validation":VALIDATION,"reward":REWARD,"cost":COST,"safe_rlhf":resolved_safe,"protocol_lock":PROTOCOL_LOCK}
    checks={name:path.exists() for name,path in paths.items()}
    checks["base_config"]=(BASE/"config.json").is_file(); checks["adapter_config"]=(ADAPTER/"adapter_config.json").is_file(); checks["adapter_weights"]=(ADAPTER/"adapter_model.safetensors").is_file()
    return {"paths":{name:str(path) for name,path in paths.items()},"checks":checks}


def scorer_path_preflight() -> dict[str, Any]:
    def model_check(path: Path) -> dict[str, Any]:
        index=path/"model.safetensors.index.json"; missing=[]
        if index.is_file():
            mapping=json.loads(index.read_text()).get("weight_map",{})
            for name in sorted(set(mapping.values())):
                shard=path/name
                if not shard.is_file() or shard.stat().st_size<=0: missing.append(name)
        else: missing.append("model.safetensors.index.json")
        config_path=path/"config.json"; config={}
        if config_path.is_file():
            try: config=json.loads(config_path.read_text(encoding="utf-8"))
            except Exception: config={}
        return {"path":str(path),"config":config_path.is_file(),"architecture":config.get("architectures"),"model_type":config.get("model_type"),"score_dim":config.get("score_dim"),"score_type":config.get("score_type"),"do_normalize":config.get("do_normalize"),"pad_token_id":config.get("pad_token_id"),"eos_token_id":config.get("eos_token_id"),"tokenizer":(path/"tokenizer_config.json").is_file(),"index":index.is_file(),"missing_shards":missing,"ready":config_path.is_file() and (path/"tokenizer_config.json").is_file() and index.is_file() and not missing}
    resolution=resolve_safe_rlhf_source(); import_error=None; symbol=None
    if resolution["ready"]:
        try:
            model_class=activate_safe_rlhf_score_runtime(Path(resolution["path"]))
            symbol=f"{model_class.__module__}.{model_class.__name__}"
        except Exception as error:
            import_error=f"{type(error).__name__}: {error}"
    safe=resolution["ready"] and import_error is None
    reward=model_check(REWARD); cost=model_check(COST)
    semantics={"reward":"higher is more helpful","cost":"higher is less safe / more harmful","harmlessness_transform":"harmlessness_raw = -cost_raw"}
    return {"reward":reward,"cost":cost,"safe_rlhf":{**resolution,"models_source":safe,"required_symbol":symbol,"import_error":import_error},"safe_rlhf_source_path":resolution["path"],"safe_rlhf_git_commit":resolution["git_commit"],"safe_rlhf_tree_hash":resolution["tree_hash"],"provenance_status":resolution["provenance_status"],"reward_model_path":str(REWARD),"cost_model_path":str(COST),"score_semantics":semantics,"ready":reward["ready"] and cost["ready"] and safe}


def score_with_one_beaver(model_path: Path, records: Sequence[Mapping[str, Any]], *, max_length: int=512) -> tuple[list[float],dict[str,Any]]:
    """Load, score, and release exactly one Beaver model."""
    import torch
    resolution=resolve_safe_rlhf_source()
    if not resolution["ready"]: raise RuntimeError(f"Safe-RLHF source unavailable: {resolution['checked']}")
    AutoModelForScore=activate_safe_rlhf_score_runtime(Path(resolution["path"]))
    from transformers import AutoTokenizer, BitsAndBytesConfig
    require_free_vram(5500); torch.cuda.reset_peak_memory_stats(); memory={"before":cuda_memory_snapshot()}
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForScore.from_pretrained(model_path,local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.float16,quantization_config=quant,device_map={"":0}).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    memory["loaded"]=cuda_memory_snapshot(); scores=[]
    with torch.inference_mode():
        for row in records:
            text=f"BEGINNING OF CONVERSATION: USER: {row['prompt']} ASSISTANT:{row['response']}"
            encoded=tokenizer(text,return_tensors="pt",truncation=True,max_length=max_length); encoded={k:v.cuda() for k,v in encoded.items()}; output=model(**encoded); end=output["end_scores"] if isinstance(output,dict) else output.end_scores; value=end.detach().float().reshape(-1)
            if value.numel()!=1 or not bool(torch.isfinite(value).all()): raise ValueError("Beaver emitted invalid score")
            scores.append(float(value.cpu()[0])); del encoded,output,end,value
    memory["peak"]=cuda_memory_snapshot(); del model,tokenizer; memory["after_release"]=release_model()
    return scores,memory


def dry_run_payload(script: str, **extra: Any) -> dict[str, Any]:
    rows=load_manifest(); manifest=validate_manifest(rows); artifacts=artifact_preflight()
    return {"status":"DRY_RUN_PASS","script":script,"model_loaded":False,"gpu_allocated":False,"manifest":manifest,"artifacts":artifacts,"alpha_order":list(ALPHA_ORDER),**extra}
