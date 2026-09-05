#!/usr/bin/env python3
"""Stage 7 evaluation suite for RAD fixed-beta and learned-router decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.adaptive_router_decoding import (  # noqa: E402
    AdaptiveRADConfig,
    AdaptiveRADSample,
    assert_frozen_without_grad,
    load_router_checkpoint_strict,
    original_fixed_beta_rad_generate,
    adaptive_rad_generate,
    sha256_file,
)
from Method.RAD.reward_modeling.reward_model import GPT2RewardModel  # noqa: E402
from router.scripts.validate_models import configure_gpt2_padding, load_reward_model  # noqa: E402


DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "router" / "evaluation"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "rad_benchmark"
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_REWARD_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_REWARD_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_REWARD_CHECKPOINT_PATH = DEFAULT_REWARD_TOKENIZER_PATH / "pytorch_model.bin"
DEFAULT_ROUTER_CHECKPOINT_PATH = Path("/tmp/router_train_smoke/best.pt")
DEFAULT_SENTIMENT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "sentiment-roberta-large-english"
DEFAULT_FIXED_BETAS = [0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
ORIGINAL_FIXED_RAD_FILES = [
    PROJECT_ROOT / "Method" / "RAD" / "utils" / "logits_processor.py",
    PROJECT_ROOT / "Method" / "RAD" / "generate.py",
    PROJECT_ROOT / "Method" / "RAD" / "rad.py",
]


POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "brilliant", "calm", "delightful",
    "excellent", "fantastic", "friendly", "good", "great", "happy", "impressive",
    "love", "loved", "nice", "perfect", "pleasant", "positive", "relaxed",
    "smooth", "soft", "strong", "superb", "wonderful",
}
NEGATIVE_WORDS = {
    "awful", "bad", "boring", "broken", "cold", "disappointing", "flat",
    "hate", "hated", "horrible", "poor", "rough", "sad", "terrible", "toxic",
    "ugly", "uncooperative", "unhappy", "weak", "worst",
}


@dataclass(frozen=True)
class DecodingProtocol:
    top_k: int = 20
    max_new_tokens: int = 64
    max_reward_length: int = 256
    temperature: float = 1.0
    do_sample: bool = False
    inverse: bool = False
    beta_max: float = 30.0


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: str = str(DEFAULT_EVALUATION_DIR)
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    base_model_path: str = str(DEFAULT_BASE_MODEL_PATH)
    reward_base_path: str = str(DEFAULT_REWARD_BASE_PATH)
    reward_tokenizer_path: str = str(DEFAULT_REWARD_TOKENIZER_PATH)
    reward_checkpoint_path: str = str(DEFAULT_REWARD_CHECKPOINT_PATH)
    router_checkpoint_path: str = str(DEFAULT_ROUTER_CHECKPOINT_PATH)
    sentiment_classifier_path: str = str(DEFAULT_SENTIMENT_CLASSIFIER_PATH)
    sentiment_classifier_revision: str = "local"
    sentiment_classifier_max_length: int = 256
    sentiment_classifier_target_label: str = "POSITIVE"
    prompt_classes: list[str] = field(default_factory=lambda: ["negative", "neutral", "positive"])
    validation_prompt_limit_per_class: int = 4
    test_prompt_limit_per_class: int = 4
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    fixed_beta_sweep: list[float] = field(default_factory=lambda: list(DEFAULT_FIXED_BETAS))
    methods: list[str] = field(default_factory=list)
    decoding: DecodingProtocol = field(default_factory=DecodingProtocol)
    device: str = "auto"
    dtype: str = "auto"
    devices: dict[str, str] = field(default_factory=lambda: {
        "base_model": "auto",
        "reward_model": "auto",
        "router": "auto",
        "sentiment_classifier": "cpu",
    })
    fallback_to_cpu_on_oom: bool = False
    independent_sentiment_evaluator_id: str = "sentiment-roberta-large-english"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 12345


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    prompt_sentiment_class: str


@dataclass(frozen=True)
class EvaluationJob:
    prompt: PromptRecord
    method: str
    seed: int
    prompt_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.prompt.prompt_id, self.method, self.seed)


@dataclass(frozen=True)
class ProgressSettings:
    enabled: bool = True
    log_every_jobs: int = 10
    log_every_seconds: float = 60.0


@dataclass(frozen=True)
class ResolvedDevices:
    base_model: torch.device
    reward_model: torch.device
    router: torch.device
    sentiment_classifier: torch.device
    diagnostics: dict[str, dict[str, Any]]


def query_nvidia_smi(timeout_seconds: float = 2.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip() or f"nvidia-smi exited {result.returncode}"}
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    return {
        "available": True,
        "driver_version": parts[0] if len(parts) > 0 else None,
        "nvidia_smi_cuda_version": parts[1] if len(parts) > 1 else None,
        "gpu_name": parts[2] if len(parts) > 2 else None,
        "gpu_vram_gb": (float(parts[3]) / 1024.0) if len(parts) > 3 and parts[3] else None,
    }


def _cuda_probe(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    index = 0 if device.index is None else int(device.index)
    current = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(index)
    probe = torch.empty(1, device=device)
    probe += 1
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(index)
    return {
        "cuda_current_device": current,
        "gpu_name": gpu_name,
        "gpu_vram_gb": float(props.total_memory) / (1024.0 ** 3),
    }


def cuda_probe_failure_reason(device: torch.device | None = None) -> str | None:
    probe_device = device or torch.device("cuda:0")
    try:
        _cuda_probe(probe_device)
    except (RuntimeError, AssertionError, torch.cuda.OutOfMemoryError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def cuda_unavailable_message(requested_device: str, reason: str, smi: dict[str, Any]) -> str:
    return (
        "CUDA was explicitly requested but is unavailable.\n\n"
        f"Reason: {reason}\n"
        f"Current PyTorch build: {torch.__version__}\n"
        f"PyTorch CUDA runtime: {torch.version.cuda}\n"
        f"Detected driver supports CUDA {smi.get('nvidia_smi_cuda_version') or 'unknown'}\n\n"
        "Install a PyTorch build compatible with the current driver,\n"
        "or update the NVIDIA driver."
    )


def resolve_device(requested_device: str) -> torch.device:
    requested = str(requested_device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "cuda:0"}:
        raise ValueError(f"Unsupported device value {requested_device!r}; expected auto, cpu, cuda, or cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    cuda_device = torch.device("cuda:0")
    smi = query_nvidia_smi()
    reason = cuda_probe_failure_reason(cuda_device)
    if reason is None:
        return cuda_device
    if requested in {"cuda", "cuda:0"}:
        raise RuntimeError(cuda_unavailable_message(requested, reason, smi))
    print("[Device] CUDA unavailable; falling back to CPU.", flush=True)
    print(f"Reason: {reason}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"PyTorch CUDA build: {torch.version.cuda}", flush=True)
    print(f"NVIDIA driver-supported CUDA: {smi.get('nvidia_smi_cuda_version') or 'unknown'}", flush=True)
    return torch.device("cpu")


def collect_device_diagnostics(requested_device: str, resolved_device: torch.device | None = None, fallback_reason: str | None = None) -> dict[str, Any]:
    smi = query_nvidia_smi()
    requested = str(requested_device or "auto").lower()
    if requested == "cpu":
        cuda_available = False
        cuda_device_count = 0
    else:
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    effective_fallback_reason = fallback_reason
    if effective_fallback_reason is None and requested in {"auto", "cuda", "cuda:0"} and resolved_device is not None and resolved_device.type == "cpu":
        effective_fallback_reason = cuda_probe_failure_reason(torch.device("cuda:0"))
    diagnostics: dict[str, Any] = {
        "requested_device": requested_device,
        "resolved_device": str(resolved_device) if resolved_device is not None else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "gpu_name": smi.get("gpu_name"),
        "gpu_vram_gb": smi.get("gpu_vram_gb"),
        "driver_version": smi.get("driver_version"),
        "nvidia_smi_cuda_version": smi.get("nvidia_smi_cuda_version"),
        "fallback_reason": effective_fallback_reason,
    }
    if resolved_device is not None and resolved_device.type == "cuda":
        try:
            probe = _cuda_probe(resolved_device)
            diagnostics.update(probe)
        except (RuntimeError, AssertionError, torch.cuda.OutOfMemoryError) as error:
            diagnostics["fallback_reason"] = f"{type(error).__name__}: {error}"
    return diagnostics


def print_device_diagnostics(diagnostics: dict[str, Any]) -> None:
    print(f"[Device] Requested: {diagnostics.get('requested_device')}", flush=True)
    print(f"[Device] PyTorch: {diagnostics.get('torch_version')}", flush=True)
    print(f"[Device] PyTorch CUDA build: {diagnostics.get('torch_cuda_version')}", flush=True)
    print(f"[Device] NVIDIA driver: {diagnostics.get('driver_version') or 'unknown'}", flush=True)
    print(f"[Device] Driver CUDA capability: {diagnostics.get('nvidia_smi_cuda_version') or 'unknown'}", flush=True)
    if diagnostics.get("fallback_reason"):
        print("[Device] CUDA initialization failed", flush=True)
        print("[Device] Falling back to CPU", flush=True)
    print(f"[Device] Resolved: {diagnostics.get('resolved_device')}", flush=True)
    if diagnostics.get("gpu_name"):
        print(f"[Device] GPU: {diagnostics.get('gpu_name')}", flush=True)
    if diagnostics.get("gpu_vram_gb") is not None:
        print(f"[Device] VRAM: {float(diagnostics['gpu_vram_gb']):.2f} GB", flush=True)


def resolve_component_devices(config: EvaluationConfig) -> ResolvedDevices:
    requested = dict(config.devices or {})
    default = config.device
    component_defaults = {
        "base_model": requested.get("base_model", default),
        "reward_model": requested.get("reward_model", default),
        "router": requested.get("router", default),
        "sentiment_classifier": requested.get("sentiment_classifier", default),
    }
    resolved: dict[str, torch.device] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for component, requested_device in component_defaults.items():
        device = resolve_device(str(requested_device))
        resolved[component] = device
        diag = collect_device_diagnostics(str(requested_device), device)
        diagnostics[component] = diag
        print(f"[Device] {component} -> {device}", flush=True)
    return ResolvedDevices(
        base_model=resolved["base_model"],
        reward_model=resolved["reward_model"],
        router=resolved["router"],
        sentiment_classifier=resolved["sentiment_classifier"],
        diagnostics=diagnostics,
    )


def dtype_for_device(dtype_config: str, device: torch.device, component: str) -> torch.dtype:
    dtype_value = str(dtype_config or "auto").lower()
    if dtype_value == "auto":
        if component == "router":
            return torch.float32
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_value in {"float16", "fp16"}:
        return torch.float16
    if dtype_value in {"float32", "fp32"}:
        return torch.float32
    if dtype_value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype value {dtype_config!r}")


def cuda_oom_message(component: str, device: torch.device) -> str:
    allocated = reserved = 0.0
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024.0 ** 3)
    diag = collect_device_diagnostics(str(device), device)
    return (
        f"CUDA out of memory during {component}.\n"
        f"GPU: {diag.get('gpu_name') or 'unknown'}\n"
        f"Total VRAM: {diag.get('gpu_vram_gb') or 'unknown'}\n"
        f"Allocated: {allocated:.2f} GB\n"
        f"Reserved: {reserved:.2f} GB\n\n"
        "Suggested actions:\n"
        "- keep sentiment classifier on CPU\n"
        "- reduce batch size\n"
        "- use float16\n"
        "- reduce max_new_tokens only for debugging\n"
        "- resume after changing configuration"
    )


def cleanup_after_phase() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


@dataclass(frozen=True)
class ClassifierSentimentEvaluator:
    model: torch.nn.Module
    tokenizer: Any
    model_id: str
    revision: str
    label_mapping: dict[int, str]
    target_label: str
    target_label_id: int
    truncation: bool
    max_length: int
    device: torch.device

    def score(self, texts: Sequence[str]) -> list[dict[str, float | int | str]]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            logits = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            ).logits
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted = probabilities.argmax(dim=-1)
        rows = []
        for index in range(len(texts)):
            target_probability = float(probabilities[index, self.target_label_id].detach().cpu().item())
            predicted_id = int(predicted[index].detach().cpu().item())
            rows.append({
                "classifier_target_probability": target_probability,
                "classifier_sentiment_success": int(predicted_id == self.target_label_id),
                "classifier_predicted_label_id": predicted_id,
                "classifier_predicted_label": self.label_mapping[predicted_id],
            })
        return rows


def write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(EvaluationConfig()), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> EvaluationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding_payload = payload.get("decoding", {})
    payload["decoding"] = DecodingProtocol(**decoding_payload)
    return EvaluationConfig(**payload)


def validate_decoding_protocol(config: EvaluationConfig) -> None:
    protocol = config.decoding
    if protocol.top_k != 20:
        raise ValueError(f"top_k must be 20, got {protocol.top_k}")
    if protocol.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if protocol.max_reward_length != 256:
        raise ValueError(f"max_reward_length must be 256, got {protocol.max_reward_length}")
    if protocol.do_sample and protocol.temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    evaluator_id = config.independent_sentiment_evaluator_id.lower()
    if any(forbidden in evaluator_id for forbidden in ["rad_reward", "rad_rm", "router_training", "gpt2rewardmodel"]):
        raise ValueError("Independent sentiment evaluator must not be the RAD reward model")
    classifier_path = Path(config.sentiment_classifier_path)
    reward_checkpoint_path = Path(config.reward_checkpoint_path)
    if classifier_path.resolve() == reward_checkpoint_path.resolve():
        raise ValueError("Independent classifier path must not equal the RAD reward checkpoint")
    if classifier_path.is_file() and reward_checkpoint_path.is_file() and sha256_file(classifier_path) == sha256_file(reward_checkpoint_path):
        raise ValueError("Independent classifier file hash matches RAD reward checkpoint")
    classifier_checkpoint = classifier_path / "pytorch_model.bin"
    if classifier_checkpoint.exists() and reward_checkpoint_path.exists() and sha256_file(classifier_checkpoint) == sha256_file(reward_checkpoint_path):
        raise ValueError("Independent classifier checkpoint hash matches RAD reward checkpoint")
    router_checkpoint_path = Path(config.router_checkpoint_path)
    if router_checkpoint_path.exists():
        try:
            router_payload = torch.load(router_checkpoint_path, map_location="cpu", weights_only=False)
            router_evaluator = str(router_payload.get("config", {}).get("evaluator_id", ""))
            if router_evaluator and router_evaluator == config.independent_sentiment_evaluator_id:
                raise ValueError("Independent classifier matches router training evaluator")
        except ValueError:
            raise
        except Exception:
            pass
    classifier_config_path = classifier_path / "config.json"
    if classifier_config_path.exists():
        classifier_config = json.loads(classifier_config_path.read_text(encoding="utf-8"))
        architectures = [str(value).lower() for value in classifier_config.get("architectures", [])]
        if any("gpt2rewardmodel" in value for value in architectures):
            raise ValueError("Independent classifier architecture matches RAD reward architecture")
        label2id = classifier_config.get("label2id", {})
        if config.sentiment_classifier_target_label not in label2id:
            raise ValueError(f"Target label {config.sentiment_classifier_target_label!r} missing from classifier label mapping")


def read_jsonl_prompts(path: Path, prompt_class: str, limit: int | None, offset: int = 0) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_value = row.get("prompt", "")
            prompt = prompt_value.get("text", "") if isinstance(prompt_value, dict) else prompt_value
            prompt = str(prompt).strip()
            if not prompt:
                continue
            if seen < offset:
                seen += 1
                continue
            seen += 1
            prompt_id = str(row.get("md5_hash") or f"{prompt_class}:{line_number}")
            records.append(PromptRecord(prompt_id=prompt_id, prompt=prompt, prompt_sentiment_class=prompt_class))
            if limit is not None and len(records) >= limit:
                break
    return records


def load_prompt_split(config: EvaluationConfig, split: str) -> list[PromptRecord]:
    assert split in {"validation", "test"}
    limit = config.validation_prompt_limit_per_class if split == "validation" else config.test_prompt_limit_per_class
    offset = 0 if split == "validation" else int(config.validation_prompt_limit_per_class or 0)
    records: list[PromptRecord] = []
    for prompt_class in config.prompt_classes:
        path = Path(config.dataset_dir) / f"{prompt_class}_prompts.jsonl"
        records.extend(read_jsonl_prompts(path, prompt_class, limit, offset=offset))
    return records


def method_names(config: EvaluationConfig, include_sweep: bool = True) -> list[str]:
    if config.methods:
        return list(config.methods)
    names = ["base_lm"]
    if include_sweep:
        names.extend(f"fixed_beta_{format_beta(beta)}" for beta in config.fixed_beta_sweep)
    names.extend([
        "best_fixed_beta",
        "best_heuristic",
        "heuristic_constant_beta_10",
        "heuristic_linear_increase",
        "heuristic_linear_decrease",
        "heuristic_entropy_beta",
        "heuristic_reward_range_beta",
        "learned_router",
    ])
    return names


def heuristic_method_names() -> list[str]:
    return [
        "heuristic_constant_beta_10",
        "heuristic_linear_increase",
        "heuristic_linear_decrease",
        "heuristic_entropy_beta",
        "heuristic_reward_range_beta",
    ]


def format_beta(beta: float) -> str:
    return str(float(beta)).rstrip("0").rstrip(".").replace(".", "p")


def generation_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (str(record["prompt_id"]), str(record["method"]), int(record["seed"]))


def load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.add(generation_key(json.loads(line)))
    return keys


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl_record(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def total_evaluation_jobs(prompts: Sequence[PromptRecord], methods: Sequence[str], seeds: Sequence[int]) -> int:
    return len(prompts) * len(methods) * len(seeds)


def evaluation_jobs(prompts: Sequence[PromptRecord], methods: Sequence[str], seeds: Sequence[int]) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        for seed in seeds:
            for method in methods:
                jobs.append(EvaluationJob(prompt=prompt, method=method, seed=seed, prompt_index=prompt_index))
    return jobs


def pending_evaluation_jobs(jobs: Sequence[EvaluationJob], completed: set[tuple[str, str, int]]) -> list[EvaluationJob]:
    return [job for job in jobs if job.key not in completed]


def completed_jobs_in_plan(jobs: Sequence[EvaluationJob], completed: set[tuple[str, str, int]]) -> int:
    planned = {job.key for job in jobs}
    return len(planned & completed)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining_seconds(completed: int, total: int, elapsed: float) -> float:
    if completed <= 0 or total <= completed:
        return 0.0
    avg = max(elapsed, 0.0) / completed
    return max((total - completed) * avg, 0.0)


class EvaluationProgress:
    def __init__(
        self,
        description: str,
        total: int,
        initial: int,
        settings: ProgressSettings,
        output_path: Path | None = None,
    ) -> None:
        self.description = description
        self.total = int(total)
        self.completed = int(initial)
        self.successes = 0
        self.failures = 0
        self.settings = settings
        self.output_path = output_path
        self.start_time = time.perf_counter()
        self.last_summary_time = self.start_time
        self.last_summary_completed = self.completed
        self._pbar = None
        self._interactive = bool(settings.enabled and sys.stderr.isatty())
        if self._interactive:
            try:
                from tqdm.auto import tqdm
                self._pbar = tqdm(
                    total=self.total,
                    initial=self.completed,
                    desc=self.description,
                    unit="job",
                    dynamic_ncols=True,
                )
            except Exception:
                self._interactive = False
        if settings.enabled and not self._interactive:
            print(f"{self.description}: {self.completed}/{self.total} jobs", flush=True)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()

    def update(self, *, job: EvaluationJob | None, success: bool) -> None:
        self.completed += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        elapsed = time.perf_counter() - self.start_time
        avg = elapsed / max(self.completed, 1)
        eta = estimate_remaining_seconds(self.completed, self.total, elapsed)
        postfix = {
            "method": job.method if job else "",
            "prompt_class": job.prompt.prompt_sentiment_class if job else "",
            "seed": job.seed if job else "",
            "successes": self.successes,
            "failures": self.failures,
            "avg_seconds_per_job": f"{avg:.1f}",
            "ETA": format_duration(eta),
        }
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(postfix)
        elif self.settings.enabled:
            prompt_index = job.prompt_index if job else ""
            print(
                f"{self.description}: {self.completed}/{self.total} "
                f"[{(100.0 * self.completed / max(self.total, 1)):.1f}%] | "
                f"class={postfix['prompt_class']} | method={postfix['method']} | "
                f"seed={postfix['seed']} | prompt={prompt_index} | "
                f"elapsed={format_duration(elapsed)} | ETA={postfix['ETA']}",
                flush=True,
            )
        self.maybe_summary()

    def maybe_summary(self) -> None:
        if not self.settings.enabled:
            return
        now = time.perf_counter()
        jobs_delta = self.completed - self.last_summary_completed
        time_delta = now - self.last_summary_time
        if jobs_delta < self.settings.log_every_jobs and time_delta < self.settings.log_every_seconds:
            return
        self.last_summary_completed = self.completed
        self.last_summary_time = now
        elapsed = now - self.start_time
        avg = elapsed / max(self.completed, 1)
        remaining = estimate_remaining_seconds(self.completed, self.total, elapsed)
        print(
            "\n".join([
                f"Completed: {self.completed}/{self.total}",
                f"Successful: {self.successes}",
                f"Failed: {self.failures}",
                f"Elapsed: {format_duration(elapsed)}",
                f"Average/job: {avg:.1f}s",
                f"Estimated remaining: {format_duration(remaining)}",
                f"Output: {self.output_path}" if self.output_path else "Output: n/a",
            ]),
            flush=True,
        )


def phase_log(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def tokenize_words(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def independent_sentiment_metrics(text: str) -> dict[str, float | int]:
    tokens = tokenize_words(text)
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / max(pos + neg, 1)
    return {
        "lexicon_sentiment_score": float(score),
        "lexicon_sentiment_success": int(score > 0),
        "lexicon_positive_hits": int(pos),
        "lexicon_negative_hits": int(neg),
    }


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index:index + n]) for index in range(0, max(len(tokens) - n + 1, 0))]


def repeated_ngram_rate(tokens: Sequence[str], n: int) -> float:
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def maximum_repeated_phrase(tokens: Sequence[str], max_n: int = 5) -> str:
    best_phrase = ""
    best_count = 1
    for n in range(1, max_n + 1):
        counts: dict[tuple[str, ...], int] = {}
        for gram in ngrams(tokens, n):
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            if count > best_count or (count == best_count and len(gram) > len(best_phrase.split())):
                best_phrase = " ".join(gram)
                best_count = count
    return best_phrase


def distinct_n(texts: Sequence[str], n: int) -> float:
    all_grams: list[tuple[str, ...]] = []
    for text in texts:
        all_grams.extend(ngrams(tokenize_words(text), n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def lexical_perplexity_proxy(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return float("inf")
    avg_len = sum(len(token) for token in tokens) / len(tokens)
    repetition = repeated_ngram_rate(tokens, 2)
    return float(math.exp(min(6.0, 0.25 * avg_len + 2.0 * repetition)))


def coherence_proxy(prompt: str, generated_text: str) -> float:
    prompt_tokens = set(tokenize_words(prompt))
    generated_tokens = set(tokenize_words(generated_text))
    if not generated_tokens:
        return 0.0
    overlap = len(prompt_tokens & generated_tokens) / max(len(generated_tokens), 1)
    length_penalty = min(len(generated_tokens) / 8.0, 1.0)
    return float(0.5 * overlap + 0.5 * length_penalty)


def fluency_proxy(perplexity: float) -> float:
    if not math.isfinite(perplexity):
        return 0.0
    return float(1.0 / (1.0 + math.log1p(perplexity)))


def sample_to_metric_record(raw: dict[str, Any]) -> dict[str, Any]:
    text = str(raw.get("generated_text", ""))
    prompt = str(raw.get("prompt", ""))
    tokens = tokenize_words(text)
    sentiment = independent_sentiment_metrics(text)
    raw_perplexity = raw.get("base_lm_perplexity")
    perplexity = float(raw_perplexity) if raw_perplexity is not None else lexical_perplexity_proxy(text)
    if not math.isfinite(perplexity):
        perplexity = 0.0
    latency = dict(raw.get("latency", {}))
    generated_token_ids = list(raw.get("selected_token_ids", []))
    record = {
        **raw,
        **sentiment,
        "classifier_sentiment_success": int(raw.get("classifier_sentiment_success", 0)),
        "classifier_target_probability": float(raw.get("classifier_target_probability", 0.0)),
        "classifier_predicted_label": raw.get("classifier_predicted_label", ""),
        "classifier_predicted_label_id": int(raw.get("classifier_predicted_label_id", -1)),
        "target_sentiment_success": int(raw.get("classifier_sentiment_success", 0)),
        "independent_sentiment_score": float(raw.get("classifier_target_probability", 0.0)),
        "base_lm_perplexity": perplexity,
        "fluency": fluency_proxy(perplexity),
        "coherence": coherence_proxy(prompt, text),
        "repeated_unigram_rate": repeated_ngram_rate(tokens, 1),
        "repeated_bigram_rate": repeated_ngram_rate(tokens, 2),
        "repeated_trigram_rate": repeated_ngram_rate(tokens, 3),
        "maximum_repeated_phrase": maximum_repeated_phrase(tokens),
        "distinct_1": distinct_n([text], 1),
        "distinct_2": distinct_n([text], 2),
        "distinct_3": distinct_n([text], 3),
        "generation_length": len(generated_token_ids),
        "eos_rate": int(bool(raw.get("eos_generated", False))),
        "latency_per_token": float(latency.get("average_latency_per_token", 0.0)),
        "total_latency": float(latency.get("total_generation_time", 0.0)),
        "failure_status": str(raw.get("failure_status", "success")),
    }
    return record


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(mean(draw))
    means.sort()
    low = means[int((alpha / 2) * (len(means) - 1))]
    high = means[int((1 - alpha / 2) * (len(means) - 1))]
    return float(low), float(high)


def paired_rows(records: Sequence[dict[str, Any]], method_a: str, method_b: str, metric: str) -> list[tuple[float, float]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        key = (str(row["prompt_id"]), int(row["seed"]))
        by_key.setdefault(key, {})[str(row["method"])] = row
    pairs = []
    for methods in by_key.values():
        if method_a in methods and method_b in methods:
            a = methods[method_a].get(metric)
            b = methods[method_b].get(metric)
            if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)):
                pairs.append((float(a), float(b)))
    return pairs


def paired_bootstrap_test(
    pairs: Sequence[tuple[float, float]],
    seed: int,
    samples: int = 1000,
) -> dict[str, float]:
    if not pairs:
        return {"mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    diffs = [a - b for a, b in pairs]
    observed = mean(diffs)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(mean(draw))
    boot.sort()
    p = sum(abs(value) >= abs(observed) for value in boot) / len(boot)
    return {
        "mean_difference": float(observed),
        "ci_low": float(boot[int(0.025 * (len(boot) - 1))]),
        "ci_high": float(boot[int(0.975 * (len(boot) - 1))]),
        "p_value": float(p),
    }


def paired_permutation_test(
    pairs: Sequence[tuple[float, float]],
    seed: int,
    samples: int = 1000,
) -> float:
    if not pairs:
        return 1.0
    diffs = [a - b for a, b in pairs]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    count = 0
    for _ in range(samples):
        signed = [diff if rng.random() < 0.5 else -diff for diff in diffs]
        if abs(mean(signed)) >= observed:
            count += 1
    return float((count + 1) / (samples + 1))


def effect_size_cohens_d(pairs: Sequence[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    diffs = [a - b for a, b in pairs]
    if len(diffs) < 2:
        return 0.0
    std = statistics.pstdev(diffs)
    return float(mean(diffs) / std) if std > 0 else 0.0


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    indexed = sorted((float(p), i) for i, p in enumerate(p_values))
    adjusted = [1.0] * len(indexed)
    prev = 1.0
    m = len(indexed)
    for rank, (p_value, original_index) in reversed(list(enumerate(indexed, start=1))):
        value = min(prev, p_value * m / rank)
        adjusted[original_index] = float(min(value, 1.0))
        prev = value
    return adjusted


def aggregate_metrics(records: Sequence[dict[str, Any]], bootstrap_seed: int, bootstrap_samples: int) -> list[dict[str, Any]]:
    metrics = [
        "target_sentiment_success", "independent_sentiment_score",
        "classifier_sentiment_success", "classifier_target_probability",
        "lexicon_sentiment_score", "lexicon_sentiment_success",
        "base_lm_perplexity",
        "fluency", "coherence", "repeated_unigram_rate", "repeated_bigram_rate",
        "repeated_trigram_rate", "distinct_1", "distinct_2", "distinct_3",
        "generation_length", "eos_rate", "latency_per_token", "total_latency",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["prompt_sentiment_class"]), str(row["method"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (prompt_class, method), rows in sorted(groups.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            ci_low, ci_high = bootstrap_ci(values, seed=bootstrap_seed, samples=bootstrap_samples)
            output.append({
                "prompt_sentiment_class": prompt_class,
                "method": method,
                "metric": metric,
                "mean": mean(values) if values else float("nan"),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n": len(values),
                "failed_generations": sum(row.get("failure_status") != "success" for row in rows),
            })
    return output


def select_best_fixed_beta(validation_metrics: Sequence[dict[str, Any]], fixed_betas: Sequence[float]) -> float:
    allowed = {f"fixed_beta_{format_beta(beta)}": float(beta) for beta in fixed_betas}
    candidates = [
        row for row in validation_metrics
        if row.get("prompt_sentiment_class") == "negative"
        and row.get("metric") == "classifier_sentiment_success"
        and row.get("method") in allowed
    ]
    if not candidates:
        raise ValueError("No validation fixed-beta metrics available for selection")
    candidates.sort(key=lambda row: (float(row["mean"]), -float(row.get("latency_per_token", 0.0))), reverse=True)
    return allowed[str(candidates[0]["method"])]


def select_best_heuristic(validation_metrics: Sequence[dict[str, Any]]) -> str:
    allowed = set(heuristic_method_names())
    candidates = [
        row for row in validation_metrics
        if row.get("prompt_sentiment_class") == "negative"
        and row.get("metric") == "classifier_sentiment_success"
        and row.get("method") in allowed
    ]
    if not candidates:
        raise ValueError("No validation heuristic metrics available for selection")
    candidates.sort(key=lambda row: (float(row["mean"]), str(row["method"])), reverse=True)
    return str(candidates[0]["method"])


def router_behavior_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence_means: list[float] = []
    for record in records:
        if record.get("method") != "learned_router":
            continue
        betas = [float(value) for value in record.get("beta_history", [])]
        if not betas:
            continue
        sequence_means.append(mean(betas))
        rewards = record.get("rad_reward_scores_history", [])
        candidates = record.get("candidate_ids_history", [])
        guided = record.get("guided_scores_history", [])
        for index, beta in enumerate(betas):
            reward_values = rewards[index] if index < len(rewards) else []
            guided_values = guided[index] if index < len(guided) else []
            entropy = entropy_from_scores(guided_values)
            reward_range = max(reward_values) - min(reward_values) if reward_values else float("nan")
            selected_reward = float("nan")
            selected = record.get("selected_token_ids", [None] * len(betas))[index]
            if index < len(candidates) and selected in candidates[index]:
                selected_reward = float(reward_values[candidates[index].index(selected)])
            rows.append({
                "prompt_id": record["prompt_id"],
                "prompt_sentiment_class": record["prompt_sentiment_class"],
                "seed": record["seed"],
                "position": index,
                "beta": beta,
                "near_zero": int(beta < 0.05 * float(record.get("beta_max", 30.0))),
                "near_beta_max": int(beta > 0.95 * float(record.get("beta_max", 30.0))),
                "base_entropy": entropy,
                "reward_range": reward_range,
                "selected_token_reward": selected_reward,
            })
    between = statistics.pvariance(sequence_means) if len(sequence_means) > 1 else 0.0
    for row in rows:
        row["between_sequence_beta_variance"] = between
    return rows


def entropy_from_scores(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float)
    probs = torch.softmax(tensor, dim=-1)
    return float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())


def statistical_tests(records: Sequence[dict[str, Any]], baseline_methods: Sequence[str], seed: int, samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p_values = []
    for baseline in baseline_methods:
        for metric in [
            "classifier_sentiment_success",
            "classifier_target_probability",
            "lexicon_sentiment_score",
            "base_lm_perplexity",
            "coherence",
            "latency_per_token",
        ]:
            pairs = paired_rows(records, "learned_router", baseline, metric)
            boot = paired_bootstrap_test(pairs, seed=seed, samples=samples)
            perm_p = paired_permutation_test(pairs, seed=seed + 1, samples=samples)
            row = {
                "comparison": f"learned_router_vs_{baseline}",
                "metric": metric,
                "paired_n": len(pairs),
                **boot,
                "permutation_p_value": perm_p,
                "cohens_d_paired": effect_size_cohens_d(pairs),
            }
            rows.append(row)
            p_values.append(row["p_value"] if math.isfinite(float(row["p_value"])) else 1.0)
    adjusted = benjamini_hochberg(p_values)
    for row, adj in zip(rows, adjusted):
        row["p_value_bh_adjusted"] = adj
    return rows


def pareto_points(aggregate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, float]] = {}
    for row in aggregate_rows:
        if row["prompt_sentiment_class"] != "negative":
            continue
        by_method.setdefault(row["method"], {})[row["metric"]] = float(row["mean"])
    points = []
    for method, values in by_method.items():
        points.append({
            "method": method,
            "sentiment_success": values.get("target_sentiment_success", float("nan")),
            "perplexity": values.get("base_lm_perplexity", float("nan")),
            "coherence": values.get("coherence", float("nan")),
            "repetition": values.get("repeated_bigram_rate", float("nan")),
            "latency": values.get("latency_per_token", float("nan")),
        })
    for x_metric, lower_is_better in [("perplexity", True), ("coherence", False), ("repetition", True), ("latency", True)]:
        for point in points:
            point[f"pareto_frontier_success_vs_{x_metric}"] = int(is_pareto_frontier(point, points, x_metric, lower_is_better))
    return points


def is_pareto_frontier(point: dict[str, Any], points: Sequence[dict[str, Any]], x_metric: str, lower_is_better: bool) -> bool:
    y = float(point["sentiment_success"])
    x = float(point[x_metric])
    if not math.isfinite(y) or not math.isfinite(x):
        return False
    for other in points:
        if other is point:
            continue
        oy = float(other["sentiment_success"])
        ox = float(other[x_metric])
        if not math.isfinite(oy) or not math.isfinite(ox):
            continue
        x_better_or_equal = ox <= x if lower_is_better else ox >= x
        x_strict = ox < x if lower_is_better else ox > x
        if oy >= y and x_better_or_equal and (oy > y or x_strict):
            return False
    return True


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_no_nan_inf_rows(rows: Sequence[dict[str, Any]], context: str) -> None:
    for index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{context} row {index} field {key} is not finite: {value}")


def write_pareto_plots(points: Sequence[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        for x_metric in ["perplexity", "coherence", "repetition", "latency"]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for point in points:
                ax.scatter(point[x_metric], point["sentiment_success"])
                ax.annotate(point["method"], (point[x_metric], point["sentiment_success"]), fontsize=7)
            ax.set_xlabel(x_metric)
            ax.set_ylabel("target sentiment success")
            fig.tight_layout()
            fig.savefig(plots_dir / f"sentiment_success_vs_{x_metric}.png", dpi=160)
            plt.close(fig)
    except Exception:
        for x_metric in ["perplexity", "coherence", "repetition", "latency"]:
            (plots_dir / f"sentiment_success_vs_{x_metric}.svg").write_text(
                f"<svg xmlns='http://www.w3.org/2000/svg' width='400' height='260'>"
                f"<text x='20' y='30'>sentiment success vs {x_metric}</text></svg>\n",
                encoding="utf-8",
            )


def git_status() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    packages = ["torch", "transformers", "evaluate", "datasets", "numpy", "pandas", "matplotlib"]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def file_hash_or_none(path: str | Path) -> str | None:
    path = Path(path)
    return sha256_file(path) if path.exists() and path.is_file() else None


def classifier_label_mapping_from_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "id2label": payload.get("id2label", {}),
        "label2id": payload.get("label2id", {}),
        "architectures": payload.get("architectures", []),
        "model_type": payload.get("model_type"),
    }


def dataset_hashes(config: EvaluationConfig) -> dict[str, str]:
    hashes = {}
    for prompt_class in config.prompt_classes:
        path = Path(config.dataset_dir) / f"{prompt_class}_prompts.jsonl"
        hashes[str(path)] = sha256_file(path)
    return hashes


def baseline_source_hashes() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in ORIGINAL_FIXED_RAD_FILES if path.exists()}


def build_raw_record(
    prompt: PromptRecord,
    method: str,
    seed: int,
    sample: AdaptiveRADSample,
    beta_max: float,
    base_lm_perplexity: float | None = None,
) -> dict[str, Any]:
    eos_token_ids = sample.selected_token_ids
    return {
        "prompt_id": prompt.prompt_id,
        "prompt": prompt.prompt,
        "prompt_sentiment_class": prompt.prompt_sentiment_class,
        "method": method,
        "seed": int(seed),
        "generated_text": sample.text,
        "selected_token_ids": sample.selected_token_ids,
        "beta_history": sample.beta_history,
        "gate_history": sample.gate_history,
        "candidate_ids_history": sample.candidate_ids_history,
        "base_logits_history": sample.base_logits_history,
        "rad_reward_scores_history": sample.rad_reward_scores_history,
        "guided_scores_history": sample.guided_scores_history,
        "length": len(sample.selected_token_ids),
        "latency": sample.latency,
        "failure_status": "success",
        "error": None,
        "eos_generated": False,
        "beta_max": beta_max,
        "base_lm_perplexity": base_lm_perplexity,
    }


def attach_classifier_metrics(record: dict[str, Any], evaluator: ClassifierSentimentEvaluator) -> dict[str, Any]:
    scored = evaluator.score([str(record.get("generated_text", ""))])[0]
    return {**record, **scored}


class ConstantRouter(torch.nn.Module):
    def __init__(self, beta: float, beta_max: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.beta = float(beta)
        self.beta_max = float(beta_max)

    def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.full((base_logits.shape[0], 1), self.beta, dtype=base_logits.dtype, device=base_logits.device)
        gate = torch.full((base_logits.shape[0], 1), self.beta / self.beta_max, dtype=base_logits.dtype, device=base_logits.device)
        return beta, gate


class HeuristicRouter(torch.nn.Module):
    def __init__(self, name: str, beta_max: float, max_new_tokens: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.name = name
        self.beta_max = float(beta_max)
        self.max_new_tokens = max_new_tokens
        self.step = 0

    def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.name == "linear_increase":
            value = self.beta_max * min(self.step / max(self.max_new_tokens - 1, 1), 1.0)
            beta = torch.full((base_logits.shape[0], 1), value, dtype=base_logits.dtype, device=base_logits.device)
        elif self.name == "linear_decrease":
            value = self.beta_max * (1.0 - min(self.step / max(self.max_new_tokens - 1, 1), 1.0))
            beta = torch.full((base_logits.shape[0], 1), value, dtype=base_logits.dtype, device=base_logits.device)
        elif self.name == "entropy":
            probs = torch.softmax(base_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1, keepdim=True)
            beta = self.beta_max * (1.0 - entropy / math.log(base_logits.shape[-1])).clamp(0, 1)
        elif self.name == "reward_range":
            reward_range = reward_scores.max(dim=-1, keepdim=True).values - reward_scores.min(dim=-1, keepdim=True).values
            beta = self.beta_max * reward_range.clamp(0, 1)
        else:
            raise ValueError(f"Unknown heuristic router {self.name}")
        self.step += 1
        gate = (beta / self.beta_max).clamp(0, 1)
        return beta, gate


def load_models(config: EvaluationConfig, devices: ResolvedDevices):
    effective_devices = devices
    base_dtype = dtype_for_device(config.dtype, devices.base_model, "base_model")
    reward_dtype = dtype_for_device(config.dtype, devices.reward_model, "reward_model")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, local_files_only=True)
    configure_gpt2_padding(tokenizer, padding_side="left")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            local_files_only=True,
            torch_dtype=base_dtype,
        ).eval().to(devices.base_model)
    except torch.cuda.OutOfMemoryError as error:
        if config.fallback_to_cpu_on_oom:
            effective_devices = ResolvedDevices(
                base_model=torch.device("cpu"),
                reward_model=effective_devices.reward_model,
                router=effective_devices.router,
                sentiment_classifier=effective_devices.sentiment_classifier,
                diagnostics=effective_devices.diagnostics,
            )
            base_dtype = torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                local_files_only=True,
                torch_dtype=base_dtype,
            ).eval().to(effective_devices.base_model)
        else:
            raise RuntimeError(cuda_oom_message("base model load", devices.base_model)) from error
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    try:
        reward_model, reward_tokenizer, _ = load_reward_model(
            Path(config.reward_base_path),
            Path(config.reward_tokenizer_path),
            Path(config.reward_checkpoint_path),
            effective_devices.reward_model,
        )
        if reward_dtype == torch.float16:
            reward_model.half()
        elif reward_dtype == torch.bfloat16:
            reward_model.to(dtype=torch.bfloat16)
    except torch.cuda.OutOfMemoryError as error:
        if config.fallback_to_cpu_on_oom:
            reward_model, reward_tokenizer, _ = load_reward_model(
                Path(config.reward_base_path),
                Path(config.reward_tokenizer_path),
                Path(config.reward_checkpoint_path),
                torch.device("cpu"),
            )
            reward_dtype = torch.float32
            effective_devices = ResolvedDevices(
                base_model=effective_devices.base_model,
                reward_model=torch.device("cpu"),
                router=effective_devices.router,
                sentiment_classifier=effective_devices.sentiment_classifier,
                diagnostics=effective_devices.diagnostics,
            )
        else:
            raise RuntimeError(cuda_oom_message("reward model load", effective_devices.reward_model)) from error
    configure_gpt2_padding(reward_tokenizer, padding_side="left")
    reward_model.eval().to(effective_devices.reward_model)
    for parameter in reward_model.parameters():
        parameter.requires_grad = False
    assert_frozen_without_grad(base_model, "base LM")
    assert_frozen_without_grad(reward_model, "reward model")
    print(f"[Dtype] Base model: {str(base_dtype).replace('torch.', '')}", flush=True)
    print(f"[Dtype] Reward model: {str(reward_dtype).replace('torch.', '')}", flush=True)
    return base_model, tokenizer, reward_model, reward_tokenizer, effective_devices


def load_sentiment_classifier(config: EvaluationConfig, device: torch.device) -> ClassifierSentimentEvaluator:
    classifier_dtype = dtype_for_device(config.dtype, device, "sentiment_classifier")
    tokenizer = AutoTokenizer.from_pretrained(config.sentiment_classifier_path, local_files_only=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.sentiment_classifier_path,
            local_files_only=True,
            torch_dtype=classifier_dtype,
        )
        model.eval().to(device)
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(cuda_oom_message("sentiment classifier load", device)) from error
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert_frozen_without_grad(model, "independent sentiment classifier")
    print(f"[Dtype] Sentiment classifier: {str(classifier_dtype).replace('torch.', '')} on {device}", flush=True)
    id2label_raw = getattr(model.config, "id2label", {})
    label_mapping = {int(key): str(value) for key, value in dict(id2label_raw).items()}
    label2id = {label: index for index, label in label_mapping.items()}
    if config.sentiment_classifier_target_label not in label2id:
        raise ValueError(f"Target label {config.sentiment_classifier_target_label!r} missing from classifier label mapping")
    return ClassifierSentimentEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_id=config.sentiment_classifier_path,
        revision=config.sentiment_classifier_revision,
        label_mapping=label_mapping,
        target_label=config.sentiment_classifier_target_label,
        target_label_id=int(label2id[config.sentiment_classifier_target_label]),
        truncation=True,
        max_length=int(config.sentiment_classifier_max_length),
        device=device,
    )


def compute_base_lm_perplexity(
    base_model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    generated_text: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    if not generated_text.strip():
        return float("inf")
    encoded = tokenizer(
        prompt + generated_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.shape[1] < 2:
        return float("inf")
    with torch.inference_mode():
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    return float(torch.exp(loss.detach().float()).cpu().item())


def decode_one(
    method: str,
    prompt: PromptRecord,
    seed: int,
    protocol: DecodingProtocol,
    base_model: torch.nn.Module,
    tokenizer: Any,
    reward_model: torch.nn.Module,
    reward_tokenizer: Any,
    router_checkpoint_path: Path,
    selected_best_beta: float,
    selected_best_heuristic: str = "heuristic_constant_beta_10",
    router_device: torch.device | None = None,
) -> AdaptiveRADSample:
    set_seed(seed)
    adaptive_config = AdaptiveRADConfig(
        top_k=protocol.top_k,
        beta_max=protocol.beta_max,
        max_new_tokens=protocol.max_new_tokens,
        max_reward_length=protocol.max_reward_length,
        do_sample=protocol.do_sample,
        temperature=protocol.temperature,
        seed=seed,
        inverse=protocol.inverse,
        base_model_id=str(DEFAULT_BASE_MODEL_PATH),
        reward_model_id=str(DEFAULT_REWARD_CHECKPOINT_PATH),
    )
    if method == "base_lm":
        return original_fixed_beta_rad_generate([prompt.prompt], base_model, tokenizer, reward_model, reward_tokenizer, 0.0, adaptive_config)[0]
    if method == "best_fixed_beta":
        return original_fixed_beta_rad_generate([prompt.prompt], base_model, tokenizer, reward_model, reward_tokenizer, selected_best_beta, adaptive_config)[0]
    if method == "best_heuristic":
        method = selected_best_heuristic
    if method.startswith("fixed_beta_"):
        beta = float(method.removeprefix("fixed_beta_").replace("p", "."))
        return original_fixed_beta_rad_generate([prompt.prompt], base_model, tokenizer, reward_model, reward_tokenizer, beta, adaptive_config)[0]
    if method == "heuristic_constant_beta_10":
        router = ConstantRouter(10.0, protocol.beta_max).to(router_device or next(base_model.parameters()).device)
    elif method == "heuristic_linear_increase":
        router = HeuristicRouter("linear_increase", protocol.beta_max, protocol.max_new_tokens).to(router_device or next(base_model.parameters()).device)
    elif method == "heuristic_linear_decrease":
        router = HeuristicRouter("linear_decrease", protocol.beta_max, protocol.max_new_tokens).to(router_device or next(base_model.parameters()).device)
    elif method == "heuristic_entropy_beta":
        router = HeuristicRouter("entropy", protocol.beta_max, protocol.max_new_tokens).to(router_device or next(base_model.parameters()).device)
    elif method == "heuristic_reward_range_beta":
        router = HeuristicRouter("reward_range", protocol.beta_max, protocol.max_new_tokens).to(router_device or next(base_model.parameters()).device)
    elif method == "learned_router":
        target_router_device = router_device or next(base_model.parameters()).device
        router = load_router_checkpoint_strict(router_checkpoint_path, adaptive_config, map_location=target_router_device).to(target_router_device)
    else:
        raise ValueError(f"Unknown method {method}")
    return adaptive_rad_generate([prompt.prompt], base_model, tokenizer, reward_model, reward_tokenizer, router, adaptive_config)[0]


def write_metrics_outputs(
    output_dir: Path,
    raw_records: Sequence[dict[str, Any]],
    config: EvaluationConfig,
    best_beta: float,
    best_heuristic: str,
    progress_settings: ProgressSettings | None = None,
    device_diagnostics: dict[str, dict[str, Any]] | None = None,
) -> None:
    if progress_settings is None or progress_settings.enabled:
        print("Independent sentiment evaluation", flush=True)
        print("Perplexity and quality metrics", flush=True)
    metric_records = [sample_to_metric_record(row) for row in raw_records]
    aggregate_rows = aggregate_metrics(metric_records, config.bootstrap_seed, config.bootstrap_samples)
    fixed_sweep_rows = [
        row for row in aggregate_rows
        if str(row["method"]).startswith("fixed_beta_")
    ]
    stats_rows = statistical_tests(
        metric_records,
        baseline_methods=[
            method for method in ["base_lm", f"fixed_beta_{format_beta(best_beta)}", "best_heuristic", *heuristic_method_names(), *[f"fixed_beta_{format_beta(beta)}" for beta in config.fixed_beta_sweep]]
            if method in {str(row.get("method")) for row in metric_records}
        ],
        seed=config.bootstrap_seed,
        samples=config.bootstrap_samples,
    )
    router_rows = router_behavior_rows(metric_records)
    pareto_rows = pareto_points(aggregate_rows)
    assert_no_nan_inf_rows(metric_records, "per-sample metrics")
    assert_no_nan_inf_rows(aggregate_rows, "aggregate metrics")
    assert_no_nan_inf_rows(stats_rows, "statistical tests")
    assert_no_nan_inf_rows(pareto_rows, "pareto points")
    write_csv(output_dir / "per_sample_metrics.csv", metric_records)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "fixed_beta_sweep.csv", fixed_sweep_rows)
    write_csv(output_dir / "statistical_tests.csv", stats_rows)
    write_csv(output_dir / "router_behavior.csv", router_rows)
    write_csv(output_dir / "pareto_points.csv", pareto_rows)
    if progress_settings is None or progress_settings.enabled:
        print("Statistical analysis", flush=True)
        print("Generating Pareto plots", flush=True)
    write_pareto_plots(pareto_rows, output_dir / "plots")
    report = build_stage7_report(config, metric_records, aggregate_rows, stats_rows, pareto_rows, best_beta, best_heuristic, device_diagnostics)
    (output_dir / "stage7_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stage7_report.md").write_text(markdown_report(report), encoding="utf-8")


def build_stage7_report(
    config: EvaluationConfig,
    metric_records: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    stats_rows: Sequence[dict[str, Any]],
    pareto_rows: Sequence[dict[str, Any]],
    best_beta: float,
    best_heuristic: str | None = None,
    device_diagnostics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    learned_vs_best = [
        row for row in stats_rows
        if row["comparison"] == f"learned_router_vs_fixed_beta_{format_beta(best_beta)}"
        and row["metric"] == "classifier_sentiment_success"
    ]
    conclusion = "statistically_indistinguishable_from_best_fixed_beta"
    if learned_vs_best:
        row = learned_vs_best[0]
        if float(row["p_value_bh_adjusted"]) < 0.05 and float(row["mean_difference"]) > 0:
            conclusion = "better_than_best_fixed_beta_on_target_success"
        elif float(row["p_value_bh_adjusted"]) < 0.05 and float(row["mean_difference"]) < 0:
            conclusion = "worse_than_best_fixed_beta_on_target_success"
    return {
        "best_validation_configuration": {
            "best_fixed_beta": best_beta,
            "best_heuristic": best_heuristic,
            "selection_split": "validation",
            "fixed_beta_grid": config.fixed_beta_sweep,
        },
        "final_held_out_test_result": {
            "num_per_sample_records": len(metric_records),
            "prompt_classes": config.prompt_classes,
            "seeds": config.seeds,
            "conclusion": conclusion,
        },
        "sample_counts": {
            "records": len(metric_records),
            "failed_generations": sum(row.get("failure_status") != "success" for row in metric_records),
            "missing_values": sum(any(value is None for value in row.values()) for row in metric_records),
        },
        "reproducibility": {
            "git": git_status(),
            "model_paths": {
                "base_model": config.base_model_path,
                "reward_base": config.reward_base_path,
                "reward_tokenizer": config.reward_tokenizer_path,
                "reward_checkpoint": config.reward_checkpoint_path,
                "router_checkpoint": config.router_checkpoint_path,
                "sentiment_classifier": config.sentiment_classifier_path,
            },
            "hashes": {
                "reward_checkpoint_sha256": file_hash_or_none(config.reward_checkpoint_path),
                "router_checkpoint_sha256": file_hash_or_none(config.router_checkpoint_path),
                "sentiment_classifier_checkpoint_sha256": file_hash_or_none(Path(config.sentiment_classifier_path) / "pytorch_model.bin"),
                "datasets": dataset_hashes(config),
                "original_fixed_rad_source": baseline_source_hashes(),
            },
            "dependency_versions": dependency_versions(),
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "cuda_available": torch.cuda.is_available(),
                "device": config.device,
                "devices": config.devices,
                "resolved_device_diagnostics": device_diagnostics or {},
            },
            "random_seeds": config.seeds,
            "decoding_config": asdict(config.decoding),
            "evaluation_model_identifiers": {
                "independent_sentiment": config.independent_sentiment_evaluator_id,
                "sentiment_classifier": {
                    "model_id": config.sentiment_classifier_path,
                    "revision": config.sentiment_classifier_revision,
                    "label_mapping": classifier_label_mapping_from_config(Path(config.sentiment_classifier_path)),
                    "target_label": config.sentiment_classifier_target_label,
                    "truncation": True,
                    "max_length": config.sentiment_classifier_max_length,
                    "target_probability": "softmax(logits)[POSITIVE]",
                },
                "lexicon_sentiment": "lexicon_sentiment_v1_secondary_metric",
                "fluency": "lexical_perplexity_proxy_v1",
                "coherence": "lexical_overlap_length_proxy_v1",
            },
        },
        "statistical_tests_preview": stats_rows[:10],
        "pareto_points": pareto_rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    best = report["best_validation_configuration"]
    result = report["final_held_out_test_result"]
    return (
        "# Stage 7 Evaluation Report\n\n"
        f"- Best fixed beta selected on validation: `{best['best_fixed_beta']}`\n"
        f"- Held-out records: `{result['num_per_sample_records']}`\n"
        f"- Conclusion: `{result['conclusion']}`\n"
        "- Interpretation rule: improvement is claimed only with paired tests and confidence intervals.\n"
    )


def run_evaluation(config: EvaluationConfig, resume: bool, progress_settings: ProgressSettings | None = None) -> dict[str, Any]:
    progress_settings = progress_settings or ProgressSettings()
    validate_decoding_protocol(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_prompts = load_prompt_split(config, "validation")
    test_prompts = load_prompt_split(config, "test")
    phase_log(1, 4, "Loading models")
    resolved_devices = resolve_component_devices(config)
    for diag in resolved_devices.diagnostics.values():
        print_device_diagnostics(diag)
    base_model, tokenizer, reward_model, reward_tokenizer, resolved_devices = load_models(config, resolved_devices)
    sentiment_evaluator = load_sentiment_classifier(config, resolved_devices.sentiment_classifier)
    protocol = config.decoding
    validation_records: list[dict[str, Any]] = []
    phase_log(2, 4, "Validation model selection")
    validation_fixed_jobs = [
        EvaluationJob(prompt=prompt, method=f"fixed_beta_{format_beta(beta)}", seed=seed, prompt_index=prompt_index)
        for prompt_index, prompt in enumerate(validation_prompts, start=1)
        for seed in config.seeds
        for beta in config.fixed_beta_sweep
    ]
    fixed_progress = EvaluationProgress(
        "Validation fixed-beta sweep",
        total=len(validation_fixed_jobs),
        initial=0,
        settings=progress_settings,
    )
    for prompt in validation_prompts:
        for seed in config.seeds:
            for beta in config.fixed_beta_sweep:
                method = f"fixed_beta_{format_beta(beta)}"
                sample = decode_one(method, prompt, seed, protocol, base_model, tokenizer, reward_model, reward_tokenizer, Path(config.router_checkpoint_path), beta, router_device=resolved_devices.router)
                ppl = compute_base_lm_perplexity(base_model, tokenizer, prompt.prompt, sample.text, resolved_devices.base_model)
                validation_records.append(attach_classifier_metrics(build_raw_record(prompt, method, seed, sample, protocol.beta_max, ppl), sentiment_evaluator))
                fixed_progress.update(job=EvaluationJob(prompt, method, seed, validation_prompts.index(prompt) + 1), success=True)
    fixed_progress.close()
    validation_heuristic_jobs = [
        EvaluationJob(prompt=prompt, method=method, seed=seed, prompt_index=prompt_index)
        for prompt_index, prompt in enumerate(validation_prompts, start=1)
        for seed in config.seeds
        for method in heuristic_method_names()
    ]
    heuristic_progress = EvaluationProgress(
        "Validation heuristic sweep",
        total=len(validation_heuristic_jobs),
        initial=0,
        settings=progress_settings,
    )
    for prompt_index, prompt in enumerate(validation_prompts, start=1):
        for seed in config.seeds:
            for method in heuristic_method_names():
                sample = decode_one(method, prompt, seed, protocol, base_model, tokenizer, reward_model, reward_tokenizer, Path(config.router_checkpoint_path), config.fixed_beta_sweep[0], router_device=resolved_devices.router)
                ppl = compute_base_lm_perplexity(base_model, tokenizer, prompt.prompt, sample.text, resolved_devices.base_model)
                validation_records.append(attach_classifier_metrics(build_raw_record(prompt, method, seed, sample, protocol.beta_max, ppl), sentiment_evaluator))
                heuristic_progress.update(job=EvaluationJob(prompt, method, seed, prompt_index), success=True)
    heuristic_progress.close()
    validation_metrics = aggregate_metrics([sample_to_metric_record(row) for row in validation_records], config.bootstrap_seed, config.bootstrap_samples)
    best_beta = select_best_fixed_beta(validation_metrics, config.fixed_beta_sweep)
    best_heuristic = select_best_heuristic(validation_metrics)
    cleanup_after_phase()

    phase_log(3, 4, "Held-out test evaluation")
    outputs_path = output_dir / "per_sample_outputs.jsonl"
    completed = load_completed_keys(outputs_path) if resume else set()
    raw_records = []
    methods = method_names(config, include_sweep=True)
    jobs = evaluation_jobs(test_prompts, methods, config.seeds)
    planned_keys = {job.key for job in jobs}
    if resume and outputs_path.exists():
        with outputs_path.open("r", encoding="utf-8") as handle:
            raw_records = [
                json.loads(line)
                for line in handle
                if line.strip() and generation_key(json.loads(line)) in planned_keys
            ]
        completed = completed & planned_keys
    total_jobs = total_evaluation_jobs(test_prompts, methods, config.seeds)
    completed_in_plan = completed_jobs_in_plan(jobs, completed)
    if resume:
        print(f"Resuming: {completed_in_plan}/{total_jobs} jobs already completed", flush=True)
    pending_jobs = pending_evaluation_jobs(jobs, completed)
    eval_progress = EvaluationProgress(
        "Stage 7 Evaluation",
        total=total_jobs,
        initial=completed_in_plan,
        settings=progress_settings,
        output_path=outputs_path,
    )
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    if not resume:
        raw_records = []
    with outputs_path.open(mode, encoding="utf-8") as output_handle:
        for job in pending_jobs:
            prompt = job.prompt
            method = job.method
            seed = job.seed
            start = time.perf_counter()
            success = False
            try:
                sample = decode_one(method, prompt, seed, protocol, base_model, tokenizer, reward_model, reward_tokenizer, Path(config.router_checkpoint_path), best_beta, best_heuristic, router_device=resolved_devices.router)
                ppl = compute_base_lm_perplexity(base_model, tokenizer, prompt.prompt, sample.text, resolved_devices.base_model)
                row = sample_to_metric_record(attach_classifier_metrics(build_raw_record(prompt, method, seed, sample, protocol.beta_max, ppl), sentiment_evaluator))
                success = row.get("failure_status") == "success"
            except Exception as error:
                row = sample_to_metric_record({
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.prompt,
                    "prompt_sentiment_class": prompt.prompt_sentiment_class,
                    "method": method,
                    "seed": seed,
                    "generated_text": "",
                    "selected_token_ids": [],
                    "beta_history": [],
                    "latency": {"latency_scope": "sample", "total_generation_time": time.perf_counter() - start, "average_latency_per_token": 0.0},
                    "failure_status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
                success = False
            append_jsonl_record(output_handle, row)
            raw_records.append(row)
            eval_progress.update(job=job, success=success)
    eval_progress.close()
    phase_log(4, 4, "Metrics, statistics, reports and plots")
    cleanup_after_phase()
    write_metrics_outputs(
        output_dir,
        raw_records,
        config,
        best_beta,
        best_heuristic,
        progress_settings,
        resolved_devices.diagnostics,
    )
    return {"output_dir": str(output_dir), "best_fixed_beta": best_beta, "best_heuristic": best_heuristic, "records": len(raw_records)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAD router against fixed-beta and heuristic baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVALUATION_DIR / "config.json")
    parser.add_argument("--write-default-config", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--log-every-jobs", type=int, default=10)
    parser.add_argument("--log-every-seconds", type=float, default=60.0)
    parser.add_argument("--check-device", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], help="Override config device for all components.")
    return parser.parse_args(argv)


def check_device(config: EvaluationConfig) -> int:
    validate_decoding_protocol(config)
    try:
        resolved = resolve_component_devices(config)
    except RuntimeError as error:
        print(str(error), flush=True)
        print("Device check: FAILED FOR CUDA, CPU FALLBACK AVAILABLE", flush=True)
        return 1
    for component, diagnostics in resolved.diagnostics.items():
        print(f"[Device] Component: {component}", flush=True)
        print_device_diagnostics(diagnostics)
    print(f"[Dtype] Base model: {str(dtype_for_device(config.dtype, resolved.base_model, 'base_model')).replace('torch.', '')}", flush=True)
    print(f"[Dtype] Reward model: {str(dtype_for_device(config.dtype, resolved.reward_model, 'reward_model')).replace('torch.', '')}", flush=True)
    print(f"[Dtype] Router: {str(dtype_for_device(config.dtype, resolved.router, 'router')).replace('torch.', '')}", flush=True)
    print(
        f"[Dtype] Sentiment classifier: "
        f"{str(dtype_for_device(config.dtype, resolved.sentiment_classifier, 'sentiment_classifier')).replace('torch.', '')} "
        f"on {resolved.sentiment_classifier}",
        flush=True,
    )
    failed_cuda = any(
        diag.get("requested_device") in {"auto", "cuda", "cuda:0"} and diag.get("resolved_device") == "cpu"
        for diag in resolved.diagnostics.values()
    )
    if failed_cuda:
        print("Device check: FAILED FOR CUDA, CPU FALLBACK AVAILABLE", flush=True)
    else:
        print("Device check: PASS", flush=True)
    print(f"base_model -> {resolved.base_model}", flush=True)
    print(f"reward_model -> {resolved.reward_model}", flush=True)
    print(f"router -> {resolved.router}", flush=True)
    print(f"sentiment_classifier -> {resolved.sentiment_classifier}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.write_default_config or not args.config.exists():
        write_default_config(args.config)
        if args.write_default_config:
            print(json.dumps({"config": str(args.config)}, indent=2))
            return
    config = load_config(args.config)
    if args.device:
        config = replace(
            config,
            device=args.device,
            devices={
                "base_model": args.device,
                "reward_model": args.device,
                "router": args.device,
                "sentiment_classifier": args.device,
            },
        )
    if args.check_device:
        raise SystemExit(check_device(config))
    progress_settings = ProgressSettings(
        enabled=bool(args.progress),
        log_every_jobs=max(int(args.log_every_jobs), 1),
        log_every_seconds=max(float(args.log_every_seconds), 0.0),
    )
    result = run_evaluation(config, resume=args.resume, progress_settings=progress_settings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
