"""Adaptive token-level router decoding for RAD.

This module adds a separate learned-router decoding path. It intentionally does
not modify or wrap the original fixed-beta RAD implementation files.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from router.features import compute_guided_scores
from router.model import RADTokenRouter
from Method.RAD.utils.logits_processor import RewardAugmentedLogitsProcessor


REWARD_TRANSFORM_NAME = "rad_clamp_0_1"
REWARD_TRANSFORM_NAME_INVERSE = "rad_clamp_0_1_inverse"
REWARD_TRANSFORM_FORMULA = "rad_reward_scores = clamp(raw_reward_scores, 0, 1); if inverse then 1 - clamped"


@dataclass(frozen=True)
class AdaptiveRADConfig:
    top_k: int = 20
    beta_max: float = 30.0
    max_new_tokens: int = 64
    max_reward_length: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    seed: int = 1
    inverse: bool = False
    base_model_id: str | None = None
    reward_model_id: str | None = None


@dataclass
class AdaptiveRADSample:
    prompt: str
    text: str
    token_ids: list[int]
    beta_history: list[float]
    gate_history: list[float]
    selected_token_ids: list[int]
    candidate_ids_history: list[list[int]]
    base_logits_history: list[list[float]]
    rad_reward_scores_history: list[list[float]]
    guided_scores_history: list[list[float]]
    latency: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rad_transform_reward(raw_reward_scores: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply the exact reward transform used by RAD before beta weighting."""
    transformed = torch.clamp(raw_reward_scores.float(), min=0.0, max=1.0)
    if inverse:
        transformed = 1.0 - transformed
    return transformed


def expected_reward_transform_name(inverse: bool) -> str:
    return REWARD_TRANSFORM_NAME_INVERSE if inverse else REWARD_TRANSFORM_NAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_mapping(tokenizer: Any) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    return {str(token): int(token_id) for token, token_id in vocab.items()}


def assert_tokenizers_compatible(base_tokenizer: Any, reward_tokenizer: Any) -> None:
    base_vocab = tokenizer_mapping(base_tokenizer)
    reward_vocab = tokenizer_mapping(reward_tokenizer)
    if base_vocab != reward_vocab:
        raise ValueError("Base LM tokenizer and reward tokenizer vocabularies differ")
    for attr in ("eos_token_id", "bos_token_id"):
        if getattr(base_tokenizer, attr, None) != getattr(reward_tokenizer, attr, None):
            raise ValueError(f"Tokenizer {attr} differs")


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None


def assert_frozen_without_grad(model: torch.nn.Module, name: str) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError(f"{name} parameter still has requires_grad=True")
        if parameter.grad is not None:
            raise AssertionError(f"{name} parameter has a gradient during adaptive RAD generation")


def streaming_parameter_sha256(model: torch.nn.Module) -> str:
    """Optional low-memory parameter digest for diagnostics outside latency paths."""
    digest = hashlib.sha256()
    for parameter in model.parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_adaptive_config(config: AdaptiveRADConfig) -> None:
    if config.top_k != 20:
        raise ValueError(f"Adaptive RAD router v1 requires top_k=20, got {config.top_k}")
    if config.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {config.max_new_tokens}")
    if config.max_reward_length <= 0:
        raise ValueError(f"max_reward_length must be positive, got {config.max_reward_length}")
    if config.do_sample and config.temperature <= 0:
        raise ValueError("temperature must be positive when sampling")


def checkpoint_router_config(payload: dict[str, Any]) -> dict[str, Any]:
    if "router_state_dict" in payload:
        training_config = dict(payload.get("config", {}))
        return {
            "beta_max": float(training_config["beta_max"]),
            "beta_init": training_config.get("beta_init"),
            "top_k": int(training_config.get("top_k", 20)),
            "hidden_dim": int(training_config.get("hidden_dim", 64)),
            "eps": float(training_config.get("eps", 1e-6)),
        }
    if payload.get("class") != "RADTokenRouter":
        raise ValueError("Unsupported router checkpoint class")
    return dict(payload["config"])


def load_router_checkpoint_strict(
    checkpoint_path: Path | str,
    config: AdaptiveRADConfig,
    map_location: str | torch.device = "cpu",
) -> RADTokenRouter:
    validate_adaptive_config(config)
    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    router_config = checkpoint_router_config(payload)
    state_dict_key = "router_state_dict" if "router_state_dict" in payload else "state_dict"
    state_dict = payload[state_dict_key]

    if int(router_config.get("top_k", 20)) != 20:
        raise ValueError(f"Router checkpoint top_k must be 20, got {router_config.get('top_k')}")
    if config.top_k != 20:
        raise ValueError(f"Adaptive RAD requires runtime top_k=20, got {config.top_k}")
    if abs(float(router_config["beta_max"]) - float(config.beta_max)) > 1e-9:
        raise ValueError(
            f"Router beta_max={router_config['beta_max']} does not match runtime beta_max={config.beta_max}"
        )

    expected_transform = expected_reward_transform_name(config.inverse)
    checkpoint_transform = payload.get("reward_transform_name")
    checkpoint_extra = payload.get("extra", {})
    if checkpoint_transform is None and isinstance(checkpoint_extra, dict):
        checkpoint_transform = checkpoint_extra.get("reward_transform_name")
    if checkpoint_transform is not None and checkpoint_transform != expected_transform:
        raise ValueError(f"Router reward transform {checkpoint_transform!r} does not match {expected_transform!r}")

    training_config = dict(payload.get("config", {})) if "router_state_dict" in payload else {}
    manifest_hashes = dict(payload.get("dataset_manifest_hashes", {}))
    for manifest_key in ("train_manifest", "validation_manifest"):
        manifest_path_value = training_config.get(manifest_key)
        if not manifest_path_value:
            continue
        manifest_path = Path(manifest_path_value)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Router checkpoint references missing feature manifest: {manifest_path}")
        expected_hash = manifest_hashes.get(str(manifest_path))
        if expected_hash is not None and sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"Feature manifest hash changed since router training: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("top_k", 20)) != config.top_k:
            raise ValueError(f"Feature manifest top_k={manifest.get('top_k')} does not match runtime top_k={config.top_k}")
        if int(manifest.get("max_reward_length", config.max_reward_length)) != config.max_reward_length:
            raise ValueError(
                f"Feature manifest max_reward_length={manifest.get('max_reward_length')} "
                f"does not match runtime max_reward_length={config.max_reward_length}"
            )
        manifest_transform = manifest.get("reward_transform_name")
        if manifest_transform != expected_transform:
            raise ValueError(f"Feature manifest reward transform {manifest_transform!r} does not match {expected_transform!r}")
        identifiers = dict(manifest.get("tokenizer_model_identifiers", {}))
        if config.base_model_id is not None and identifiers.get("base_model") not in {None, config.base_model_id}:
            raise ValueError("Router feature cache base model does not match decoding base model")
        if config.reward_model_id is not None and identifiers.get("reward_checkpoint") not in {None, config.reward_model_id}:
            raise ValueError("Router feature cache reward checkpoint does not match decoding reward model")

    router = RADTokenRouter(**router_config)
    router.load_state_dict(state_dict, strict=True)
    freeze_model(router)
    return router


def score_candidate_rewards(
    reward_model: torch.nn.Module,
    candidate_input_ids: torch.Tensor,
    candidate_attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        output = reward_model(
            input_ids=candidate_input_ids.to(device),
            attention_mask=candidate_attention_mask.to(device),
            labels=None,
            use_cache=False,
        )
    if isinstance(output, tuple):
        reward_scores = output[1]
    else:
        reward_scores = output.logits
    if reward_scores.ndim == 1:
        reward_scores = reward_scores.unsqueeze(-1)
    if reward_scores.ndim != 2 or reward_scores.shape[1] != 1:
        raise ValueError(f"Reward model must return candidate scores with shape [N, 1], got {tuple(reward_scores.shape)}")
    return reward_scores[:, 0].detach().float().to(candidate_input_ids.device)


def build_candidate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    max_reward_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, prefix_length = input_ids.shape
    top_k = candidate_ids.shape[1]
    expanded_ids = input_ids.unsqueeze(1).expand(batch_size, top_k, prefix_length)
    expanded_mask = attention_mask.unsqueeze(1).expand(batch_size, top_k, prefix_length)
    candidate_input_ids = torch.cat([expanded_ids, candidate_ids.unsqueeze(-1)], dim=-1)
    candidate_attention_mask = torch.cat(
        [expanded_mask, torch.ones(batch_size, top_k, 1, dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=-1,
    )
    candidate_input_ids = candidate_input_ids.reshape(batch_size * top_k, prefix_length + 1)
    candidate_attention_mask = candidate_attention_mask.reshape(batch_size * top_k, prefix_length + 1)
    return (
        candidate_input_ids[:, -max_reward_length:],
        candidate_attention_mask[:, -max_reward_length:],
    )


def select_next_candidate(
    guided_scores: torch.Tensor,
    do_sample: bool,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not do_sample:
        return guided_scores.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    probs = torch.softmax(guided_scores.float() / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def summarize_latency(
    total_generation_time: float,
    per_step_latency: list[dict[str, float]],
    generated_tokens: int,
) -> dict[str, Any]:
    totals = {
        "base_lm_latency": sum(step["base_lm_latency"] for step in per_step_latency),
        "reward_model_latency": sum(step["reward_model_latency"] for step in per_step_latency),
        "router_latency": sum(step["router_latency"] for step in per_step_latency),
        "sampling_latency": sum(step["sampling_latency"] for step in per_step_latency),
    }
    return {
        "latency_scope": "batch",
        "total_generation_time": float(total_generation_time),
        "per_token_total_latency": [float(step["total_latency"]) for step in per_step_latency],
        **{key: float(value) for key, value in totals.items()},
        "average_latency_per_token": float(total_generation_time / max(generated_tokens, 1)),
    }


def adaptive_rad_generate(
    prompts: Sequence[str],
    base_model: torch.nn.Module,
    base_tokenizer: Any,
    reward_model: torch.nn.Module,
    reward_tokenizer: Any,
    router: torch.nn.Module,
    config: AdaptiveRADConfig,
) -> list[AdaptiveRADSample]:
    validate_adaptive_config(config)
    assert_tokenizers_compatible(base_tokenizer, reward_tokenizer)
    freeze_model(base_model)
    freeze_model(reward_model)
    freeze_model(router)
    assert_frozen_without_grad(base_model, "base LM")
    assert_frozen_without_grad(reward_model, "reward model")
    assert_frozen_without_grad(router, "router")

    base_device = next(base_model.parameters()).device
    reward_device = next(reward_model.parameters()).device
    router_device = next(router.parameters()).device
    generator = torch.Generator(device=base_device)
    generator.manual_seed(int(config.seed))

    encoded = base_tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(base_device)
    attention_mask = encoded["attention_mask"].to(base_device)

    beta_history: list[list[float]] = [[] for _ in prompts]
    gate_history: list[list[float]] = [[] for _ in prompts]
    selected_token_ids: list[list[int]] = [[] for _ in prompts]
    candidate_ids_history: list[list[list[int]]] = [[] for _ in prompts]
    base_logits_history: list[list[list[float]]] = [[] for _ in prompts]
    rad_reward_scores_history: list[list[list[float]]] = [[] for _ in prompts]
    guided_scores_history: list[list[list[float]]] = [[] for _ in prompts]
    per_step_latency: list[dict[str, float]] = []
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=base_device)
    eos_token_id = getattr(base_tokenizer, "eos_token_id", None)
    pad_token_id = getattr(base_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    total_start = time.perf_counter()
    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise AssertionError("Gradients must be disabled during adaptive RAD generation")
        for _step in range(config.max_new_tokens):
            if bool(finished.all().item()):
                break
            step_start = time.perf_counter()
            base_start = time.perf_counter()
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_logits = logits[:, -1, :].detach()
            base_latency = time.perf_counter() - base_start

            topk_scores, topk_ids = torch.topk(next_token_logits, k=config.top_k, dim=-1)
            candidate_input_ids, candidate_attention_mask = build_candidate_inputs(
                input_ids,
                attention_mask,
                topk_ids,
                max_reward_length=config.max_reward_length,
            )

            reward_start = time.perf_counter()
            raw_rewards = score_candidate_rewards(reward_model, candidate_input_ids, candidate_attention_mask, reward_device)
            rad_rewards = rad_transform_reward(raw_rewards.reshape(len(prompts), config.top_k), inverse=config.inverse).to(base_device)
            reward_latency = time.perf_counter() - reward_start

            router_start = time.perf_counter()
            beta, gate = router(
                topk_scores.detach().float().to(router_device),
                rad_rewards.detach().float().to(router_device),
            )
            beta = beta.to(device=base_device, dtype=topk_scores.dtype)
            gate = gate.to(device=base_device, dtype=topk_scores.dtype)
            router_latency = time.perf_counter() - router_start

            sampling_start = time.perf_counter()
            guided_scores = compute_guided_scores(topk_scores, rad_rewards, beta, top_k=config.top_k)
            selected_candidate_indices = select_next_candidate(
                guided_scores,
                do_sample=config.do_sample,
                temperature=config.temperature,
                generator=generator,
            )
            next_token_ids = topk_ids.gather(1, selected_candidate_indices.unsqueeze(-1)).squeeze(-1)
            append_token_ids = torch.where(finished, torch.full_like(next_token_ids, int(pad_token_id)), next_token_ids)
            append_attention = (~finished).long().unsqueeze(-1)
            sampling_latency = time.perf_counter() - sampling_start

            input_ids = torch.cat([input_ids, append_token_ids.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, append_attention], dim=-1)

            for row in range(len(prompts)):
                if bool(finished[row].item()):
                    continue
                beta_history[row].append(float(beta[row, 0].detach().cpu().item()))
                gate_history[row].append(float(gate[row, 0].detach().cpu().item()))
                selected_token_ids[row].append(int(next_token_ids[row].detach().cpu().item()))
                candidate_ids_history[row].append([int(value) for value in topk_ids[row].detach().cpu().tolist()])
                base_logits_history[row].append([float(value) for value in topk_scores[row].detach().cpu().float().tolist()])
                rad_reward_scores_history[row].append([float(value) for value in rad_rewards[row].detach().cpu().float().tolist()])
                guided_scores_history[row].append([float(value) for value in guided_scores[row].detach().cpu().float().tolist()])
                if eos_token_id is not None and int(next_token_ids[row].detach().cpu().item()) == int(eos_token_id):
                    finished[row] = True

            per_step_latency.append(
                {
                    "base_lm_latency": base_latency,
                    "reward_model_latency": reward_latency,
                    "router_latency": router_latency,
                    "sampling_latency": sampling_latency,
                    "total_latency": time.perf_counter() - step_start,
                }
            )

    total_time = time.perf_counter() - total_start
    assert_frozen_without_grad(base_model, "base LM")
    assert_frozen_without_grad(reward_model, "reward model")
    assert_frozen_without_grad(router, "router")

    samples: list[AdaptiveRADSample] = []
    for row, prompt in enumerate(prompts):
        generated_ids = selected_token_ids[row]
        text = base_tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency = summarize_latency(total_time, per_step_latency, len(selected_token_ids[row]))
        samples.append(
            AdaptiveRADSample(
                prompt=str(prompt),
                text=str(text),
                token_ids=[int(value) for value in generated_ids],
                beta_history=beta_history[row],
                gate_history=gate_history[row],
                selected_token_ids=selected_token_ids[row],
                candidate_ids_history=candidate_ids_history[row],
                base_logits_history=base_logits_history[row],
                rad_reward_scores_history=rad_reward_scores_history[row],
                guided_scores_history=guided_scores_history[row],
                latency=latency,
            )
        )
    return samples


def fixed_beta_rad_generate(
    prompts: Sequence[str],
    base_model: torch.nn.Module,
    base_tokenizer: Any,
    reward_model: torch.nn.Module,
    reward_tokenizer: Any,
    beta: float,
    config: AdaptiveRADConfig,
) -> list[AdaptiveRADSample]:
    class _ConstantRouter(torch.nn.Module):
        def __init__(self, beta_value: float, beta_max: float) -> None:
            super().__init__()
            self.beta_value = float(beta_value)
            self.beta_max = float(beta_max)
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            beta_tensor = torch.full((base_logits.shape[0], 1), self.beta_value, dtype=base_logits.dtype, device=base_logits.device)
            gate_tensor = torch.full((base_logits.shape[0], 1), self.beta_value / self.beta_max, dtype=base_logits.dtype, device=base_logits.device)
            return beta_tensor, gate_tensor

    return adaptive_rad_generate(
        prompts,
        base_model,
        base_tokenizer,
        reward_model,
        reward_tokenizer,
        _ConstantRouter(beta, config.beta_max).to(next(base_model.parameters()).device),
        config,
    )


def original_fixed_beta_rad_generate(
    prompts: Sequence[str],
    base_model: torch.nn.Module,
    base_tokenizer: Any,
    reward_model: torch.nn.Module,
    reward_tokenizer: Any,
    beta: float,
    config: AdaptiveRADConfig,
) -> list[AdaptiveRADSample]:
    """CPU-safe fixed-beta wrapper that uses RAD's original score transform."""
    validate_adaptive_config(config)
    assert_tokenizers_compatible(base_tokenizer, reward_tokenizer)
    freeze_model(base_model)
    freeze_model(reward_model)
    assert_frozen_without_grad(base_model, "base LM")
    assert_frozen_without_grad(reward_model, "reward model")

    processor = RewardAugmentedLogitsProcessor.__new__(RewardAugmentedLogitsProcessor)
    processor._inverse = config.inverse
    processor._method = "linear"
    processor._beta = float(beta)

    base_device = next(base_model.parameters()).device
    reward_device = next(reward_model.parameters()).device
    encoded = base_tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(base_device)
    attention_mask = encoded["attention_mask"].to(base_device)
    selected_token_ids: list[list[int]] = [[] for _ in prompts]
    candidate_ids_history: list[list[list[int]]] = [[] for _ in prompts]
    base_logits_history: list[list[list[float]]] = [[] for _ in prompts]
    rad_reward_scores_history: list[list[list[float]]] = [[] for _ in prompts]
    guided_scores_history: list[list[list[float]]] = [[] for _ in prompts]
    beta_history: list[list[float]] = [[] for _ in prompts]
    gate_history: list[list[float]] = [[] for _ in prompts]
    per_step_latency: list[dict[str, float]] = []
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=base_device)
    eos_token_id = getattr(base_tokenizer, "eos_token_id", None)
    pad_token_id = getattr(base_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    total_start = time.perf_counter()
    with torch.inference_mode():
        for _step in range(config.max_new_tokens):
            if bool(finished.all().item()):
                break
            step_start = time.perf_counter()
            base_start = time.perf_counter()
            logits = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
            topk_scores, topk_ids = torch.topk(logits, k=config.top_k, dim=-1)
            base_latency = time.perf_counter() - base_start

            candidate_input_ids, candidate_attention_mask = build_candidate_inputs(
                input_ids,
                attention_mask,
                topk_ids,
                max_reward_length=config.max_reward_length,
            )
            reward_start = time.perf_counter()
            raw_rewards = score_candidate_rewards(reward_model, candidate_input_ids, candidate_attention_mask, reward_device)
            rad_rewards = rad_transform_reward(raw_rewards.reshape(len(prompts), config.top_k), inverse=config.inverse).to(base_device)
            reward_latency = time.perf_counter() - reward_start

            sampling_start = time.perf_counter()
            guided_scores = torch.stack([
                processor.apply_function(topk_scores[row], rad_rewards[row], float(beta))
                for row in range(len(prompts))
            ])
            selected_candidate_indices = select_next_candidate(
                guided_scores,
                do_sample=config.do_sample,
                temperature=config.temperature,
                generator=torch.Generator(device=base_device).manual_seed(int(config.seed) + _step),
            )
            next_token_ids = topk_ids.gather(1, selected_candidate_indices.unsqueeze(-1)).squeeze(-1)
            append_token_ids = torch.where(finished, torch.full_like(next_token_ids, int(pad_token_id)), next_token_ids)
            append_attention = (~finished).long().unsqueeze(-1)
            sampling_latency = time.perf_counter() - sampling_start

            input_ids = torch.cat([input_ids, append_token_ids.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, append_attention], dim=-1)
            for row in range(len(prompts)):
                if bool(finished[row].item()):
                    continue
                token_id = int(next_token_ids[row].detach().cpu().item())
                selected_token_ids[row].append(token_id)
                beta_history[row].append(float(beta))
                gate_history[row].append(1.0)
                candidate_ids_history[row].append([int(value) for value in topk_ids[row].detach().cpu().tolist()])
                base_logits_history[row].append([float(value) for value in topk_scores[row].detach().cpu().float().tolist()])
                rad_reward_scores_history[row].append([float(value) for value in rad_rewards[row].detach().cpu().float().tolist()])
                guided_scores_history[row].append([float(value) for value in guided_scores[row].detach().cpu().float().tolist()])
                if eos_token_id is not None and token_id == int(eos_token_id):
                    finished[row] = True
            per_step_latency.append(
                {
                    "base_lm_latency": base_latency,
                    "reward_model_latency": reward_latency,
                    "router_latency": 0.0,
                    "sampling_latency": sampling_latency,
                    "total_latency": time.perf_counter() - step_start,
                }
            )

    total_time = time.perf_counter() - total_start
    assert_frozen_without_grad(base_model, "base LM")
    assert_frozen_without_grad(reward_model, "reward model")
    return [
        AdaptiveRADSample(
            prompt=str(prompt),
            text=str(base_tokenizer.decode(selected_token_ids[row], skip_special_tokens=True)),
            token_ids=selected_token_ids[row],
            beta_history=beta_history[row],
            gate_history=gate_history[row],
            selected_token_ids=selected_token_ids[row],
            candidate_ids_history=candidate_ids_history[row],
            base_logits_history=base_logits_history[row],
            rad_reward_scores_history=rad_reward_scores_history[row],
            guided_scores_history=guided_scores_history[row],
            latency=summarize_latency(total_time, per_step_latency, len(selected_token_ids[row])),
        )
        for row, prompt in enumerate(prompts)
    ]


def write_generation_json(path: Path, samples: Sequence[AdaptiveRADSample], config: AdaptiveRADConfig) -> None:
    return write_adaptive_generation_json(path, samples, config)


def write_adaptive_generation_json(
    path: Path,
    samples: Sequence[AdaptiveRADSample],
    config: AdaptiveRADConfig,
    router_checkpoint_path: Path | str | None = None,
    reward_checkpoint_path: Path | str | None = None,
    decoding_mode: str = "learned_router",
) -> None:
    router_checkpoint_sha256 = sha256_file(Path(router_checkpoint_path)) if router_checkpoint_path else None
    reward_checkpoint_sha256 = sha256_file(Path(reward_checkpoint_path)) if reward_checkpoint_path else None
    payload = {
        "config": asdict(config),
        "decoding_mode": decoding_mode,
        "router_checkpoint_sha256": router_checkpoint_sha256,
        "base_model_id": config.base_model_id,
        "reward_checkpoint_sha256": reward_checkpoint_sha256,
        "max_reward_length": config.max_reward_length,
        "reward_transform_name": expected_reward_transform_name(config.inverse),
        "reward_transform_formula": REWARD_TRANSFORM_FORMULA,
        "samples": [sample.to_dict() for sample in samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
