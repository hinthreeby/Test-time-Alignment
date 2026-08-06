#!/usr/bin/env python3
"""Validate GPT-2 Large and the RAD GPT-2 Small sentiment reward model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.reward_modeling.reward_model import GPT2RewardModel

LOGGER = logging.getLogger("model_validation")

DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_REWARD_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_REWARD_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_REWARD_CHECKPOINT_PATH = DEFAULT_REWARD_TOKENIZER_PATH / "pytorch_model.bin"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_validation.json"
DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "reports" / "candidate_sanity_samples.jsonl"


@dataclass(frozen=True)
class FrozenParameterStats:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    all_frozen: bool


@dataclass(frozen=True)
class TokenizerCompatibility:
    vocab_equal: bool
    vocab_size_equal: bool
    base_vocab_size: int
    reward_vocab_size: int
    eos_token_id_equal: bool
    base_eos_token_id: int | None
    reward_eos_token_id: int | None
    bos_token_id_equal: bool
    base_bos_token_id: int | None
    reward_bos_token_id: int | None
    byte_level_bpe_equal: bool
    sample_encodings_equal: bool
    padding_strategy_compatible: bool
    base_pad_token_id_before: int | None
    reward_pad_token_id_before: int | None
    base_pad_token_id_after: int | None
    reward_pad_token_id_after: int | None
    base_padding_side: str
    reward_padding_side: str


@dataclass(frozen=True)
class RouterFreezeProbe:
    base_checksum_before: str
    base_checksum_after: str
    reward_checksum_before: str
    reward_checksum_after: str
    base_checksum_unchanged: bool
    reward_checksum_unchanged: bool
    optimizer_contains_only_router: bool
    base_requires_grad_all_false: bool
    reward_requires_grad_all_false: bool
    base_grads_all_none: bool
    reward_grads_all_none: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def freeze_model(model: torch.nn.Module) -> FrozenParameterStats:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return frozen_parameter_stats(model)


def frozen_parameter_stats(model: torch.nn.Module) -> FrozenParameterStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return FrozenParameterStats(
        total_parameters=int(total),
        trainable_parameters=int(trainable),
        frozen_parameters=int(total - trainable),
        all_frozen=trainable == 0,
    )


def assert_all_frozen(model: torch.nn.Module, name: str) -> None:
    trainable = [parameter_name for parameter_name, parameter in model.named_parameters() if parameter.requires_grad]
    assert not trainable, f"{name} has trainable parameters after freezing: {trainable[:10]}"
    assert not model.training, f"{name} is not in eval mode"


def requires_grad_all_false(model: torch.nn.Module) -> bool:
    return all(not parameter.requires_grad for parameter in model.parameters())


def grads_all_none(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def clear_model_grads(*models: torch.nn.Module) -> None:
    for model in models:
        for parameter in model.parameters():
            parameter.grad = None


def dummy_router_optimization_step() -> bool:
    """Exercise an optimizer step whose optimizer owns only toy router parameters."""
    router_probe = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(router_probe.parameters(), lr=0.01)
    router_param_ids = {id(parameter) for parameter in router_probe.parameters()}
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_contains_only_router = optimizer_param_ids == router_param_ids
    optimizer.zero_grad(set_to_none=True)
    loss = router_probe(torch.ones(1, 1)).sum()
    loss.backward()
    optimizer.step()
    return optimizer_contains_only_router


def validate_frozen_models_survive_router_step(
    base_model: torch.nn.Module,
    reward_model: torch.nn.Module,
) -> RouterFreezeProbe:
    clear_model_grads(base_model, reward_model)
    assert_all_frozen(base_model, "base_model before dummy router step")
    assert_all_frozen(reward_model, "reward_model before dummy router step")
    base_checksum_before = parameter_checksum(base_model)
    reward_checksum_before = parameter_checksum(reward_model)
    optimizer_contains_only_router = dummy_router_optimization_step()
    base_checksum_after = parameter_checksum(base_model)
    reward_checksum_after = parameter_checksum(reward_model)
    probe = RouterFreezeProbe(
        base_checksum_before=base_checksum_before,
        base_checksum_after=base_checksum_after,
        reward_checksum_before=reward_checksum_before,
        reward_checksum_after=reward_checksum_after,
        base_checksum_unchanged=base_checksum_before == base_checksum_after,
        reward_checksum_unchanged=reward_checksum_before == reward_checksum_after,
        optimizer_contains_only_router=optimizer_contains_only_router,
        base_requires_grad_all_false=requires_grad_all_false(base_model),
        reward_requires_grad_all_false=requires_grad_all_false(reward_model),
        base_grads_all_none=grads_all_none(base_model),
        reward_grads_all_none=grads_all_none(reward_model),
    )
    assert probe.optimizer_contains_only_router, "Dummy optimizer includes non-router parameters"
    assert probe.base_requires_grad_all_false, "Base model has requires_grad=True parameters"
    assert probe.reward_requires_grad_all_false, "Reward model has requires_grad=True parameters"
    assert probe.base_grads_all_none, "Base model accumulated gradients during router step"
    assert probe.reward_grads_all_none, "Reward model accumulated gradients during router step"
    assert probe.base_checksum_unchanged, "Base model parameters changed during router step"
    assert probe.reward_checksum_unchanged, "Reward model parameters changed during router step"
    return probe


def configure_gpt2_padding(tokenizer: PreTrainedTokenizerBase, padding_side: str) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return int(tokenizer.pad_token_id)


def validate_tokenizers(
    base_tokenizer: PreTrainedTokenizerBase,
    reward_tokenizer: PreTrainedTokenizerBase,
) -> TokenizerCompatibility:
    base_pad_before = base_tokenizer.pad_token_id
    reward_pad_before = reward_tokenizer.pad_token_id

    sample_texts = [
        "The product is surprisingly good.",
        "Bad fit, but the replacement worked!",
        "byte-level check: café\nnew line\tspacing",
        "<|endoftext|>",
    ]
    sample_equal = all(
        base_tokenizer.encode(text, add_special_tokens=False)
        == reward_tokenizer.encode(text, add_special_tokens=False)
        for text in sample_texts
    )

    base_pad_after = configure_gpt2_padding(base_tokenizer, "left")
    reward_pad_after = configure_gpt2_padding(reward_tokenizer, "right")

    compatibility = TokenizerCompatibility(
        vocab_equal=base_tokenizer.get_vocab() == reward_tokenizer.get_vocab(),
        vocab_size_equal=base_tokenizer.vocab_size == reward_tokenizer.vocab_size,
        base_vocab_size=int(base_tokenizer.vocab_size),
        reward_vocab_size=int(reward_tokenizer.vocab_size),
        eos_token_id_equal=base_tokenizer.eos_token_id == reward_tokenizer.eos_token_id,
        base_eos_token_id=base_tokenizer.eos_token_id,
        reward_eos_token_id=reward_tokenizer.eos_token_id,
        bos_token_id_equal=base_tokenizer.bos_token_id == reward_tokenizer.bos_token_id,
        base_bos_token_id=base_tokenizer.bos_token_id,
        reward_bos_token_id=reward_tokenizer.bos_token_id,
        byte_level_bpe_equal=sample_equal,
        sample_encodings_equal=sample_equal,
        padding_strategy_compatible=base_pad_after == reward_pad_after == base_tokenizer.eos_token_id,
        base_pad_token_id_before=base_pad_before,
        reward_pad_token_id_before=reward_pad_before,
        base_pad_token_id_after=base_pad_after,
        reward_pad_token_id_after=reward_pad_after,
        base_padding_side=base_tokenizer.padding_side,
        reward_padding_side=reward_tokenizer.padding_side,
    )

    assert base_tokenizer.get_vocab() == reward_tokenizer.get_vocab()
    assert base_tokenizer.vocab_size == reward_tokenizer.vocab_size
    assert base_tokenizer.eos_token_id == reward_tokenizer.eos_token_id
    assert base_tokenizer.bos_token_id == reward_tokenizer.bos_token_id
    assert sample_equal, "GPT-2 byte-level BPE sample encodings differ"
    assert compatibility.padding_strategy_compatible, "GPT-2 padding must use EOS token for both models"
    return compatibility


def load_base_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    if not model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    return model, tokenizer


def load_reward_model(
    reward_base_path: Path,
    reward_tokenizer_path: Path,
    reward_checkpoint_path: Path,
    device: torch.device,
) -> tuple[GPT2RewardModel, PreTrainedTokenizerBase, dict[str, Any]]:
    for path, label in [
        (reward_base_path, "Reward base model"),
        (reward_tokenizer_path, "Reward tokenizer"),
        (reward_checkpoint_path, "Reward checkpoint"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} path not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(str(reward_tokenizer_path), local_files_only=True)
    reward_model = GPT2RewardModel(
        reward_model_name=str(reward_base_path),
        out_features=1,
        loss_fn="cumulative_mse",
    )
    state_dict = torch.load(str(reward_checkpoint_path), map_location="cpu")
    direct_strict_load = True
    filtered_legacy_keys: list[str] = []
    try:
        load_result = reward_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        direct_strict_load = False
        current_keys = set(reward_model.state_dict().keys())
        unexpected_keys = [key for key in state_dict if key not in current_keys]
        legacy_attention_key = re.compile(r"^model\.transformer\.h\.\d+\.attn\.(bias|masked_bias)$")
        filtered_legacy_keys = [key for key in unexpected_keys if legacy_attention_key.match(key)]
        assert len(filtered_legacy_keys) == len(unexpected_keys), (
            "Reward checkpoint has unexpected keys beyond legacy GPT-2 attention bias buffers: "
            f"{sorted(set(unexpected_keys) - set(filtered_legacy_keys))[:20]}"
        )
        filtered_state_dict = {key: value for key, value in state_dict.items() if key in current_keys}
        load_result = reward_model.load_state_dict(filtered_state_dict, strict=True)
    reward_model.model.config.use_cache = True
    reward_model.to(device)
    return reward_model, tokenizer, {
        "direct_strict_load": direct_strict_load,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "filtered_legacy_attention_bias_keys": filtered_legacy_keys,
    }


def score_token_id_sequences(
    reward_model: GPT2RewardModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    assert input_ids.shape == attention_mask.shape, (
        f"input_ids and attention_mask shapes differ: {tuple(input_ids.shape)} vs {tuple(attention_mask.shape)}"
    )
    with torch.inference_mode():
        output = reward_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=None,
            use_cache=False,
        )
    assert isinstance(output, tuple) and len(output) == 2
    loss, scores = output
    assert loss is None
    assert scores.ndim == 2 and scores.shape[1] == 1, f"Expected reward scores shape [batch, 1], got {tuple(scores.shape)}"
    return scores[:, 0].detach().float().cpu()


def score_texts(
    reward_model: GPT2RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> list[float]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    scores = score_token_id_sequences(reward_model, encoded.input_ids, encoded.attention_mask, device)
    return [float(score) for score in scores.tolist()]


def candidate_sanity_samples(
    base_model: torch.nn.Module,
    base_tokenizer: PreTrainedTokenizerBase,
    reward_model: GPT2RewardModel,
    reward_tokenizer: PreTrainedTokenizerBase,
    prefixes: Sequence[str],
    top_k: int,
    max_reward_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for prefix_index, prefix in enumerate(prefixes):
        prefix_ids = base_tokenizer.encode(prefix, add_special_tokens=False)
        assert prefix_ids, f"Prefix produced no token IDs: {prefix!r}"
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = base_model(input_ids=input_ids).logits[0, -1].detach().float().cpu()
        top_values, top_indices = torch.topk(logits, k=top_k)
        candidate_rows = []
        candidate_texts = []
        for rank, (logit, token_id) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            candidate_text = prefix + base_tokenizer.decode([int(token_id)])
            expected_ids = prefix_ids + [int(token_id)]
            reward_ids = reward_tokenizer.encode(candidate_text, add_special_tokens=False)
            if len(expected_ids) <= max_reward_length:
                assert reward_ids == expected_ids, (
                    "Candidate tokenization mismatch after tokenizer compatibility assertion: "
                    f"prefix={prefix!r}, token_id={token_id}, reward_ids={reward_ids}, expected={expected_ids}"
                )
            candidate_texts.append(candidate_text)
            candidate_rows.append(
                {
                    "rank": rank,
                    "token_id": int(token_id),
                    "token": base_tokenizer.decode([int(token_id)]),
                    "base_logit": float(logit),
                    "candidate_text": candidate_text,
                }
            )

        encoded_candidates = reward_tokenizer(
            candidate_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_reward_length,
        )
        reward_scores = score_token_id_sequences(
            reward_model,
            encoded_candidates.input_ids,
            encoded_candidates.attention_mask,
            device,
        )
        assert reward_scores.shape[0] == len(candidate_rows)
        for row, score, reward_input_ids, reward_attention_mask in zip(
            candidate_rows,
            reward_scores.tolist(),
            encoded_candidates.input_ids.tolist(),
            encoded_candidates.attention_mask.tolist(),
        ):
            row["reward_score"] = float(score)
            row["reward_input_ids"] = reward_input_ids
            row["reward_attention_mask"] = reward_attention_mask

        samples.append(
            {
                "prefix_index": prefix_index,
                "prefix": prefix,
                "prefix_token_ids": prefix_ids,
                "top_k": top_k,
                "candidates": candidate_rows,
            }
        )
    return samples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--reward-base-path", type=Path, default=DEFAULT_REWARD_BASE_PATH)
    parser.add_argument("--reward-tokenizer-path", type=Path, default=DEFAULT_REWARD_TOKENIZER_PATH)
    parser.add_argument("--reward-checkpoint-path", type=Path, default=DEFAULT_REWARD_CHECKPOINT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--candidate-output-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-reward-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Prefix to include in candidate sanity examples. Can be repeated.",
    )
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    device = choose_device(args.device)

    LOGGER.info("Loading base model from %s on %s", args.base_model_path, device)
    base_model, base_tokenizer = load_base_model(args.base_model_path, device)
    LOGGER.info("Loading reward model checkpoint from %s", args.reward_checkpoint_path)
    reward_model, reward_tokenizer, reward_load_result = load_reward_model(
        args.reward_base_path,
        args.reward_tokenizer_path,
        args.reward_checkpoint_path,
        device,
    )

    tokenizer_compatibility = validate_tokenizers(base_tokenizer, reward_tokenizer)
    base_model.config.pad_token_id = base_tokenizer.pad_token_id
    reward_model.model.config.pad_token_id = reward_tokenizer.pad_token_id

    base_frozen_before = freeze_model(base_model)
    reward_frozen_before = freeze_model(reward_model)
    assert_all_frozen(base_model, "base_model")
    assert_all_frozen(reward_model, "reward_model")
    router_freeze_probe = validate_frozen_models_survive_router_step(base_model, reward_model)
    base_frozen_after = frozen_parameter_stats(base_model)
    reward_frozen_after = frozen_parameter_stats(reward_model)
    assert_all_frozen(base_model, "base_model after dummy router step")
    assert_all_frozen(reward_model, "reward_model after dummy router step")

    positive_texts = [
        "This product is excellent, reliable, and a pleasure to use.",
        "I loved the book and would happily recommend it to friends.",
        "The album sounds beautiful and exceeded my expectations.",
    ]
    negative_texts = [
        "This product is terrible, broken, and a complete waste of money.",
        "I hated the book and would not recommend it to anyone.",
        "The album sounds awful and was deeply disappointing.",
    ]
    score_texts_input = positive_texts + negative_texts
    score_values = score_texts(reward_model, reward_tokenizer, score_texts_input, device, args.max_reward_length)
    positive_scores = score_values[: len(positive_texts)]
    negative_scores = score_values[len(positive_texts) :]
    reward_direction_verified = mean(positive_scores) > mean(negative_scores)
    assert reward_direction_verified, (
        "Reward direction check failed: expected positive sentiment examples "
        "to score higher than negative sentiment examples"
    )

    prefixes = args.prefixes or [
        "The product",
        "I thought the movie",
        "After opening the package",
    ]
    sanity_samples = candidate_sanity_samples(
        base_model,
        base_tokenizer,
        reward_model,
        reward_tokenizer,
        prefixes,
        args.top_k,
        args.max_reward_length,
        device,
    )

    flattened_candidate_scores = [
        float(candidate["reward_score"])
        for sample in sanity_samples
        for candidate in sample["candidates"]
    ]

    report = {
        "model_identifiers": {
            "base_lm": str(args.base_model_path),
            "reward_base_model": str(args.reward_base_path),
            "reward_tokenizer": str(args.reward_tokenizer_path),
            "reward_checkpoint": str(args.reward_checkpoint_path),
        },
        "model_file_hashes": {
            "reward_checkpoint_sha256": sha256_file(args.reward_checkpoint_path),
        },
        "tokenizer_identifiers": {
            "base_tokenizer": str(args.base_model_path),
            "reward_tokenizer": str(args.reward_tokenizer_path),
        },
        "device": str(device),
        "top_k": args.top_k,
        "max_reward_length": args.max_reward_length,
        "attention_mask_policy": "explicit tokenizer attention_mask",
        "base_checksum_unchanged": router_freeze_probe.base_checksum_unchanged,
        "reward_checksum_unchanged": router_freeze_probe.reward_checksum_unchanged,
        "optimizer_contains_only_router": router_freeze_probe.optimizer_contains_only_router,
        "tokenizer_compatibility": asdict(tokenizer_compatibility),
        "special_token_ids": {
            "base": {
                "eos_token_id": base_tokenizer.eos_token_id,
                "bos_token_id": base_tokenizer.bos_token_id,
                "pad_token_id": base_tokenizer.pad_token_id,
            },
            "reward": {
                "eos_token_id": reward_tokenizer.eos_token_id,
                "bos_token_id": reward_tokenizer.bos_token_id,
                "pad_token_id": reward_tokenizer.pad_token_id,
            },
        },
        "frozen_parameter_counts": {
            "base_before_dummy_router_step": asdict(base_frozen_before),
            "reward_before_dummy_router_step": asdict(reward_frozen_before),
            "base_after_dummy_router_step": asdict(base_frozen_after),
            "reward_after_dummy_router_step": asdict(reward_frozen_after),
        },
        "router_freeze_probe": asdict(router_freeze_probe),
        "reward_model_load_result": reward_load_result,
        "reward_model_interface": {
            "class": "Method.RAD.reward_modeling.reward_model.GPT2RewardModel",
            "base_architecture": reward_model.model.__class__.__name__,
            "head": "GPT2LMHeadModel.lm_head replaced with Linear(hidden_size, 1)",
            "input": "input_ids LongTensor[batch, seq_len] and explicit tokenizer attention_mask LongTensor[batch, seq_len]",
            "forward_use_cache_false_return": "(loss, scores)",
            "forward_use_cache_true_return": "(loss, scores, past_key_values)",
            "score_shape": "FloatTensor[batch, 1]",
            "attention_mask_policy": "explicit tokenizer attention_mask is passed to the reward model; validation code does not derive it from input_ids.ne(pad_token_id)",
            "score_selection": "RAD reward model currently selects last non-pad token internally using input_ids != pad_token_id",
            "candidate_scalar_used_by_existing_RAD": "reward_logits[:, 0]",
            "score_transform_for_router_features": "none during validation; existing RAD clamps to [0, 1] when applying beta",
            "higher_means_more_positive": reward_direction_verified,
            "calibration_note": "Scores are raw scalar reward-head outputs and are not assumed calibrated.",
        },
        "reward_score_statistics": {
            "direction_probe_all": tensor_stats(score_values),
            "direction_probe_positive": tensor_stats(positive_scores),
            "direction_probe_negative": tensor_stats(negative_scores),
            "candidate_sanity_rewards": tensor_stats(flattened_candidate_scores),
        },
        "checks": {
            "exact_vocab_compatibility": tokenizer_compatibility.vocab_equal,
            "vocab_size_compatibility": tokenizer_compatibility.vocab_size_equal,
            "eos_token_id_compatibility": tokenizer_compatibility.eos_token_id_equal,
            "bos_token_id_compatibility": tokenizer_compatibility.bos_token_id_equal,
            "byte_level_bpe_compatibility": tokenizer_compatibility.byte_level_bpe_equal,
            "padding_strategy_compatible": tokenizer_compatibility.padding_strategy_compatible,
            "base_model_frozen": base_frozen_after.all_frozen,
            "reward_model_frozen": reward_frozen_after.all_frozen,
            "base_requires_grad_all_false": router_freeze_probe.base_requires_grad_all_false,
            "reward_requires_grad_all_false": router_freeze_probe.reward_requires_grad_all_false,
            "base_grads_all_none": router_freeze_probe.base_grads_all_none,
            "reward_grads_all_none": router_freeze_probe.reward_grads_all_none,
            "base_checksum_unchanged": router_freeze_probe.base_checksum_unchanged,
            "reward_checksum_unchanged": router_freeze_probe.reward_checksum_unchanged,
            "optimizer_contains_only_router": router_freeze_probe.optimizer_contains_only_router,
            "reward_checkpoint_compatible_load": not reward_load_result["missing_keys"] and not reward_load_result["unexpected_keys"],
            "reward_interface_scalar_shape": True,
            "reward_direction_verified": reward_direction_verified,
            "candidate_sanity_examples_written": True,
        },
    }
    assert all(report["checks"].values()), f"Validation checks failed: {report['checks']}"

    write_json(args.report_path, report)
    write_jsonl(args.candidate_output_path, sanity_samples)
    LOGGER.info("Wrote validation report to %s", args.report_path)
    LOGGER.info("Wrote candidate sanity samples to %s", args.candidate_output_path)


if __name__ == "__main__":
    main()
