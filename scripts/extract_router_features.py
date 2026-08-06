#!/usr/bin/env python3
"""Extract offline token-level RAD router features into resumable tensor shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Method.RAD.reward_modeling.reward_model import GPT2RewardModel
from scripts.validate_models import (
    configure_gpt2_padding,
    freeze_model,
    load_reward_model,
    validate_tokenizers,
)

LOGGER = logging.getLogger("router_feature_cache")

DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "RAD_train" / "router_amazon_polarity"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cache" / "router_features"
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_REWARD_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_REWARD_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_REWARD_CHECKPOINT_PATH = DEFAULT_REWARD_TOKENIZER_PATH / "pytorch_model.bin"
SCHEMA_VERSION = 2
REWARD_TRANSFORM_FORMULA = "rad_reward_scores = clamp(raw_reward_scores, 0, 1); if inverse then 1 - clamped"


@dataclass(frozen=True)
class RouterExample:
    sample_id: str
    prompt: str
    continuation: str


@dataclass(frozen=True)
class TokenStepFeature:
    sample_id: str
    position: int
    candidate_ids: torch.Tensor
    base_logits: torch.Tensor
    raw_reward_scores: torch.Tensor
    rad_reward_scores: torch.Tensor
    gold_token_id: int
    gold_index: int
    gold_was_in_topk: bool
    attention_length: int
    reward_effective_length: torch.Tensor
    reward_transform_name: str
    original_gold_rank: int
    base_entropy: float
    reward_range: float


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add_values(self, values: Iterable[float]) -> None:
        for value in values:
            value = float(value)
            self.count += 1
            self.total += value
            self.total_sq += value * value
            self.min_value = value if self.min_value is None else min(self.min_value, value)
            self.max_value = value if self.max_value is None else max(self.max_value, value)

    def add_tensor(self, tensor: torch.Tensor) -> None:
        self.add_values(float(value) for value in tensor.detach().float().cpu().flatten().tolist())

    def to_summary(self) -> dict[str, float | None]:
        if self.count == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean_value = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean_value * mean_value), 0.0)
        return {
            "mean": float(mean_value),
            "std": float(math.sqrt(variance)),
            "min": float(self.min_value) if self.min_value is not None else None,
            "max": float(self.max_value) if self.max_value is not None else None,
        }

    def to_manifest(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total": self.total,
            "total_sq": self.total_sq,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_manifest(cls, payload: dict[str, Any] | None) -> "RunningStats":
        if not payload:
            return cls()
        return cls(
            count=int(payload.get("count", 0)),
            total=float(payload.get("total", 0.0)),
            total_sq=float(payload.get("total_sq", 0.0)),
            min_value=payload.get("min_value"),
            max_value=payload.get("max_value"),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_mapping_hash(tokenizer: PreTrainedTokenizerBase) -> str:
    payload = {
        "vocab": sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def reward_transform_name(inverse: bool) -> str:
    return "rad_clamp_0_1_inverse" if inverse else "rad_clamp_0_1"


def transform_reward_scores(raw_reward_scores: torch.Tensor, inverse: bool) -> torch.Tensor:
    transformed = torch.clamp(raw_reward_scores.float(), min=0.0, max=1.0)
    if inverse:
        transformed = 1.0 - transformed
    return transformed


def stats_from_existing_shards(shards: Sequence[dict[str, Any]]) -> tuple[RunningStats, RunningStats, RunningStats, int]:
    base_stats = RunningStats()
    reward_stats = RunningStats()
    continuation_stats = RunningStats()
    gold_in_topk_count = 0
    per_sample_counts: dict[str, int] = {}
    for shard in shards:
        payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
        base_stats.add_tensor(payload["base_logits"])
        reward_stats.add_tensor(payload["rad_reward_scores"])
        gold_in_topk_count += int(payload["gold_was_in_topk"].sum().item())
        for sample_id in payload["sample_ids"]:
            sample_id = str(sample_id)
            per_sample_counts[sample_id] = per_sample_counts.get(sample_id, 0) + 1
    continuation_stats.add_values(per_sample_counts.values())
    return base_stats, reward_stats, continuation_stats, gold_in_topk_count


def read_jsonl(path: Path, max_samples: int | None = None) -> list[RouterExample]:
    rows: list[RouterExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                rows.append(
                    RouterExample(
                        sample_id=str(row["id"]),
                        prompt=str(row["prompt"]),
                        continuation=str(row["continuation"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing required field in {path}:{line_number}: {exc}") from exc
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def next_shard_index(split_dir: Path) -> int:
    indices = []
    for path in split_dir.glob("shard_*.pt"):
        try:
            indices.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0


def append_gold_to_topk(
    logits: torch.Tensor,
    gold_token_id: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, int]:
    top_values, top_indices = torch.topk(logits.detach().float().cpu(), k=top_k)
    matches = (top_indices == int(gold_token_id)).nonzero(as_tuple=False).flatten()
    if len(matches) > 0:
        gold_index = int(matches[0].item())
        return top_indices.long(), top_values.float(), gold_index, True, gold_index

    candidate_ids = top_indices.clone().long()
    base_logits = top_values.clone().float()
    candidate_ids[-1] = int(gold_token_id)
    base_logits[-1] = logits[int(gold_token_id)].detach().float().cpu()
    return candidate_ids, base_logits, top_k - 1, False, -1


def score_candidate_sequences(
    reward_model: torch.nn.Module,
    candidate_sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    assert candidate_sequences.shape == attention_mask.shape
    with torch.inference_mode():
        output = reward_model(
            input_ids=candidate_sequences.to(device),
            attention_mask=attention_mask.to(device),
            labels=None,
            use_cache=False,
        )
    assert isinstance(output, tuple) and len(output) == 2
    loss, scores = output
    assert loss is None
    assert scores.ndim == 2 and scores.shape[1] == 1
    return scores[:, 0].detach().float().cpu()


def pad_token_id_for_reward_model(reward_model: torch.nn.Module) -> int:
    return int(getattr(reward_model, "pad_token_id", 0))


def pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in sequences)
    padded = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row_index, row in enumerate(sequences):
        padded[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    return padded, attention_mask


def score_many_candidate_sequences(
    reward_model: torch.nn.Module,
    candidate_sequences: Sequence[Sequence[int]],
    reward_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_token_id = pad_token_id_for_reward_model(reward_model)
    all_scores: list[torch.Tensor] = []
    all_effective_lengths: list[torch.Tensor] = []
    for start in range(0, len(candidate_sequences), reward_batch_size):
        batch_sequences = candidate_sequences[start : start + reward_batch_size]
        padded, attention_mask = pad_sequences(batch_sequences, pad_token_id)
        all_scores.append(score_candidate_sequences(reward_model, padded, attention_mask, device))
        all_effective_lengths.append(attention_mask.sum(dim=1).long())
    return torch.cat(all_scores, dim=0), torch.cat(all_effective_lengths, dim=0)


def compute_token_step_feature(
    base_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    context_ids: Sequence[int],
    gold_token_id: int,
    sample_id: str,
    position: int,
    top_k: int,
    max_reward_length: int,
    inverse: bool,
    device: torch.device,
) -> TokenStepFeature:
    assert context_ids, f"Empty context for {sample_id}:{position}"
    context_tensor = torch.tensor([list(context_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        next_logits = base_model(input_ids=context_tensor).logits[0, -1].detach().float().cpu()
    candidate_ids, base_logits, gold_index, gold_was_in_topk, original_gold_rank = append_gold_to_topk(
        next_logits,
        int(gold_token_id),
        top_k,
    )
    assert candidate_ids.shape == (top_k,)
    assert base_logits.shape == (top_k,)
    assert int(candidate_ids[gold_index].item()) == int(gold_token_id)

    candidate_sequences = [
        (list(context_ids) + [int(candidate_id)])[-max_reward_length:]
        for candidate_id in candidate_ids.tolist()
    ]
    padded, attention_mask = pad_sequences(candidate_sequences, pad_token_id_for_reward_model(reward_model))
    reward_scores = score_candidate_sequences(reward_model, padded, attention_mask, device)
    assert reward_scores.shape == (top_k,)
    rad_reward_scores = transform_reward_scores(reward_scores, inverse)
    reward_effective_length = attention_mask.sum(dim=1).long()

    probs = torch.softmax(next_logits.float(), dim=-1)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    reward_range = float((rad_reward_scores.max() - rad_reward_scores.min()).item())
    for tensor_name, tensor in [
        ("base_logits", base_logits),
        ("raw_reward_scores", reward_scores),
        ("rad_reward_scores", rad_reward_scores),
    ]:
        assert torch.isfinite(tensor).all(), f"{tensor_name} contains NaN or infinity at {sample_id}:{position}"

    return TokenStepFeature(
        sample_id=sample_id,
        position=position,
        candidate_ids=candidate_ids.long(),
        base_logits=base_logits.float(),
        raw_reward_scores=reward_scores.float(),
        rad_reward_scores=rad_reward_scores.float(),
        gold_token_id=int(gold_token_id),
        gold_index=int(gold_index),
        gold_was_in_topk=bool(gold_was_in_topk),
        attention_length=int(len(context_ids)),
        reward_effective_length=reward_effective_length,
        reward_transform_name=reward_transform_name(inverse),
        original_gold_rank=int(original_gold_rank),
        base_entropy=entropy,
        reward_range=reward_range,
    )


def iter_example_features(
    example: RouterExample,
    base_model: torch.nn.Module,
    base_tokenizer: PreTrainedTokenizerBase,
    reward_model: torch.nn.Module,
    top_k: int,
    max_reward_length: int,
    inverse: bool,
    reward_batch_size: int,
    device: torch.device,
) -> Iterator[TokenStepFeature]:
    prompt_ids = base_tokenizer.encode(example.prompt, add_special_tokens=False)
    continuation_ids = base_tokenizer.encode(example.continuation, add_special_tokens=False)
    assert prompt_ids, f"Prompt has no tokens: {example.sample_id}"
    assert continuation_ids, f"Continuation has no tokens: {example.sample_id}"

    full_ids = prompt_ids + continuation_ids
    model_input_ids = full_ids[:-1]
    input_tensor = torch.tensor([model_input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        all_logits = base_model(input_ids=input_tensor).logits[0].detach().float().cpu()

    pending: list[dict[str, Any]] = []
    candidate_sequences: list[list[int]] = []
    for position, gold_token_id in enumerate(continuation_ids):
        context_ids = full_ids[: len(prompt_ids) + position]
        logit_index = len(prompt_ids) - 1 + position
        next_logits = all_logits[logit_index]
        candidate_ids, base_logits, gold_index, gold_was_in_topk, original_gold_rank = append_gold_to_topk(
            next_logits,
            int(gold_token_id),
            top_k,
        )
        assert int(candidate_ids[gold_index].item()) == int(gold_token_id)
        for candidate_id in candidate_ids.tolist():
            candidate_sequences.append((context_ids + [int(candidate_id)])[-max_reward_length:])
        probs = torch.softmax(next_logits.float(), dim=-1)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        pending.append(
            {
                "position": position,
                "candidate_ids": candidate_ids,
                "base_logits": base_logits,
                "gold_token_id": int(gold_token_id),
                "gold_index": int(gold_index),
                "gold_was_in_topk": bool(gold_was_in_topk),
                "attention_length": int(len(context_ids)),
                "original_gold_rank": int(original_gold_rank),
                "base_entropy": entropy,
            }
        )

    raw_scores_flat, reward_effective_lengths_flat = score_many_candidate_sequences(
        reward_model,
        candidate_sequences,
        reward_batch_size,
        device,
    )
    for index, item in enumerate(pending):
        start = index * top_k
        end = start + top_k
        raw_reward_scores = raw_scores_flat[start:end].float()
        rad_reward_scores = transform_reward_scores(raw_reward_scores, inverse)
        reward_effective_length = reward_effective_lengths_flat[start:end].long()
        for tensor_name, tensor in [
            ("base_logits", item["base_logits"]),
            ("raw_reward_scores", raw_reward_scores),
            ("rad_reward_scores", rad_reward_scores),
        ]:
            assert torch.isfinite(tensor).all(), f"{tensor_name} contains NaN or infinity at {example.sample_id}:{item['position']}"
        yield TokenStepFeature(
            sample_id=example.sample_id,
            position=item["position"],
            candidate_ids=item["candidate_ids"].long(),
            base_logits=item["base_logits"].float(),
            raw_reward_scores=raw_reward_scores,
            rad_reward_scores=rad_reward_scores,
            gold_token_id=item["gold_token_id"],
            gold_index=item["gold_index"],
            gold_was_in_topk=item["gold_was_in_topk"],
            attention_length=item["attention_length"],
            reward_effective_length=reward_effective_length,
            reward_transform_name=reward_transform_name(inverse),
            original_gold_rank=item["original_gold_rank"],
            base_entropy=item["base_entropy"],
            reward_range=float((rad_reward_scores.max() - rad_reward_scores.min()).item()),
        )
        context_ids.append(int(gold_token_id))


def shard_payload(records: Sequence[TokenStepFeature], split: str, shard_index: int, top_k: int) -> dict[str, Any]:
    assert records, "Cannot write empty shard"
    for record in records:
        assert record.candidate_ids.shape == (top_k,)
        assert record.base_logits.shape == (top_k,)
        assert record.raw_reward_scores.shape == (top_k,)
        assert record.rad_reward_scores.shape == (top_k,)
        assert record.reward_effective_length.shape == (top_k,)
        assert int(record.candidate_ids[record.gold_index].item()) == record.gold_token_id
        assert torch.isfinite(record.base_logits).all()
        assert torch.isfinite(record.raw_reward_scores).all()
        assert torch.isfinite(record.rad_reward_scores).all()
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "shard_index": shard_index,
        "top_k": top_k,
        "num_records": len(records),
        "reward_transform_name": records[0].reward_transform_name,
        "sample_ids": [record.sample_id for record in records],
        "position": torch.tensor([record.position for record in records], dtype=torch.long),
        "candidate_ids": torch.stack([record.candidate_ids for record in records]).long(),
        "base_logits": torch.stack([record.base_logits for record in records]).float(),
        "raw_reward_scores": torch.stack([record.raw_reward_scores for record in records]).float(),
        "rad_reward_scores": torch.stack([record.rad_reward_scores for record in records]).float(),
        "gold_token_id": torch.tensor([record.gold_token_id for record in records], dtype=torch.long),
        "gold_index": torch.tensor([record.gold_index for record in records], dtype=torch.long),
        "gold_was_in_topk": torch.tensor([record.gold_was_in_topk for record in records], dtype=torch.bool),
        "attention_length": torch.tensor([record.attention_length for record in records], dtype=torch.long),
        "reward_effective_length": torch.stack([record.reward_effective_length for record in records]).long(),
        "reward_transform_name_per_record": [record.reward_transform_name for record in records],
        "original_gold_rank": torch.tensor([record.original_gold_rank for record in records], dtype=torch.long),
        "base_entropy": torch.tensor([record.base_entropy for record in records], dtype=torch.float),
        "reward_range": torch.tensor([record.reward_range for record in records], dtype=torch.float),
    }


def save_shard(split_dir: Path, records: Sequence[TokenStepFeature], split: str, shard_index: int, top_k: int) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"shard_{shard_index:05d}.pt"
    tmp_path = split_dir / f"shard_{shard_index:05d}.pt.tmp"
    payload = shard_payload(records, split, shard_index, top_k)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "num_records": len(records),
        "sample_ids": sorted(set(record.sample_id for record in records)),
    }


def completed_sample_ids_from_manifest(manifest: dict[str, Any] | None) -> set[str]:
    if not manifest:
        return set()
    return set(str(sample_id) for sample_id in manifest.get("completed_sample_ids", []))


def cache_config(args: argparse.Namespace, source_hash: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset_hash": source_hash,
        "top_k": args.top_k,
        "max_reward_length": args.max_reward_length,
        "base_model_path": str(args.base_model_path),
        "reward_base_path": str(args.reward_base_path),
        "reward_tokenizer_path": str(args.reward_tokenizer_path),
        "reward_checkpoint_path": str(args.reward_checkpoint_path),
        "reward_checkpoint_sha256": args.reward_checkpoint_sha256,
        "base_tokenizer_mapping_sha256": args.base_tokenizer_mapping_sha256,
        "reward_tokenizer_mapping_sha256": args.reward_tokenizer_mapping_sha256,
        "reward_transform_name": reward_transform_name(args.inverse),
        "inverse": bool(args.inverse),
    }


def verify_existing_shards(manifest: dict[str, Any]) -> None:
    for shard in manifest.get("shards", []):
        path = Path(shard["path"])
        if not path.exists():
            raise FileNotFoundError(f"Manifest references missing shard: {path}")
        actual_hash = sha256_file(path)
        expected_hash = str(shard.get("sha256", ""))
        if actual_hash != expected_hash:
            raise ValueError(f"Shard hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")


def validate_resume_compatibility(existing_manifest: dict[str, Any], expected_config: dict[str, Any]) -> None:
    existing_config = existing_manifest.get("cache_config", {})
    checks = [
        ("schema_version", existing_manifest.get("schema_version"), SCHEMA_VERSION),
        ("source_dataset_hash", existing_manifest.get("source_dataset_hash"), expected_config["source_dataset_hash"]),
        ("top_k", existing_manifest.get("top_k"), expected_config["top_k"]),
        ("max_reward_length", existing_manifest.get("max_reward_length"), expected_config["max_reward_length"]),
        ("base_model_path", existing_config.get("base_model_path"), expected_config["base_model_path"]),
        ("reward_base_path", existing_config.get("reward_base_path"), expected_config["reward_base_path"]),
        ("reward_tokenizer_path", existing_config.get("reward_tokenizer_path"), expected_config["reward_tokenizer_path"]),
        ("reward_checkpoint_path", existing_config.get("reward_checkpoint_path"), expected_config["reward_checkpoint_path"]),
        ("reward_checkpoint_sha256", existing_config.get("reward_checkpoint_sha256"), expected_config["reward_checkpoint_sha256"]),
        ("base_tokenizer_mapping_sha256", existing_config.get("base_tokenizer_mapping_sha256"), expected_config["base_tokenizer_mapping_sha256"]),
        ("reward_tokenizer_mapping_sha256", existing_config.get("reward_tokenizer_mapping_sha256"), expected_config["reward_tokenizer_mapping_sha256"]),
        ("reward_transform_name", existing_config.get("reward_transform_name"), expected_config["reward_transform_name"]),
        ("inverse", existing_config.get("inverse"), expected_config["inverse"]),
    ]
    mismatches = [
        {"field": field, "existing": existing, "expected": expected}
        for field, existing, expected in checks
        if existing != expected
    ]
    if mismatches:
        raise ValueError(f"Resume cache is incompatible with current config: {json.dumps(mismatches, sort_keys=True)}")
    verify_existing_shards(existing_manifest)


def online_recompute_record(
    base_model: torch.nn.Module,
    base_tokenizer: PreTrainedTokenizerBase,
    reward_model: torch.nn.Module,
    example: RouterExample,
    position: int,
    top_k: int,
    max_reward_length: int,
    inverse: bool,
    device: torch.device,
) -> TokenStepFeature:
    prompt_ids = base_tokenizer.encode(example.prompt, add_special_tokens=False)
    continuation_ids = base_tokenizer.encode(example.continuation, add_special_tokens=False)
    assert 0 <= position < len(continuation_ids)
    context_ids = prompt_ids + continuation_ids[:position]
    return compute_token_step_feature(
        base_model=base_model,
        reward_model=reward_model,
        context_ids=context_ids,
        gold_token_id=int(continuation_ids[position]),
        sample_id=example.sample_id,
        position=position,
        top_k=top_k,
        max_reward_length=max_reward_length,
        inverse=inverse,
        device=device,
    )


def assert_record_matches_recompute(
    cached: dict[str, Any],
    row_index: int,
    recomputed: TokenStepFeature,
    atol: float,
) -> None:
    assert str(cached["sample_ids"][row_index]) == recomputed.sample_id
    assert int(cached["position"][row_index].item()) == recomputed.position
    assert torch.equal(cached["candidate_ids"][row_index].cpu(), recomputed.candidate_ids.cpu())
    assert int(cached["gold_token_id"][row_index].item()) == recomputed.gold_token_id
    assert int(cached["gold_index"][row_index].item()) == recomputed.gold_index
    assert bool(cached["gold_was_in_topk"][row_index].item()) == recomputed.gold_was_in_topk
    assert int(cached["attention_length"][row_index].item()) == recomputed.attention_length
    assert torch.equal(cached["reward_effective_length"][row_index].cpu(), recomputed.reward_effective_length.cpu())
    assert str(cached["reward_transform_name_per_record"][row_index]) == recomputed.reward_transform_name
    assert torch.allclose(cached["base_logits"][row_index].cpu(), recomputed.base_logits.cpu(), atol=atol, rtol=0.0)
    assert torch.allclose(cached["raw_reward_scores"][row_index].cpu(), recomputed.raw_reward_scores.cpu(), atol=atol, rtol=0.0)
    assert torch.allclose(cached["rad_reward_scores"][row_index].cpu(), recomputed.rad_reward_scores.cpu(), atol=atol, rtol=0.0)


def split_manifest(
    split: str,
    source_path: Path,
    source_hash: str,
    source_samples: int,
    completed_sample_ids: set[str],
    shards: list[dict[str, Any]],
    skipped_samples: int,
    failed_samples: list[dict[str, str]],
    base_logit_stats: RunningStats,
    reward_score_stats: RunningStats,
    continuation_length_stats: RunningStats,
    gold_in_topk_count: int,
    total_records: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = cache_config(args, source_hash)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "source_path": str(source_path),
        "source_dataset_hash": source_hash,
        "cache_config": config,
        "source_samples": source_samples,
        "completed_samples": len(completed_sample_ids),
        "completed_sample_ids": sorted(completed_sample_ids),
        "cached_token_steps": total_records,
        "top_k": args.top_k,
        "gold_in_original_topk_rate": (gold_in_topk_count / total_records) if total_records else None,
        "average_continuation_length": continuation_length_stats.to_summary()["mean"],
        "base_logit_statistics": base_logit_stats.to_summary(),
        "reward_score_statistics": reward_score_stats.to_summary(),
        "reward_transform_name": config["reward_transform_name"],
        "reward_transform_formula": REWARD_TRANSFORM_FORMULA,
        "reward_checkpoint_sha256": args.reward_checkpoint_sha256,
        "statistics_accumulators": {
            "base_logits": base_logit_stats.to_manifest(),
            "rad_reward_scores": reward_score_stats.to_manifest(),
            "continuation_lengths": continuation_length_stats.to_manifest(),
            "gold_in_topk_count": gold_in_topk_count,
        },
        "skipped_samples": skipped_samples,
        "failed_samples": failed_samples,
        "tokenizer_model_identifiers": {
            "base_model": str(args.base_model_path),
            "base_tokenizer": str(args.base_model_path),
            "reward_base_model": str(args.reward_base_path),
            "reward_tokenizer": str(args.reward_tokenizer_path),
            "reward_checkpoint": str(args.reward_checkpoint_path),
            "reward_checkpoint_sha256": args.reward_checkpoint_sha256,
            "base_tokenizer_mapping_sha256": args.base_tokenizer_mapping_sha256,
            "reward_tokenizer_mapping_sha256": args.reward_tokenizer_mapping_sha256,
        },
        "max_reward_length": args.max_reward_length,
        "shard_size_steps": args.shard_size_steps,
        "reward_batch_size": args.reward_batch_size,
        "shards": shards,
    }


def extract_split(
    split: str,
    examples: list[RouterExample],
    source_path: Path,
    source_hash: str,
    base_model: torch.nn.Module,
    base_tokenizer: PreTrainedTokenizerBase,
    reward_model: torch.nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"
    existing_manifest = load_manifest(manifest_path) if args.resume else None
    if existing_manifest:
        validate_resume_compatibility(existing_manifest, cache_config(args, source_hash))
    completed_sample_ids = completed_sample_ids_from_manifest(existing_manifest)
    shards: list[dict[str, Any]] = list(existing_manifest.get("shards", [])) if existing_manifest else []
    total_records = int(existing_manifest.get("cached_token_steps", 0)) if existing_manifest else 0
    if existing_manifest and existing_manifest.get("statistics_accumulators"):
        accumulators = existing_manifest["statistics_accumulators"]
        base_logit_stats = RunningStats.from_manifest(accumulators.get("base_logits"))
        reward_score_stats = RunningStats.from_manifest(
            accumulators.get("rad_reward_scores", accumulators.get("reward_scores"))
        )
        continuation_length_stats = RunningStats.from_manifest(accumulators.get("continuation_lengths"))
        gold_in_topk_count = int(accumulators.get("gold_in_topk_count", 0))
    elif existing_manifest:
        base_logit_stats, reward_score_stats, continuation_length_stats, gold_in_topk_count = stats_from_existing_shards(shards)
    else:
        base_logit_stats = RunningStats()
        reward_score_stats = RunningStats()
        continuation_length_stats = RunningStats()
        gold_in_topk_count = 0
    failed_samples: list[dict[str, str]] = list(existing_manifest.get("failed_samples", [])) if existing_manifest else []
    skipped_samples = 0
    shard_index = next_shard_index(split_dir)
    pending_records: list[TokenStepFeature] = []

    for example_index, example in enumerate(examples, start=1):
        if example.sample_id in completed_sample_ids:
            skipped_samples += 1
            continue
        try:
            example_records = list(
                iter_example_features(
                    example=example,
                    base_model=base_model,
                    base_tokenizer=base_tokenizer,
                    reward_model=reward_model,
                    top_k=args.top_k,
                    max_reward_length=args.max_reward_length,
                    inverse=args.inverse,
                    reward_batch_size=args.reward_batch_size,
                    device=device,
                )
            )
        except Exception as exc:
            failed_samples.append({"sample_id": example.sample_id, "error": repr(exc)})
            LOGGER.exception("Failed sample %s", example.sample_id)
            continue

        assert example_records, f"No continuation records for {example.sample_id}"
        continuation_length_stats.add_values([len(example_records)])
        completed_sample_ids.add(example.sample_id)
        for record in example_records:
            gold_in_topk_count += int(record.gold_was_in_topk)
            total_records += 1
            base_logit_stats.add_tensor(record.base_logits)
            reward_score_stats.add_tensor(record.rad_reward_scores)
            pending_records.append(record)

        if len(pending_records) >= args.shard_size_steps:
            shards.append(save_shard(split_dir, pending_records, split, shard_index, args.top_k))
            shard_index += 1
            pending_records = []
            manifest = split_manifest(
                split,
                source_path,
                source_hash,
                len(examples),
                completed_sample_ids,
                shards,
                skipped_samples,
                failed_samples,
                base_logit_stats,
                reward_score_stats,
                continuation_length_stats,
                gold_in_topk_count,
                total_records,
                args,
            )
            write_json(manifest_path, manifest)
            LOGGER.info("%s: completed %d/%d samples, %d token steps", split, example_index, len(examples), total_records)

    if pending_records:
        shards.append(save_shard(split_dir, pending_records, split, shard_index, args.top_k))

    manifest = split_manifest(
        split,
        source_path,
        source_hash,
        len(examples),
        completed_sample_ids,
        shards,
        skipped_samples,
        failed_samples,
        base_logit_stats,
        reward_score_stats,
        continuation_length_stats,
        gold_in_topk_count,
        total_records,
        args,
    )
    write_json(manifest_path, manifest)
    return manifest


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    return model, tokenizer


def load_models(args: argparse.Namespace, device: torch.device):
    base_model, base_tokenizer = load_base_model(args.base_model_path, device)
    reward_model, reward_tokenizer, reward_load_result = load_reward_model(
        args.reward_base_path,
        args.reward_tokenizer_path,
        args.reward_checkpoint_path,
        device,
    )
    validate_tokenizers(base_tokenizer, reward_tokenizer)
    configure_gpt2_padding(base_tokenizer, "left")
    configure_gpt2_padding(reward_tokenizer, "right")
    base_model.config.pad_token_id = base_tokenizer.pad_token_id
    reward_model.model.config.pad_token_id = reward_tokenizer.pad_token_id
    freeze_model(base_model)
    freeze_model(reward_model)
    return base_model, base_tokenizer, reward_model, reward_tokenizer, reward_load_result


def verify_manifest_integrity(
    manifest: dict[str, Any],
    examples_by_id: dict[str, RouterExample],
    base_model: torch.nn.Module,
    base_tokenizer: PreTrainedTokenizerBase,
    reward_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    candidates: list[tuple[dict[str, Any], int]] = []
    for shard in manifest.get("shards", []):
        payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
        assert payload["candidate_ids"].shape[1] == args.top_k
        row_count = int(payload["num_records"])
        for row_index in range(row_count):
            candidates.append((shard, row_index))

    rng = random.Random(args.integrity_seed)
    rng.shuffle(candidates)
    selected = candidates[: min(args.integrity_samples, len(candidates))]
    payload_cache: dict[str, dict[str, Any]] = {}
    checked_records: list[dict[str, Any]] = []
    checked_sample_ids: set[str] = set()
    checked_shards: set[str] = set()
    for shard, row_index in selected:
        shard_path = str(shard["path"])
        if shard_path not in payload_cache:
            payload_cache[shard_path] = torch.load(shard_path, map_location="cpu", weights_only=False)
        payload = payload_cache[shard_path]
        sample_id = str(payload["sample_ids"][row_index])
        example = examples_by_id[sample_id]
        position = int(payload["position"][row_index].item())
        recomputed = online_recompute_record(
            base_model,
            base_tokenizer,
            reward_model,
            example,
            position,
            args.top_k,
            args.max_reward_length,
            args.inverse,
            device,
        )
        assert_record_matches_recompute(payload, row_index, recomputed, args.integrity_atol)
        checked_sample_ids.add(sample_id)
        checked_shards.add(shard_path)
        checked_records.append({"shard": shard_path, "row_index": row_index, "sample_id": sample_id, "position": position})
    return {
        "checked_records": len(checked_records),
        "checked_samples": len(checked_sample_ids),
        "checked_shards": len(checked_shards),
        "seed": args.integrity_seed,
        "atol": args.integrity_atol,
        "records": checked_records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--reward-base-path", type=Path, default=DEFAULT_REWARD_BASE_PATH)
    parser.add_argument("--reward-tokenizer-path", type=Path, default=DEFAULT_REWARD_TOKENIZER_PATH)
    parser.add_argument("--reward-checkpoint-path", type=Path, default=DEFAULT_REWARD_CHECKPOINT_PATH)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-reward-length", type=int, default=256)
    parser.add_argument("--inverse", action="store_true", help="Apply RAD inverse reward transform 1 - clamp(raw, 0, 1).")
    parser.add_argument("--shard-size-steps", type=int, default=2048)
    parser.add_argument("--reward-batch-size", type=int, default=64)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--integrity-samples", type=int, default=3)
    parser.add_argument("--integrity-seed", type=int, default=42)
    parser.add_argument("--integrity-atol", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    assert args.top_k == 20, "Version 1 RAD router feature cache requires top-k=20"
    device = choose_device(args.device)
    LOGGER.info("Loading models on %s", device)
    base_model, base_tokenizer, reward_model, _reward_tokenizer, reward_load_result = load_models(args, device)
    args.reward_checkpoint_sha256 = sha256_file(args.reward_checkpoint_path)
    args.base_tokenizer_mapping_sha256 = tokenizer_mapping_hash(base_tokenizer)
    args.reward_tokenizer_mapping_sha256 = tokenizer_mapping_hash(_reward_tokenizer)

    top_manifest = {
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(args.output_dir),
        "top_k": args.top_k,
        "max_reward_length": args.max_reward_length,
        "reward_transform_name": reward_transform_name(args.inverse),
        "reward_transform_formula": REWARD_TRANSFORM_FORMULA,
        "reward_checkpoint_sha256": args.reward_checkpoint_sha256,
        "base_tokenizer_mapping_sha256": args.base_tokenizer_mapping_sha256,
        "reward_tokenizer_mapping_sha256": args.reward_tokenizer_mapping_sha256,
        "splits": {},
        "reward_model_load_result": reward_load_result,
    }
    for split in args.splits:
        source_path = args.input_dir / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing router data split: {source_path}")
        examples = read_jsonl(source_path, args.max_samples_per_split)
        source_hash = sha256_file(source_path)
        LOGGER.info("Extracting %s: %d samples from %s", split, len(examples), source_path)
        manifest = extract_split(
            split=split,
            examples=examples,
            source_path=source_path,
            source_hash=source_hash,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            reward_model=reward_model,
            output_dir=args.output_dir,
            args=args,
            device=device,
        )
        integrity = verify_manifest_integrity(
            manifest,
            {example.sample_id: example for example in examples},
            base_model,
            base_tokenizer,
            reward_model,
            args,
            device,
        )
        manifest["integrity_check"] = integrity
        write_json(args.output_dir / split / "manifest.json", manifest)
        top_manifest["splits"][split] = {
            "manifest_path": str(args.output_dir / split / "manifest.json"),
            "cached_token_steps": manifest["cached_token_steps"],
            "completed_samples": manifest["completed_samples"],
            "integrity_check": integrity,
        }
    write_json(args.output_dir / "manifest.json", top_manifest)
    LOGGER.info("Wrote feature-cache manifest to %s", args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
