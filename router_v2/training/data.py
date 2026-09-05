"""Sequence-preserving Stage 3 cache loader for Stage 5 training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from router_v2.cache.io import load_cache_shard, sha256_file
from router_v2.candidates import TAROTopKBatch
from router_v2.guide_model.data import RADSample, load_rad_samples


@dataclass(frozen=True)
class CachedSequence:
    sample: RADSample
    position: torch.Tensor
    base_token_ids: torch.Tensor
    base_logits: torch.Tensor
    guide_token_ids: torch.Tensor
    guide_logits: torch.Tensor
    gold_token_id: torch.Tensor
    preference: torch.Tensor | None

    @property
    def length(self) -> int:
        return int(self.position.numel())


@dataclass(frozen=True)
class CachedRouterBatch:
    sample_ids: tuple[str, ...]
    samples: tuple[RADSample, ...]
    topk: TAROTopKBatch
    gold_token_ids: torch.Tensor
    position: torch.Tensor
    valid_mask: torch.Tensor
    preference: torch.Tensor | None

    def to(self, device: torch.device) -> "CachedRouterBatch":
        return CachedRouterBatch(
            sample_ids=self.sample_ids,
            samples=self.samples,
            topk=TAROTopKBatch(
                base_token_ids=self.topk.base_token_ids.to(device),
                base_logits=self.topk.base_logits.to(device),
                reward_token_ids=self.topk.reward_token_ids.to(device),
                reward_logits=self.topk.reward_logits.to(device),
            ),
            gold_token_ids=self.gold_token_ids.to(device),
            position=self.position.to(device),
            valid_mask=self.valid_mask.to(device),
            preference=(
                self.preference.to(device)
                if self.preference is not None
                else None
            ),
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _sequence_ranges(sample_ids: Sequence[str]) -> list[tuple[int, int, str]]:
    if not sample_ids:
        raise ValueError("Cache shard contains no sample IDs")
    ranges = []
    start = 0
    current = str(sample_ids[0])
    for index in range(1, len(sample_ids) + 1):
        if index == len(sample_ids) or str(sample_ids[index]) != current:
            ranges.append((start, index, current))
            if index < len(sample_ids):
                start = index
                current = str(sample_ids[index])
    if len({item[2] for item in ranges}) != len(ranges):
        raise ValueError("A sample is split into non-contiguous cache ranges")
    return ranges


def collate_cached_sequences(
    sequences: Sequence[CachedSequence],
) -> CachedRouterBatch:
    if not sequences:
        raise ValueError("Cannot collate an empty cache sequence batch")
    batch_size = len(sequences)
    max_length = max(sequence.length for sequence in sequences)
    top_k = int(sequences[0].base_token_ids.shape[-1])
    base_token_ids = torch.zeros(batch_size, max_length, top_k, dtype=torch.long)
    guide_token_ids = torch.zeros_like(base_token_ids)
    base_logits = torch.zeros(batch_size, max_length, top_k, dtype=torch.float32)
    guide_logits = torch.zeros_like(base_logits)
    gold = torch.zeros(batch_size, max_length, dtype=torch.long)
    position = torch.zeros(batch_size, max_length, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    preferences = []
    has_preference = [sequence.preference is not None for sequence in sequences]
    if any(has_preference) and not all(has_preference):
        raise ValueError("A batch cannot mix missing and present preferences")
    for row, sequence in enumerate(sequences):
        if sequence.base_token_ids.shape != (sequence.length, top_k):
            raise ValueError("Cached base Top-K sequence has an invalid shape")
        if sequence.guide_token_ids.shape != (sequence.length, top_k):
            raise ValueError("Cached guide Top-K sequence has an invalid shape")
        length = sequence.length
        base_token_ids[row, :length] = sequence.base_token_ids
        guide_token_ids[row, :length] = sequence.guide_token_ids
        base_logits[row, :length] = sequence.base_logits
        guide_logits[row, :length] = sequence.guide_logits
        gold[row, :length] = sequence.gold_token_id
        position[row, :length] = sequence.position
        valid_mask[row, :length] = True
        if sequence.preference is not None:
            preferences.append(sequence.preference)
    preference = torch.stack(preferences) if preferences else None
    return CachedRouterBatch(
        sample_ids=tuple(sequence.sample.sample_id for sequence in sequences),
        samples=tuple(sequence.sample for sequence in sequences),
        topk=TAROTopKBatch(
            base_token_ids=base_token_ids,
            base_logits=base_logits,
            reward_token_ids=guide_token_ids,
            reward_logits=guide_logits,
        ),
        gold_token_ids=gold,
        position=position,
        valid_mask=valid_mask,
        preference=preference,
    )


class ShardedSequenceCache:
    """Iterate complete samples while loading at most one cache shard at once."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        verify_hashes: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.cache_root = Path(cache_root).resolve()
        self.split = split
        self.split_dir = self.cache_root / split
        self.manifest_path = self.split_dir / "manifest.json"
        self.manifest = _load_json(self.manifest_path)
        if (
            not self.manifest.get("complete")
            or self.manifest.get("task") != "sentiment"
            or self.manifest.get("precision") != "fp32"
            or self.manifest.get("feature_schema") != "router_v2.feature_cache.v1"
        ):
            raise ValueError(f"Invalid scientific cache manifest: {self.manifest_path}")
        if int(self.manifest.get("number_shards", 0)) <= 0:
            raise ValueError("Training cache must contain tensor shards")
        source_path = Path(self.manifest["source_dataset"]["path"])
        if sha256_file(source_path) != self.manifest["source_dataset"]["sha256"]:
            raise ValueError("Training source hash does not match cache manifest")
        source_samples = load_rad_samples(source_path)
        self.samples_by_id = {
            sample.sample_id: sample for sample in source_samples
        }
        if len(self.samples_by_id) != len(source_samples):
            raise ValueError("Training source contains duplicate sample IDs")
        self.shards = tuple(self.manifest["shards"])
        if verify_hashes:
            for shard in self.shards:
                path = self.split_dir / shard["path"]
                if sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"Cache shard hash mismatch: {path}")

    @property
    def top_k(self) -> int:
        return int(self.manifest["top_k"])

    @property
    def vocab_size(self) -> int:
        return int(self.manifest["base_tokenizer"]["length"])

    @property
    def max_position(self) -> int:
        return int(self.manifest["max_position"])

    def has_preferences(self) -> bool:
        states = set()
        for shard_info in self.shards:
            shard = load_cache_shard(self.split_dir / shard_info["path"])
            states.add(shard["preference_vector"] is not None)
        if not states:
            raise ValueError("Training cache contains no shards")
        if len(states) != 1:
            raise ValueError("Preference availability differs across cache shards")
        return states.pop()

    def iter_batches(
        self,
        *,
        batch_size: int,
        epoch: int,
        seed: int,
        shuffle: bool,
        max_samples: int | None = None,
    ) -> Iterator[CachedRouterBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epoch < 0 or seed < 0:
            raise ValueError("epoch and seed must be non-negative")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or null")
        generator = random.Random(seed + 1000003 * epoch)
        shard_infos = list(self.shards)
        if shuffle:
            generator.shuffle(shard_infos)
        yielded_samples = 0
        for shard_info in shard_infos:
            shard = load_cache_shard(self.split_dir / shard_info["path"])
            ranges = _sequence_ranges(shard["sample_ids"])
            if shuffle:
                generator.shuffle(ranges)
            for batch_start in range(0, len(ranges), batch_size):
                selected_ranges = ranges[batch_start : batch_start + batch_size]
                if max_samples is not None:
                    remaining = max_samples - yielded_samples
                    if remaining <= 0:
                        return
                    selected_ranges = selected_ranges[:remaining]
                sequences = [
                    self._sequence_from_range(shard, start, end, sample_id)
                    for start, end, sample_id in selected_ranges
                ]
                yielded_samples += len(sequences)
                yield collate_cached_sequences(sequences)
                if max_samples is not None and yielded_samples >= max_samples:
                    return

    def _sequence_from_range(
        self,
        shard: dict[str, Any],
        start: int,
        end: int,
        sample_id: str,
    ) -> CachedSequence:
        if sample_id not in self.samples_by_id:
            raise ValueError(f"Cached sample is absent from source: {sample_id}")
        position = shard["position"][start:end].clone()
        expected_position = torch.arange(end - start, dtype=torch.long)
        if not torch.equal(position, expected_position):
            raise ValueError(f"Non-contiguous positions for sample {sample_id}")
        preference_rows = shard["preference_vector"]
        preference = None
        if preference_rows is not None:
            values = preference_rows[start:end]
            if not torch.allclose(values, values[0].expand_as(values)):
                raise ValueError(
                    f"Preference changes within cached sample {sample_id}"
                )
            preference = values[0].clone()
        return CachedSequence(
            sample=self.samples_by_id[sample_id],
            position=position,
            base_token_ids=shard["base_token_ids"][start:end].clone(),
            base_logits=shard["base_logits"][start:end].float().clone(),
            guide_token_ids=shard["guide_token_ids"][start:end].clone(),
            guide_logits=shard["guide_logits"][start:end].float().clone(),
            gold_token_id=shard["gold_token_id"][start:end].clone(),
            preference=preference,
        )
