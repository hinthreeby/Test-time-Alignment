from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.extract_router_features import (
    RouterExample,
    append_gold_to_topk,
    assert_record_matches_recompute,
    compute_token_step_feature,
    extract_split,
    online_recompute_record,
    read_jsonl,
    sha256_file,
    transform_reward_scores,
    validate_resume_compatibility,
)


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class TinyBaseModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.arange(self.vocab_size, dtype=torch.float).repeat(batch, seq_len, 1)
        logits = logits + input_ids[:, -1:].float().view(batch, 1, 1) * 0.01
        return type("Output", (), {"logits": logits})()


class TinyRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, labels=None, use_cache=False):
        del labels, use_cache
        scores = (input_ids.float() * attention_mask.float()).sum(dim=1, keepdim=True) / 100.0
        return None, scores


class FeatureExtractionTests(unittest.TestCase):
    class Args:
        top_k = 5
        max_reward_length = 16
        shard_size_steps = 2
        reward_batch_size = 4
        resume = True
        inverse = False
        base_model_path = Path("base")
        reward_base_path = Path("reward-base")
        reward_tokenizer_path = Path("reward-tokenizer")
        reward_checkpoint_path = Path("reward.pt")
        reward_checkpoint_sha256 = "reward-sha"
        base_tokenizer_mapping_sha256 = "base-tokenizer-sha"
        reward_tokenizer_mapping_sha256 = "reward-tokenizer-sha"

    def test_force_gold_replaces_lowest_candidate(self) -> None:
        logits = torch.arange(30, dtype=torch.float)
        candidate_ids, base_logits, gold_index, gold_was_in_topk, original_rank = append_gold_to_topk(logits, 3, 5)
        self.assertEqual(candidate_ids.tolist(), [29, 28, 27, 26, 3])
        self.assertEqual(base_logits.tolist(), [29.0, 28.0, 27.0, 26.0, 3.0])
        self.assertEqual(gold_index, 4)
        self.assertFalse(gold_was_in_topk)
        self.assertEqual(original_rank, -1)

    def test_gold_kept_when_already_in_topk(self) -> None:
        logits = torch.arange(30, dtype=torch.float)
        candidate_ids, _base_logits, gold_index, gold_was_in_topk, original_rank = append_gold_to_topk(logits, 28, 5)
        self.assertEqual(candidate_ids.tolist(), [29, 28, 27, 26, 25])
        self.assertEqual(gold_index, 1)
        self.assertTrue(gold_was_in_topk)
        self.assertEqual(original_rank, 1)

    def test_cached_record_matches_online_recompute(self) -> None:
        device = torch.device("cpu")
        base_model = TinyBaseModel()
        reward_model = TinyRewardModel()
        tokenizer = TinyTokenizer()
        example = RouterExample("sample-1", "1 2", "3 4")
        record = compute_token_step_feature(
            base_model,
            reward_model,
            context_ids=[1, 2, 3],
            gold_token_id=4,
            sample_id="sample-1",
            position=1,
            top_k=5,
            max_reward_length=16,
            inverse=False,
            device=device,
        )
        recomputed = online_recompute_record(
            base_model,
            tokenizer,
            reward_model,
            example,
            position=1,
            top_k=5,
            max_reward_length=16,
            inverse=False,
            device=device,
        )
        cached = {
            "sample_ids": [record.sample_id],
            "position": torch.tensor([record.position]),
            "candidate_ids": record.candidate_ids.unsqueeze(0),
            "base_logits": record.base_logits.unsqueeze(0),
            "raw_reward_scores": record.raw_reward_scores.unsqueeze(0),
            "rad_reward_scores": record.rad_reward_scores.unsqueeze(0),
            "gold_token_id": torch.tensor([record.gold_token_id]),
            "gold_index": torch.tensor([record.gold_index]),
            "gold_was_in_topk": torch.tensor([record.gold_was_in_topk]),
            "attention_length": torch.tensor([record.attention_length]),
            "reward_effective_length": record.reward_effective_length.unsqueeze(0),
            "reward_transform_name_per_record": [record.reward_transform_name],
        }
        assert_record_matches_recompute(cached, 0, recomputed, atol=0.0)

    def test_raw_and_transformed_reward_are_cached(self) -> None:
        raw = torch.tensor([-2.0, 0.25, 2.0])
        self.assertEqual(transform_reward_scores(raw, inverse=False).tolist(), [0.0, 0.25, 1.0])
        self.assertEqual(transform_reward_scores(raw, inverse=True).tolist(), [1.0, 0.75, 0.0])

    def test_extract_split_resume_skips_completed_samples_without_duplicates(self) -> None:
        examples = [
            RouterExample("sample-1", "1 2", "3 4"),
            RouterExample("sample-2", "2 3", "4"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest1 = extract_split(
                "train",
                examples,
                Path("source.jsonl"),
                "abc",
                TinyBaseModel(),
                TinyTokenizer(),
                TinyRewardModel(),
                output_dir,
                self.Args(),
                torch.device("cpu"),
            )
            manifest2 = extract_split(
                "train",
                examples,
                Path("source.jsonl"),
                "abc",
                TinyBaseModel(),
                TinyTokenizer(),
                TinyRewardModel(),
                output_dir,
                self.Args(),
                torch.device("cpu"),
            )
            self.assertEqual(manifest1["cached_token_steps"], 3)
            self.assertEqual(manifest2["cached_token_steps"], 3)
            self.assertEqual(manifest2["skipped_samples"], 2)
            self.assertEqual(len(manifest1["shards"]), len(manifest2["shards"]))

    def test_resume_with_dataset_hash_or_config_mismatch_fails(self) -> None:
        examples = [RouterExample("sample-1", "1 2", "3")]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = extract_split(
                "train",
                examples,
                Path("source.jsonl"),
                "abc",
                TinyBaseModel(),
                TinyTokenizer(),
                TinyRewardModel(),
                output_dir,
                self.Args(),
                torch.device("cpu"),
            )
            with self.assertRaises(ValueError):
                extract_split(
                    "train",
                    examples,
                    Path("source.jsonl"),
                    "different-hash",
                    TinyBaseModel(),
                    TinyTokenizer(),
                    TinyRewardModel(),
                    output_dir,
                    self.Args(),
                    torch.device("cpu"),
                )
            bad_args = self.Args()
            bad_args.max_reward_length = 8
            with self.assertRaises(ValueError):
                validate_resume_compatibility(manifest, {
                    **manifest["cache_config"],
                    "max_reward_length": bad_args.max_reward_length,
                })

    def test_resume_with_bad_shard_hash_fails(self) -> None:
        examples = [RouterExample("sample-1", "1 2", "3")]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = extract_split(
                "train",
                examples,
                Path("source.jsonl"),
                "abc",
                TinyBaseModel(),
                TinyTokenizer(),
                TinyRewardModel(),
                output_dir,
                self.Args(),
                torch.device("cpu"),
            )
            shard_path = Path(manifest["shards"][0]["path"])
            shard_path.write_bytes(shard_path.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                extract_split(
                    "train",
                    examples,
                    Path("source.jsonl"),
                    "abc",
                    TinyBaseModel(),
                    TinyTokenizer(),
                    TinyRewardModel(),
                    output_dir,
                    self.Args(),
                    torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
