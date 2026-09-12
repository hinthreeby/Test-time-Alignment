from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import pyarrow as pa
from peft import LoraConfig, get_peft_model
from pyarrow import ipc
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from router_v2.cache.io import save_cache_shard
from router_v2.cache.schema import build_cache_shard
from router_v2.guide_model.cache_manifest import (
    build_split_manifest,
    cleanup_stale_shard_temps,
    recover_shard_state,
)
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.data import (
    PositiveContinuationDataset,
    RADSample,
    collate_tokenized_samples,
    tokenize_sample,
)
from router_v2.guide_model.evaluation import (
    load_negative_audit_samples,
    paired_score_summary,
    score_tokenized_samples,
)
from router_v2.guide_model.extraction import (
    CandidateBatch,
    combine_extracted_records,
    model_topk,
    prepare_extraction_batch,
    select_topk_at_positions,
    should_flush_before_extraction_batch,
)
from router_v2.scripts.audit_feature_cache import (
    _mismatch_diagnostic,
    group_records_by_extraction_batch,
)
from router_v2.guide_model.provenance import (
    tokenizer_descriptor,
    tokenizers_exactly_compatible,
)
from router_v2.guide_model.training import (
    assert_optimizer_parameters_fp32,
    cast_trainable_parameters_to_fp32,
    cleanup_stale_checkpoint_temps,
    deterministic_epoch_batches,
    latest_checkpoint,
    save_adapter_directory_atomic,
)


class TinyTokenizer:
    def __init__(self, *, reverse_vocab: bool = False) -> None:
        tokens = list("abcdefghijklmnopqrstuvwxyz .")
        if reverse_vocab:
            tokens = list(reversed(tokens))
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self.eos_token = "<eos>"
        self.eos_token_id = len(tokens)
        self.pad_token = "<eos>"
        self.pad_token_id = self.eos_token_id
        self.bos_token = None
        self.bos_token_id = None
        self.unk_token = None
        self.unk_token_id = None
        self.padding_side = "right"
        self.model_max_length = 128

    def __len__(self) -> int:
        return len(self._vocab) + 1

    def get_vocab(self) -> dict[str, int]:
        return {**self._vocab, self.eos_token: self.eos_token_id}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        del add_special_tokens
        return [self._vocab[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return "".join(
            self.eos_token if value == self.eos_token_id else inverse[value]
            for value in token_ids
        )


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, *, reverse: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.reverse = reverse
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.grad_enabled_during_forward: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, position_ids, use_cache
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        vocabulary = torch.arange(
            self.vocab_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        if self.reverse:
            vocabulary = vocabulary.flip(0)
        logits = (
            vocabulary.reshape(1, 1, -1)
            + input_ids.float().unsqueeze(-1) * 0.01
            + self.anchor * 0.0
        )
        return SimpleNamespace(logits=logits)


class SaveableStub:
    def save_pretrained(
        self,
        path: Path,
        *,
        safe_serialization: bool = True,
    ) -> None:
        self.safe_serialization = safe_serialization
        torch.save({"weight": torch.tensor([1.0])}, path / "adapter.bin")


class TokenizerSaveableStub:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def make_sample(
    sample_id: str = "sample-a",
    *,
    prompt: str = "ab",
    continuation: str = "cd",
    source_index: int = 0,
) -> RADSample:
    return RADSample(
        sample_id=sample_id,
        prompt=prompt,
        continuation=continuation,
        label=1,
        source="amazon_polarity",
        source_split="train",
        source_index=source_index,
    )


class GuideDataTests(unittest.TestCase):
    def test_config_pins_documented_sentiment_objective(self) -> None:
        config = SentimentGuideConfig()

        self.assertEqual(config.task, "sentiment")
        self.assertEqual(config.objective, "positive_continuation_causal_nll")
        self.assertEqual(config.base_model_path, "models/gpt2-medium")
        self.assertGreater(config.minimum_positive_nll_improvement, 0)
        self.assertGreater(config.minimum_direction_delta_gap, 0)

    def test_negative_audit_uses_held_out_negative_arrow_rows(self) -> None:
        tokenizer = TinyTokenizer()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "test.arrow"
            table = pa.table(
                {
                    "label": [1, 0],
                    "title": ["good item", "bad item"],
                    "content": [
                        "it was good and useful",
                        "it was bad and not useful",
                    ],
                }
            )
            with pa.OSFile(str(path), "wb") as sink:
                with ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)

            samples = load_negative_audit_samples(
                path,
                tokenizer,
                max_samples=1,
                max_length=64,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].label, 0)
        self.assertEqual(samples[0].source_split, "test")
        self.assertEqual(samples[0].source_index, 1)

    def test_prompt_is_masked_and_continuation_plus_eos_is_supervised(self) -> None:
        tokenizer = TinyTokenizer()
        tokenized = tokenize_sample(
            tokenizer,
            make_sample(),
            max_length=16,
            include_eos_target=True,
        )

        self.assertEqual(tokenized.prompt_length, 2)
        self.assertEqual(tokenized.labels[:2], (-100, -100))
        self.assertEqual(
            tokenized.labels[2:],
            (*tokenized.continuation_ids, tokenizer.eos_token_id),
        )

    def test_collator_masks_padding(self) -> None:
        tokenizer = TinyTokenizer()
        dataset = PositiveContinuationDataset(
            [
                make_sample(),
                make_sample(
                    "sample-b",
                    prompt="a",
                    continuation="b",
                    source_index=1,
                ),
            ],
            tokenizer,
            max_length=16,
            include_eos_target=True,
        )

        batch = collate_tokenized_samples(
            dataset.items,
            pad_token_id=tokenizer.pad_token_id,
        )

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)


class TeacherForcingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.sample = make_sample()

    def test_logit_positions_predict_each_gold_continuation_token(self) -> None:
        batch = prepare_extraction_batch(
            self.tokenizer,
            [self.sample],
            max_length=16,
            max_continuation_tokens=2,
        )

        self.assertEqual(batch.logit_positions, ((1, 2),))
        self.assertEqual(batch.continuation_ids[0], (2, 3))
        self.assertEqual(
            batch.input_ids[0, :4].tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(batch.position_ids[0, :4].tolist(), [0, 1, 2, 3])

    def test_topk_selector_has_no_gold_argument_and_is_independent(self) -> None:
        signature = inspect.signature(select_topk_at_positions)
        self.assertNotIn("gold", " ".join(signature.parameters))
        base = torch.tensor([[[9.0, 8.0, 0.0, -1.0]]])
        guide = torch.tensor([[[-1.0, 0.0, 8.0, 9.0]]])

        base_topk = select_topk_at_positions(base, ((0,),), top_k=2)
        guide_topk = select_topk_at_positions(guide, ((0,),), top_k=2)

        self.assertEqual(base_topk[0].token_ids.tolist(), [[0, 1]])
        self.assertEqual(guide_topk[0].token_ids.tolist(), [[3, 2]])
        self.assertEqual(base_topk[0].boundary_margin.tolist(), [8.0])
        self.assertEqual(guide_topk[0].boundary_margin.tolist(), [8.0])
        self.assertEqual(base_topk[0].source_dtype, "torch.float32")

    def test_model_output_contract_and_inference_mode(self) -> None:
        batch = prepare_extraction_batch(
            self.tokenizer,
            [self.sample],
            max_length=16,
            max_continuation_tokens=2,
        )
        model = TinyCausalLM(len(self.tokenizer))

        candidates = model_topk(
            model,
            batch,
            top_k=3,
            device=torch.device("cpu"),
            use_fp16=False,
        )

        self.assertEqual(candidates[0].token_ids.shape, (2, 3))
        self.assertEqual(candidates[0].logits.shape, (2, 3))
        self.assertEqual(model.grad_enabled_during_forward, [False])
        self.assertFalse(model.training)

    def test_fp32_path_rejects_non_fp32_model_logits(self) -> None:
        class HalfOutputModel(TinyCausalLM):
            def forward(self, *args: object, **kwargs: object) -> SimpleNamespace:
                output = super().forward(*args, **kwargs)
                return SimpleNamespace(logits=output.logits.half())

        batch = prepare_extraction_batch(
            self.tokenizer,
            [self.sample],
            max_length=16,
            max_continuation_tokens=2,
        )

        with self.assertRaisesRegex(TypeError, "Expected torch.float32"):
            model_topk(
                HalfOutputModel(len(self.tokenizer)),
                batch,
                top_k=3,
                device=torch.device("cpu"),
                use_fp16=False,
            )

    def test_gold_is_added_only_after_both_candidate_streams(self) -> None:
        batch = prepare_extraction_batch(
            self.tokenizer,
            [self.sample],
            max_length=16,
            max_continuation_tokens=2,
        )
        base = model_topk(
            TinyCausalLM(len(self.tokenizer)),
            batch,
            top_k=2,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        guide = model_topk(
            TinyCausalLM(len(self.tokenizer), reverse=True),
            batch,
            top_k=2,
            device=torch.device("cpu"),
            use_fp16=False,
        )

        records = combine_extracted_records(batch, base, guide)

        self.assertEqual(records[0].gold_token_id.tolist(), [2, 3])
        self.assertNotEqual(
            records[0].base_token_ids.tolist(),
            records[0].guide_token_ids.tolist(),
        )

    def test_streamed_cache_matches_online_recomputation(self) -> None:
        batch = prepare_extraction_batch(
            self.tokenizer,
            [self.sample],
            max_length=16,
            max_continuation_tokens=2,
        )
        base_model = TinyCausalLM(len(self.tokenizer))
        guide_model = TinyCausalLM(len(self.tokenizer), reverse=True)
        first_base = model_topk(
            base_model,
            batch,
            top_k=3,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        first_guide = model_topk(
            guide_model,
            batch,
            top_k=3,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        record = combine_extracted_records(
            batch,
            first_base,
            first_guide,
        )[0]
        shard = build_cache_shard(
            split="train",
            shard_index=0,
            sample_ids=[self.sample.sample_id] * record.num_records,
            position=record.position,
            base_token_ids=record.base_token_ids,
            base_logits=record.base_logits,
            guide_token_ids=record.guide_token_ids,
            guide_logits=record.guide_logits,
            gold_token_id=record.gold_token_id,
            max_position=10,
            vocab_size=len(self.tokenizer),
            source={"test": True},
        )
        online_base = model_topk(
            base_model,
            batch,
            top_k=3,
            device=torch.device("cpu"),
            use_fp16=False,
        )[0]
        online_guide = model_topk(
            guide_model,
            batch,
            top_k=3,
            device=torch.device("cpu"),
            use_fp16=False,
        )[0]

        self.assertNotIn("base_full_logits", shard)
        self.assertNotIn("guide_full_logits", shard)
        torch.testing.assert_close(shard["base_logits"], online_base.logits)
        torch.testing.assert_close(shard["guide_logits"], online_guide.logits)
        self.assertTrue(
            torch.equal(shard["base_token_ids"], online_base.token_ids)
        )
        self.assertTrue(
            torch.equal(shard["guide_token_ids"], online_guide.token_ids)
        )


class TokenizerAndScoringTests(unittest.TestCase):
    def test_exact_tokenizer_compatibility_and_hash(self) -> None:
        first = TinyTokenizer()
        second = TinyTokenizer()
        different = TinyTokenizer(reverse_vocab=True)

        compatible, checks = tokenizers_exactly_compatible(first, second)
        incompatible, _ = tokenizers_exactly_compatible(first, different)
        first_descriptor = tokenizer_descriptor(first, "first")
        second_descriptor = tokenizer_descriptor(second, "second")

        self.assertTrue(compatible)
        self.assertTrue(all(checks.values()))
        self.assertFalse(incompatible)
        self.assertEqual(
            first_descriptor["vocab_sha256"],
            second_descriptor["vocab_sha256"],
        )

    def test_continuation_scores_align_for_base_and_guide(self) -> None:
        tokenizer = TinyTokenizer()
        dataset = PositiveContinuationDataset(
            [make_sample()],
            tokenizer,
            max_length=16,
            include_eos_target=True,
        )
        base = score_tokenized_samples(
            TinyCausalLM(len(tokenizer)),
            dataset.items,
            batch_size=1,
            pad_token_id=tokenizer.pad_token_id,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        guide = score_tokenized_samples(
            TinyCausalLM(len(tokenizer), reverse=True),
            dataset.items,
            batch_size=1,
            pad_token_id=tokenizer.pad_token_id,
            device=torch.device("cpu"),
            use_fp16=False,
        )

        summary = paired_score_summary(base, guide)

        self.assertEqual(summary["num_samples"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(summary["base_mean_nll"])))

    def test_documented_fp16_tolerance_covers_rounding_only(self) -> None:
        values = torch.tensor([1.2341, -8.7654, 0.0012])
        round_trip = values.half().float()

        torch.testing.assert_close(values, round_trip, atol=0.02, rtol=0)


def build_resume_shard(
    *,
    shard_index: int,
    sample_id: str,
    source_start: int,
    fingerprint: str,
) -> dict[str, object]:
    return build_cache_shard(
        split="train",
        shard_index=shard_index,
        sample_ids=[sample_id, sample_id],
        position=torch.tensor([0, 1]),
        base_token_ids=torch.tensor([[5, 4], [6, 5]]),
        base_logits=torch.tensor([[3.0, 2.0], [4.0, 1.0]]),
        guide_token_ids=torch.tensor([[1, 2], [2, 3]]),
        guide_logits=torch.tensor([[5.0, 1.0], [3.0, 2.0]]),
        gold_token_id=torch.tensor([3, 4]),
        max_position=10,
        vocab_size=8,
        source={
            "extraction_fingerprint": fingerprint,
            "source_start_index": source_start,
            "source_end_index_exclusive": source_start + 1,
            "source_ids": [sample_id],
        },
    )


class ResumeAndManifestTests(unittest.TestCase):
    def test_shard_flush_never_splits_an_extraction_batch(self) -> None:
        self.assertFalse(
            should_flush_before_extraction_batch(
                buffered_records=0,
                incoming_batch_records=240,
                shard_size=4096,
            )
        )
        self.assertFalse(
            should_flush_before_extraction_batch(
                buffered_records=3800,
                incoming_batch_records=200,
                shard_size=4096,
            )
        )
        self.assertTrue(
            should_flush_before_extraction_batch(
                buffered_records=3900,
                incoming_batch_records=200,
                shard_size=4096,
            )
        )

    def test_audit_groups_records_by_original_extraction_batch(self) -> None:
        records = [
            {"sample_id": "sample-a", "position": 1},
            {"sample_id": "sample-b", "position": 2},
            {"sample_id": "sample-c", "position": 3},
        ]
        groups = group_records_by_extraction_batch(
            records,
            {"sample-a": 0, "sample-b": 3, "sample-c": 4},
            batch_size=4,
        )

        self.assertEqual(sorted(groups), [0, 4])
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[4]), 1)

    def test_mismatch_diagnostic_contains_reproducibility_context(self) -> None:
        cached = {
            "sample_id": "sample-a",
            "base_token_ids": torch.tensor([1, 2]),
            "base_logits": torch.tensor([4.0, 3.0]),
            "guide_token_ids": torch.tensor([3, 4]),
            "guide_logits": torch.tensor([5.0, 2.0]),
        }
        base = CandidateBatch(
            token_ids=torch.tensor([[1, 2]]),
            logits=torch.tensor([[4.0, 3.0]]),
            boundary_margin=torch.tensor([0.25]),
            source_dtype="torch.float32",
        )
        guide = CandidateBatch(
            token_ids=torch.tensor([[3, 4]]),
            logits=torch.tensor([[5.0, 2.0]]),
            boundary_margin=torch.tensor([0.5]),
            source_dtype="torch.float32",
        )
        diagnostic = _mismatch_diagnostic(
            cached=cached,
            online_base=base,
            online_guide=guide,
            position=0,
            prefix={
                "prefix_token_ids": [7, 8],
                "prefix_token_ids_sha256": "prefix-hash",
            },
            precision="fp32",
            base_checkpoint_sha256="base-hash",
            guide_checkpoint_sha256="guide-hash",
        )

        required = {
            "cached_base_topk_token_ids",
            "online_base_topk_token_ids",
            "cached_guide_topk_token_ids",
            "online_guide_topk_token_ids",
            "minimum_topk_boundary_margin",
            "prefix_token_ids",
            "prefix_token_ids_sha256",
            "base_model_output_dtype",
            "guide_model_output_dtype",
            "base_checkpoint_sha256",
            "guide_checkpoint_sha256",
        }
        self.assertTrue(required.issubset(diagnostic))
        self.assertEqual(diagnostic["minimum_topk_boundary_margin"], 0.25)

    def test_amp_keeps_frozen_fp16_and_trainable_lora_gradients_fp32(
        self,
    ) -> None:
        base = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_ctx=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
            )
        ).half()
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["c_attn"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).half()
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertTrue(
            all(
                parameter.dtype == torch.float16
                for parameter in model.parameters()
            )
        )

        trainable = cast_trainable_parameters_to_fp32(model)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        assert_optimizer_parameters_fp32(optimizer)

        loss = sum(parameter.square().sum() for parameter in trainable)
        loss.backward()

        frozen = [
            parameter
            for parameter in model.parameters()
            if not parameter.requires_grad
        ]
        self.assertTrue(frozen)
        self.assertTrue(
            all(parameter.dtype == torch.float16 for parameter in frozen)
        )
        self.assertTrue(all(parameter.grad is None for parameter in frozen))
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in trainable)
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                and parameter.grad.dtype == torch.float32
                for parameter in trainable
            )
        )

    def test_deterministic_epoch_order_supports_exact_resume(self) -> None:
        first = deterministic_epoch_batches(
            num_samples=11,
            batch_size=3,
            seed=42,
            epoch=0,
        )
        second = deterministic_epoch_batches(
            num_samples=11,
            batch_size=3,
            seed=42,
            epoch=0,
        )

        self.assertEqual(first, second)
        self.assertEqual(first[2:], second[2:])
        self.assertNotEqual(
            first,
            deterministic_epoch_batches(
                num_samples=11,
                batch_size=3,
                seed=42,
                epoch=1,
            ),
        )

    def test_atomic_checkpoint_and_latest_resume_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in (2, 10):
                destination = root / f"checkpoint-step-{step:07d}"
                save_adapter_directory_atomic(
                    model=SaveableStub(),
                    tokenizer=TokenizerSaveableStub(),
                    destination=destination,
                    training_state={"global_step": step},
                    metadata={"step": step},
                )

            self.assertEqual(
                latest_checkpoint(root).name,
                "checkpoint-step-0000010",
            )
            with self.assertRaises(FileExistsError):
                save_adapter_directory_atomic(
                    model=SaveableStub(),
                    tokenizer=TokenizerSaveableStub(),
                    destination=root / "checkpoint-step-0000010",
                    training_state={},
                    metadata={},
                )

    def test_interrupted_temp_recovery_and_orphan_shard_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stale = root / ".shard_00001.pt.dead.tmp"
            stale.write_bytes(b"partial")
            save_cache_shard(
                build_resume_shard(
                    shard_index=0,
                    sample_id="sample-a",
                    source_start=0,
                    fingerprint="run-a",
                ),
                root / "shard_00000.pt",
            )

            removed = cleanup_stale_shard_temps(root)
            state = recover_shard_state(
                root,
                extraction_fingerprint="run-a",
            )

            self.assertEqual(removed, [stale.name])
            self.assertEqual(state["next_shard_index"], 1)
            self.assertEqual(state["next_source_index"], 1)
            self.assertEqual(state["num_records"], 2)

    def test_duplicate_sample_position_across_shards_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_cache_shard(
                build_resume_shard(
                    shard_index=0,
                    sample_id="duplicate",
                    source_start=0,
                    fingerprint="run-a",
                ),
                root / "shard_00000.pt",
            )
            save_cache_shard(
                build_resume_shard(
                    shard_index=1,
                    sample_id="duplicate",
                    source_start=1,
                    fingerprint="run-a",
                ),
                root / "shard_00001.pt",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate records"):
                recover_shard_state(
                    root,
                    extraction_fingerprint="run-a",
                )

    def test_manifest_contains_hashes_counts_and_feature_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, sample_id in enumerate(("sample-a", "sample-b")):
                save_cache_shard(
                    build_resume_shard(
                        shard_index=index,
                        sample_id=sample_id,
                        source_start=index,
                        fingerprint="run-a",
                    ),
                    root / f"shard_{index:05d}.pt",
                )

            manifest = build_split_manifest(
                root,
                split="train",
                extraction_fingerprint="run-a",
                extraction_parameters={"batch_size": 2, "shard_size": 4},
                source_dataset={"path": "source", "sha256": "data-hash"},
                base_model={"checkpoint_sha256": "base-hash"},
                guide_model={"checkpoint_sha256": "guide-hash"},
                base_tokenizer={"semantic_sha256": "tokenizer-hash"},
                guide_tokenizer={"semantic_sha256": "tokenizer-hash"},
                top_k=2,
                max_position=10,
                precision="fp32",
                device="cpu",
                expected_source_examples=2,
                stale_temps_removed=[],
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["number_shards"], 2)
            self.assertEqual(manifest["number_token_records"], 4)
            self.assertEqual(manifest["source_ids"], ["sample-a", "sample-b"])
            self.assertEqual(manifest["nan_inf_count"], 0)
            self.assertEqual(manifest["top_k_uniqueness_failures"], 0)
            self.assertEqual(manifest["extraction_parameters"]["batch_size"], 2)
            self.assertEqual(len(manifest["shards"][0]["sha256"]), 64)

    def test_training_temp_cleanup_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stale = root / ".checkpoint-step-0000001.dead.tmp"
            stale.mkdir()
            unrelated = root / "keep"
            unrelated.mkdir()

            removed = cleanup_stale_checkpoint_temps(root)

            self.assertEqual(removed, [stale.name])
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
