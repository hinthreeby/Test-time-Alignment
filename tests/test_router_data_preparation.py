from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pyarrow as pa
from pyarrow import ipc

from scripts.data.build_router_amazon_polarity import (
    BuildArgs,
    SourceRecord,
    benchmark_leakage_index,
    choose_split_index,
    combine_review,
    has_benchmark_leakage,
    inspect_source,
    jaccard,
    prepare_record,
    split_examples,
    text_fingerprint,
    token_set,
)


class SpaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = []
        index = 0
        for match in __import__("re").finditer(r"\S+\s*", text):
            tokens.append(index)
            index += 1
        self._last_parts = __import__("re").findall(r"\S+\s*", text)
        return list(range(len(self._last_parts)))

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self._last_parts[index] for index in token_ids)


def write_arrow(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def make_source(root: Path) -> Path:
    source = root / "amazon_polarity"
    for split in ["train", "test"]:
        (source / split).mkdir(parents=True)
    (source / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "test"]}))
    info = {
        "features": {
            "label": {"names": ["negative", "positive"], "_type": "ClassLabel"},
            "title": {"dtype": "string", "_type": "Value"},
            "content": {"dtype": "string", "_type": "Value"},
        },
        "splits": {
            "train": {"name": "train", "num_examples": 2},
            "test": {"name": "test", "num_examples": 1},
        },
    }
    for split in ["train", "test"]:
        (source / split / "dataset_info.json").write_text(json.dumps(info))
        (source / split / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data.arrow"}]}))
    write_arrow(
        source / "train" / "data.arrow",
        [
            {"label": 1, "title": "Great", "content": "This product works very well and I would buy it again."},
            {"label": 0, "title": "Bad", "content": "This product failed quickly."},
        ],
    )
    write_arrow(source / "test" / "data.arrow", [{"label": 1, "title": "Fine", "content": "Kept unused."}])
    return source


def base_args(root: Path) -> BuildArgs:
    return BuildArgs(
        source_dir=root / "source",
        benchmark_dir=root / "benchmark",
        output_dir=root / "out",
        train_size=2,
        validation_size=1,
        dev_size=0,
        smoke_test=False,
        smoke_train_size=1,
        smoke_validation_size=1,
        smoke_dev_size=0,
        min_total_tokens=5,
        max_total_tokens=100,
        min_prompt_tokens=2,
        min_continuation_tokens=2,
        max_continuation_tokens=80,
        min_split_ratio=0.30,
        max_split_ratio=0.70,
        seed=42,
        near_duplicate_threshold=0.9,
        batch_size=10,
        tokenizer_name="gpt2",
        local_files_only=True,
        overwrite=True,
        write_dev=False,
    )


class RouterDataPreparationTests(unittest.TestCase):
    def test_source_inspection_detects_arrow_fields_and_positive_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = make_source(Path(tmp))
            inspection = inspect_source(source)
            self.assertEqual(inspection.format, "huggingface_saved_arrow")
            self.assertEqual(inspection.fields["label"], "ClassLabel")
            self.assertEqual(inspection.title_field, "title")
            self.assertEqual(inspection.content_field, "content")
            self.assertEqual(inspection.positive_label, 1)
            self.assertTrue(inspection.official_splits_available)
            self.assertEqual(inspection.row_counts["train"], 2)

    def test_positive_label_mapping_rejects_negative_rows(self) -> None:
        args = base_args(Path("/tmp"))
        tokenizer = SpaceTokenizer()
        record = SourceRecord("train", 0, 0, "Bad", "This item does not work well enough.")
        example, status = prepare_record(record, tokenizer, 1, args)
        self.assertIsNone(example)
        self.assertEqual(status, "non_positive_label")

    def test_token_boundary_split_and_reconstruction(self) -> None:
        tokenizer = SpaceTokenizer()
        text = "One two three four five six seven eight nine ten."
        ids = tokenizer.encode(text)
        split = choose_split_index(ids, text, tokenizer, 2, 2, 8, 0.3, 0.6, random.Random(7))
        self.assertIsNotNone(split)
        prompt = tokenizer.decode(ids[:split])
        continuation = tokenizer.decode(ids[split:])
        self.assertEqual(prompt + continuation, text)
        self.assertGreaterEqual(len(tokenizer.encode(prompt)), 2)
        self.assertGreaterEqual(len(tokenizer.encode(continuation)), 2)

    def test_prepare_record_reconstructs_combined_review(self) -> None:
        args = base_args(Path("/tmp"))
        tokenizer = SpaceTokenizer()
        record = SourceRecord(
            "train",
            123,
            1,
            "Useful",
            "This product works very well and feels sturdy after many uses.",
        )
        example, status = prepare_record(record, tokenizer, 1, args)
        self.assertEqual(status, "accepted")
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.prompt + example.continuation, combine_review(record.title, record.content))
        self.assertEqual(example.label, 1)
        self.assertEqual(example.source_index, 123)

    def test_exact_duplicate_fingerprints_match(self) -> None:
        left = text_fingerprint("  Great PRODUCT!!! ")
        right = text_fingerprint("great product")
        self.assertEqual(left, right)

    def test_near_duplicate_jaccard(self) -> None:
        left = token_set("this product is sturdy reliable and useful")
        right = token_set("this product is reliable useful and sturdy")
        self.assertGreaterEqual(jaccard(left, right), 0.9)

    def test_deterministic_splitting(self) -> None:
        examples = []
        args = base_args(Path("/tmp"))
        tokenizer = SpaceTokenizer()
        for i in range(6):
            record = SourceRecord("train", i, 1, f"Title {i}", "one two three four five six seven eight nine ten")
            example, status = prepare_record(record, tokenizer, 1, args)
            self.assertEqual(status, "accepted")
            assert example is not None
            examples.append(example)
        first = split_examples(examples, 3, 2, 1, 99)
        second = split_examples(list(reversed(examples)), 3, 2, 1, 99)
        self.assertEqual([row.id for row in first["train"]], [row.id for row in second["train"]])
        self.assertEqual([row.id for row in first["validation"]], [row.id for row in second["validation"]])

    def test_benchmark_leakage_detection_nested_prompt_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in ["negative_prompts.jsonl", "neutral_prompts.jsonl", "positive_prompts.jsonl"]:
                path = root / name
                path.write_text(
                    json.dumps({"prompt": {"text": "Great product"}, "continuation": {"text": " works well"}}) + "\n"
                )
                paths.append(path)
            exact, near = benchmark_leakage_index(paths)
            args = base_args(root)
            tokenizer = SpaceTokenizer()
            example, status = prepare_record(
                SourceRecord("train", 5, 1, None, "Great product works well for everyone today."),
                tokenizer,
                1,
                replace(args, near_duplicate_threshold=0.5),
            )
            self.assertEqual(status, "accepted")
            assert example is not None
            leaked, reason = has_benchmark_leakage(example, exact, near, 0.5)
            self.assertTrue(leaked)
            self.assertIn(reason, {"benchmark_exact_leakage", "benchmark_near_leakage"})


if __name__ == "__main__":
    unittest.main()
