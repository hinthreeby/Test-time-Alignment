from __future__ import annotations

import json
import lzma
import tempfile
import unittest
from pathlib import Path

from PARM_TARO.data.multi_objective import (
    PROCESSED_FIELDS,
    deterministic_prompt_group_split,
    prepare_multi_objective_dataset,
    sha256_file,
    validate_processed_dataset,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def source_row(index: int, *, prompt: str | None = None) -> dict[str, object]:
    return {
        "prompt": prompt or f"prompt {index}",
        "response_0": f"response zero {index}",
        "response_1": f"response one {index}",
        "better_response_id": index % 2,
        "safer_response_id": (index // 2) % 2,
        "is_response_0_safe": index % 3 != 0,
        "is_response_1_safe": index % 4 != 0,
    }


class MultiObjectivePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "train.jsonl.xz"
        self.parm_data = self.root / "parm_data"
        self.parm_data.mkdir()
        (self.parm_data / "relabel.py").write_text(
            "# immutable author placeholder\n", encoding="utf-8"
        )
        rows = [source_row(index) for index in range(18)]
        rows[1] = dict(rows[0])
        rows[3]["prompt"] = rows[2]["prompt"]
        rows[5]["prompt"] = rows[4]["prompt"]
        with lzma.open(self.source, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self, name: str = "processed") -> Path:
        output = self.root / name
        report = prepare_multi_objective_dataset(
            source_path=self.source,
            output_dir=output,
            parm_data_path=self.parm_data,
            train_records=10,
            validation_records=3,
            seed="test-seed",
        )
        self.assertEqual(report["status"], "PASS")
        return output

    def test_prompt_group_split_is_exact_isolated_and_complete(self) -> None:
        with lzma.open(self.source, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        assignment = deterministic_prompt_group_split(
            rows,
            train_records=10,
            validation_records=3,
            seed="test-seed",
        )
        self.assertEqual({name: len(value) for name, value in assignment.items()}, {
            "train": 10,
            "validation": 3,
            "test": 5,
        })
        prompt_sets = {
            split: {rows[index]["prompt"] for index in indices}
            for split, indices in assignment.items()
        }
        self.assertFalse(prompt_sets["train"] & prompt_sets["validation"])
        self.assertFalse(prompt_sets["train"] & prompt_sets["test"])
        self.assertFalse(prompt_sets["validation"] & prompt_sets["test"])

    def test_processing_preserves_both_labels_and_exact_duplicates(self) -> None:
        output = self.prepare()
        records = []
        for filename in ("train.json", "validation.json", "test.json"):
            records.extend(json.loads((output / filename).read_text(encoding="utf-8")))
        by_source = {row["source_index"]: row for row in records}
        self.assertEqual(set(by_source), set(range(18)))
        self.assertNotEqual(by_source[0]["sample_id"], by_source[1]["sample_id"])
        for row in records:
            self.assertEqual(tuple(row), PROCESSED_FIELDS)
            self.assertEqual(
                row["helpfulness_preferred_response_id"],
                row["better_response_id"],
            )
            self.assertEqual(
                row["harmlessness_preferred_response_id"],
                row["safer_response_id"],
            )

    def test_processing_is_byte_deterministic(self) -> None:
        first = self.prepare("first")
        second = self.prepare("second")
        names = (
            "train.json",
            "validation.json",
            "test.json",
            "test_prompt_only.json",
            "source_report.json",
            "manifest.json",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )

    def test_source_and_author_data_are_not_modified(self) -> None:
        source_hash = sha256_file(self.source)
        relabel_hash = sha256_file(self.parm_data / "relabel.py")
        self.prepare()
        self.assertEqual(sha256_file(self.source), source_hash)
        self.assertEqual(
            sha256_file(self.parm_data / "relabel.py"), relabel_hash
        )

    def test_existing_output_is_not_overwritten_by_default(self) -> None:
        output = self.prepare()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            prepare_multi_objective_dataset(
                source_path=self.source,
                output_dir=output,
                parm_data_path=self.parm_data,
                train_records=10,
                validation_records=3,
                seed="test-seed",
            )

    def test_validator_detects_file_tampering(self) -> None:
        output = self.prepare()
        with (output / "train.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_processed_dataset(output, source_path=self.source)

    def test_validator_detects_manifest_assignment_tampering(self) -> None:
        output = self.prepare()
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split_assignment_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "assignment hash mismatch"):
            validate_processed_dataset(output, source_path=self.source)

    def test_manifest_records_named_and_author_objective_orders(self) -> None:
        output = self.prepare()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        semantics = manifest["objective_semantics"]
        self.assertEqual(
            semantics["named_order_in_processed_data"],
            ["helpfulness", "harmlessness"],
        )
        self.assertEqual(
            semantics["parm_author_pblora_vector_order"],
            ["harmlessness", "helpfulness"],
        )
        self.assertFalse(manifest["processing"]["download_performed"])
        self.assertFalse(manifest["processing"]["relabel_performed"])


class MaterializedDatasetTests(unittest.TestCase):
    def test_workspace_dataset_passes_full_validator(self) -> None:
        output = WORKSPACE_ROOT / "dataset/parm_taro"
        source = (
            WORKSPACE_ROOT
            / "dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz"
        )
        report = validate_processed_dataset(output, source_path=source)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["split_counts"],
            {"train": 8000, "validation": 500, "test": 1500},
        )


if __name__ == "__main__":
    unittest.main()
