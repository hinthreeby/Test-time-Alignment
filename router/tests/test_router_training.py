from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from router.model import RADTokenRouter
from router.training import (
    FeatureShardDataset,
    RouterTrainingConfig,
    assert_no_external_params_in_optimizer,
    candidate_nll_from_scores,
    fixed_beta_metrics,
    optimizer_contains_only_router,
    sha256_file,
    train_router,
)


def write_manifest_with_shard(root: Path, split: str, base_logits: torch.Tensor, reward_scores: torch.Tensor, gold_index: torch.Tensor) -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    shard_path = split_dir / "shard_00000.pt"
    payload = {
        "schema_version": 2,
        "split": split,
        "shard_index": 0,
        "top_k": 20,
        "num_records": int(base_logits.shape[0]),
        "sample_ids": [f"{split}-{i}" for i in range(base_logits.shape[0])],
        "position": torch.arange(base_logits.shape[0]),
        "candidate_ids": torch.arange(20).repeat(base_logits.shape[0], 1),
        "base_logits": base_logits.float(),
        "raw_reward_scores": reward_scores.float(),
        "rad_reward_scores": reward_scores.float(),
        "gold_token_id": gold_index.long(),
        "gold_index": gold_index.long(),
        "gold_was_in_topk": torch.ones(base_logits.shape[0], dtype=torch.bool),
        "attention_length": torch.ones(base_logits.shape[0], dtype=torch.long),
    }
    torch.save(payload, shard_path)
    manifest = {
        "schema_version": 2,
        "split": split,
        "source_dataset_hash": "synthetic",
        "cached_token_steps": int(base_logits.shape[0]),
        "top_k": 20,
        "shards": [
            {
                "path": str(shard_path),
                "sha256": sha256_file(shard_path),
                "num_records": int(base_logits.shape[0]),
            }
        ],
    }
    manifest_path = split_dir / "manifest.json"
    import json
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


class RouterTrainingTests(unittest.TestCase):
    def training_config(
        self,
        train_manifest: Path,
        validation_manifest: Path,
        output_dir: Path,
        epochs: int,
        learning_rate: float = 1e-3,
        patience: int = 5,
        batch_size: int = 4,
    ) -> RouterTrainingConfig:
        return RouterTrainingConfig(
            train_manifest=str(train_manifest),
            validation_manifest=str(validation_manifest),
            output_dir=str(output_dir),
            beta_max=5.0,
            beta_init=1.0,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=batch_size,
            epochs=epochs,
            gradient_clip_norm=1.0,
            early_stopping_patience=patience,
            seed=123,
            fixed_beta_baselines=[0.0, 1.0, 2.0],
            near_zero_fraction=0.05,
            near_max_fraction=0.95,
            device="cpu",
        )

    def test_dataset_requires_rad_reward_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "train"
            split_dir.mkdir()
            shard_path = split_dir / "shard_00000.pt"
            torch.save({"base_logits": torch.zeros(1, 20), "reward_scores": torch.zeros(1, 20), "gold_index": torch.zeros(1, dtype=torch.long), "num_records": 1}, shard_path)
            import json
            manifest_path = split_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"top_k": 20, "cached_token_steps": 1, "shards": [{"path": str(shard_path), "sha256": sha256_file(shard_path), "num_records": 1}]}) + "\n")
            dataset = FeatureShardDataset(manifest_path)
            with self.assertRaises(KeyError):
                next(dataset.iter_batches(batch_size=1, device=torch.device("cpu"), shuffle=False, seed=0))

    def test_optimizer_contains_only_router_not_frozen_models(self) -> None:
        router = RADTokenRouter(beta_max=10.0)
        base_lm = nn.Linear(2, 2)
        reward_model = nn.Linear(2, 1)
        for model in [base_lm, reward_model]:
            for parameter in model.parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        self.assertTrue(optimizer_contains_only_router(optimizer, router))
        assert_no_external_params_in_optimizer(optimizer, router, [base_lm, reward_model])

    def test_candidate_nll_matches_cross_entropy(self) -> None:
        guided = torch.randn(4, 20)
        gold = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(candidate_nll_from_scores(guided, gold), torch.nn.functional.cross_entropy(guided, gold)))

    def test_training_writes_best_last_and_resume_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(7)
            base = torch.randn(16, 20)
            reward = torch.randn(16, 20)
            gold = (base + 2.0 * reward).argmax(dim=-1)
            train_manifest = write_manifest_with_shard(root / "features", "train", base, reward, gold)
            validation_manifest = write_manifest_with_shard(root / "features", "validation", base[:8], reward[:8], gold[:8])
            output_dir = root / "out"
            config = self.training_config(train_manifest, validation_manifest, output_dir, epochs=2)
            first = train_router(config, resume=False)
            self.assertTrue(Path(first["last_checkpoint"]).exists())
            self.assertTrue(Path(first["best_checkpoint"]).exists())
            self.assertEqual(len(first["history"]), 2)
            last_payload = torch.load(first["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(last_payload["global_step"], 8)
            self.assertEqual(first["history"][0]["train"]["optimizer_steps"], 4)
            resumed_config = RouterTrainingConfig(**{**config.__dict__, "epochs": 3})
            resumed = train_router(resumed_config, resume=True)
            self.assertEqual(len(resumed["history"]), 3)
            self.assertEqual([row["epoch"] for row in resumed["history"]], [0, 1, 2])
            resumed_last = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(resumed_last["global_step"], 12)
            self.assertEqual(resumed_last["dataset_manifest_hashes"], last_payload["dataset_manifest_hashes"])

    def test_fixed_beta_baseline_runs_on_rad_reward_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.zeros(3, 20)
            reward = torch.zeros(3, 20)
            reward[:, 5] = 10.0
            gold = torch.tensor([5, 5, 5])
            manifest = write_manifest_with_shard(root, "validation", base, reward, gold)
            metrics = fixed_beta_metrics(FeatureShardDataset(manifest), beta=1.0, batch_size=2, device=torch.device("cpu"))
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertLess(metrics["nll"], 0.01)

    def test_early_stopping_triggers_when_bad_epochs_reach_patience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.zeros(8, 20)
            reward = torch.zeros(8, 20)
            gold = torch.zeros(8, dtype=torch.long)
            train_manifest = write_manifest_with_shard(root / "features", "train", base, reward, gold)
            validation_manifest = write_manifest_with_shard(root / "features", "validation", base, reward, gold)
            config = self.training_config(
                train_manifest,
                validation_manifest,
                root / "out",
                epochs=10,
                learning_rate=0.0,
                patience=1,
            )
            result = train_router(config, resume=False)
            self.assertTrue(len(result["history"]), 2)
            self.assertTrue(Path(result["training_log"]).exists())
            import json
            log = json.loads(Path(result["training_log"]).read_text())
            self.assertTrue(log["early_stopped"])
            self.assertEqual(log["stopped_epoch"], 1)
            self.assertEqual(log["patience_bad_epochs"], 1)

    def test_resume_restores_patience_bad_epochs_and_does_not_duplicate_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.zeros(8, 20)
            reward = torch.zeros(8, 20)
            gold = torch.zeros(8, dtype=torch.long)
            train_manifest = write_manifest_with_shard(root / "features", "train", base, reward, gold)
            validation_manifest = write_manifest_with_shard(root / "features", "validation", base, reward, gold)
            output_dir = root / "out"
            config = self.training_config(
                train_manifest,
                validation_manifest,
                output_dir,
                epochs=2,
                learning_rate=0.0,
                patience=5,
            )
            first = train_router(config, resume=False)
            first_initial = first["initial_validation"]["nll"]
            first_last = torch.load(first["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(first_last["patience_bad_epochs"], 1)
            resumed_config = RouterTrainingConfig(**{**config.__dict__, "epochs": 3})
            resumed = train_router(resumed_config, resume=True)
            self.assertEqual(resumed["initial_validation"]["nll"], first_initial)
            self.assertIsNotNone(resumed["resume_start_validation"])
            self.assertEqual([row["epoch"] for row in resumed["history"]], [0, 1, 2])
            resumed_last = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(resumed_last["patience_bad_epochs"], 2)
            self.assertEqual(len(resumed_last["history"]), 3)

    def test_resume_validates_checkpoint_config_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.randn(8, 20)
            reward = torch.randn(8, 20)
            gold = torch.zeros(8, dtype=torch.long)
            train_manifest = write_manifest_with_shard(root / "features", "train", base, reward, gold)
            validation_manifest = write_manifest_with_shard(root / "features", "validation", base, reward, gold)
            output_dir = root / "out"
            config = self.training_config(train_manifest, validation_manifest, output_dir, epochs=1)
            train_router(config, resume=False)
            bad_config = RouterTrainingConfig(**{**config.__dict__, "learning_rate": 9e-4, "epochs": 2})
            with self.assertRaises(ValueError):
                train_router(bad_config, resume=True)
            Path(train_manifest).write_text(Path(train_manifest).read_text() + "\n")
            good_config_more_epochs = RouterTrainingConfig(**{**config.__dict__, "epochs": 2})
            with self.assertRaises(ValueError):
                train_router(good_config_more_epochs, resume=True)


if __name__ == "__main__":
    unittest.main()
