from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import torch

from Method.CURA.core.cache_io import SCHEMA_VERSION, ShardedCuraDataset, atomic_json, atomic_shard, verify_cache
from Method.CURA.core.features import controller_features
from Method.CURA.core.fusion import fuse_policies
from Method.CURA.models.calibrator import HeteroscedasticSignalCalibrator, RobustSignalCalibrator
from Method.CURA.models.controller import CuraController
from Method.CURA.core.train import collate, compute_loss
from Method.CURA.core.config import load_config


class CuraMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = torch.randn(3, 10)
        self.raw = torch.randn(3, 10, 4)
        self.mask = torch.ones(3, 4, dtype=torch.bool)
        self.calibrator = RobustSignalCalibrator(4)
        self.mu, self.log_var = self.calibrator(self.raw, self.mask)
        self.features = controller_features(self.base, self.mu, self.log_var, self.mask)
        self.controller = CuraController(4, self.features.size(-1), hidden_dim=16, dropout=0.0)

    def test_calibration_is_finite_and_uncertainty_positive(self):
        self.assertTrue(torch.isfinite(self.mu).all())
        self.assertTrue((self.log_var.exp() > 0).all())

    def test_learned_calibrator_is_candidate_specific(self):
        calibrator = HeteroscedasticSignalCalibrator(4, hidden_dim=8)
        mu, log_var = calibrator(self.raw, self.mask)
        self.assertEqual(mu.shape, self.raw.shape)
        self.assertTrue(torch.isfinite(log_var).all())
        self.assertGreater(float(log_var.var(dim=1).mean()), 0.0)

    def test_controller_ranges(self):
        output = self.controller(self.features, self.mask)
        torch.testing.assert_close(output["weights"].sum(-1), torch.ones(3))
        self.assertTrue(((output["gate"] >= 0) & (output["gate"] <= 1)).all())
        self.assertTrue(((output["strength"] >= 0) & (output["strength"] <= 3)).all())
        routing = controller_features(self.base, self.mu, self.log_var, self.mask)[:, :8]
        selected = self.controller.select_mask(routing, budget=2)
        self.assertTrue((selected.sum(-1) == 2).all())

    def test_fusion_respects_kl_budget(self):
        output = self.controller(self.features, self.mask)
        fused = fuse_policies(self.base, self.mu, self.log_var, output, epsilon_kl=0.05)
        self.assertTrue((fused["kl"] <= 0.05001).all())
        torch.testing.assert_close(fused["probabilities"].sum(-1), torch.ones(3))

    def test_zero_gate_equals_base(self):
        output = self.controller(self.features, self.mask)
        output["gate"] = torch.zeros_like(output["gate"])
        fused = fuse_policies(self.base, self.mu, self.log_var, output)
        torch.testing.assert_close(fused["probabilities"], torch.softmax(self.base, -1))

    def test_training_step_updates_controller(self):
        config, _ = load_config("Method/CURA/configs/sentiment.json")
        rows = []
        for index in range(4):
            rows.append({
                "base_logits": torch.randn(10), "raw_scores": torch.randn(3, 10),
                "signal_mask": torch.ones(3, dtype=torch.bool), "gold_index": index,
                "position": index, "step": index, "prefix_length": 12 + index,
            })
        batch = collate(rows)
        calibrator = RobustSignalCalibrator(3)
        mu, log_var = calibrator(batch["raw_scores"], batch["signal_mask"])
        features = controller_features(batch["base_logits"], mu, log_var, batch["signal_mask"])
        controller = CuraController(3, features.size(-1), hidden_dim=16, dropout=0.0, signal_costs=[1, 1, 1])
        optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3)
        before = controller.weight_head.weight.detach().clone()
        loss, _ = compute_loss(batch, calibrator, controller, config, torch.device("cpu"))
        loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, controller.weight_head.weight.detach()))

    def test_learned_calibration_preference_step(self):
        config, _ = load_config("Method/CURA/configs/sentiment_paper.json")
        rows = []
        for index in range(4):
            rows.append({
                "base_logits": torch.randn(10), "raw_scores": torch.randn(3, 10),
                "signal_mask": torch.ones(3, dtype=torch.bool), "gold_index": index,
                "position": index, "step": index, "prefix_length": 12 + index,
                "target_utilities": torch.linspace(0, 1, 10).roll(index),
            })
        batch = collate(rows)
        calibrator = HeteroscedasticSignalCalibrator(3, hidden_dim=8)
        mu, log_var = calibrator(batch["raw_scores"], batch["signal_mask"])
        features = controller_features(batch["base_logits"], mu, log_var, batch["signal_mask"])
        controller = CuraController(3, features.size(-1), hidden_dim=16, dropout=0.0, signal_costs=[1, .8, 1.1])
        loss, pieces = compute_loss(batch, calibrator, controller, config, torch.device("cpu"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(pieces["preference"]), 0.0)
        self.assertIsNotNone(calibrator.networks[0][0].weight.grad)
        self.assertIsNotNone(controller.selector[-1].weight.grad)


class CuraCacheTests(unittest.TestCase):
    def test_atomic_shard_verify_and_lazy_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"prompt_id": "p0", "step": 0, "prefix_token_ids": torch.arange(2), "candidate_token_ids": torch.arange(3),
                   "base_logits": torch.randn(3), "raw_scores": torch.randn(4, 3),
                   "signal_mask": torch.ones(4, dtype=torch.bool), "gold_index": 0}
            relative = "shards/shard_000000.pt"
            digest = atomic_shard({"schema_version": SCHEMA_VERSION, "rows": [row]}, root / relative)
            manifest = {"schema_version": SCHEMA_VERSION, "completed_shards": [
                {"id": 0, "file": relative, "num_steps": 1, "sha256": digest}
            ]}
            atomic_json(manifest, root / "manifest.json")
            self.assertTrue(verify_cache(root)["valid"])
            dataset = ShardedCuraDataset(root)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["prompt_id"], "p0")
            self.assertFalse((root / "shards/shard_000000.pt.tmp").exists())

    def test_train_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = load_config("Method/CURA/configs/sentiment.json")
            config["controller"].update({"hidden_dim": 16, "num_layers": 1, "dropout": 0.0})
            config["training"].update({"epochs": 1, "batch_size": 2, "mixed_precision": False})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            for split in ("train", "validation"):
                cache_dir = root / split
                rows = []
                for index in range(4):
                    rows.append({
                        "prompt_id": f"{split}-{index}", "step": index,
                        "prefix_token_ids": torch.arange(3),
                        "candidate_token_ids": torch.arange(5), "base_logits": torch.randn(5),
                        "raw_scores": torch.randn(3, 5), "signal_mask": torch.ones(3, dtype=torch.bool),
                        "gold_index": index % 5, "position": index, "prefix_length": 10 + index,
                    })
                relative = "shards/shard_000000.pt"
                digest = atomic_shard({"schema_version": SCHEMA_VERSION, "rows": rows}, cache_dir / relative)
                atomic_json({
                    "schema_version": SCHEMA_VERSION, "signals": config["signals"],
                    "objective": config["objective"], "split": split, "status": "complete", "failed_prompt_ids": [],
                    "completed_shards": [{"id": 0, "file": relative, "num_steps": len(rows), "sha256": digest}],
                }, cache_dir / "manifest.json")
            output = root / "cura.pt"
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "core/train.py"),
                "--config", str(config_path), "--cache-dir", str(root / "train"),
                "--validation-cache-dir", str(root / "validation"), "--output", str(output), "--device", "cpu",
            ], check=False, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())
            self.assertTrue((root / "cura.latest.pt").exists())
            state = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["signals"], ["rad", "cdq", "args"])


if __name__ == "__main__":
    unittest.main()
