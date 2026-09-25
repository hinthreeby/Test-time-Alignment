from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from Method.CURA.core.cache_io import SCHEMA_VERSION, ShardedCuraDataset, atomic_json, atomic_shard, verify_cache
from Method.CURA.core.features import controller_features
from Method.CURA.core.fusion import fuse_policies
from Method.CURA.models.calibrator import (
    HeteroscedasticSignalCalibrator, RobustSignalCalibrator, StableRobustSignalCalibrator,
)
from Method.CURA.models.controller import CuraController
from Method.CURA.core.train import collate, compute_loss
from Method.CURA.core.config import load_config
from Method.CURA.core.audit_signals import artifact_path, audit
from Method.CURA.core.annotate_targets import annotate_row
from Method.CURA.core.generate import apply_inference_overrides


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

    def test_stable_calibrator_bounds_degenerate_scores(self):
        calibrator = StableRobustSignalCalibrator(2, z_clip=6.0)
        raw = torch.zeros(1, 10, 2)
        raw[0, -1, 0] = 1e6
        mu, log_var = calibrator(raw, torch.ones(1, 2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(log_var).all())
        self.assertLessEqual(float(mu.abs().max()), 6.0)

    def test_disagreement_clip_limits_penalty(self):
        output = self.controller(self.features, self.mask)
        extreme = self.mu.clone()
        extreme[:, 0, 0] = 1e6
        fused = fuse_policies(
            self.base, extreme, self.log_var, output,
            disagreement_penalty=0.1, disagreement_clip=4.0,
        )
        self.assertLessEqual(float(fused["disagreement"].max()), 4.0)

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
        self.assertTrue((fused["pre_projection_kl"] >= fused["kl"] - 1e-7).all())
        torch.testing.assert_close(fused["kl_limit_hit"], fused["pre_projection_kl"] > 0.05)
        torch.testing.assert_close(fused["probabilities"].sum(-1), torch.ones(3))

    def test_fixed_strength_is_reported_and_only_changed_by_kl_projection(self):
        output = self.controller(self.features, self.mask)
        output["strength"] = torch.full_like(output["strength"], 4.0)
        unbounded = fuse_policies(self.base, self.mu, self.log_var, output, epsilon_kl=None)
        torch.testing.assert_close(unbounded["requested_strength"], torch.full((3,), 4.0))
        torch.testing.assert_close(unbounded["projected_strength"], torch.full((3,), 4.0))
        self.assertFalse(unbounded["kl_limit_hit"].any())

        bounded = fuse_policies(self.base, self.mu, self.log_var, output, epsilon_kl=1e-4)
        torch.testing.assert_close(bounded["requested_strength"], torch.full((3,), 4.0))
        self.assertTrue(bounded["kl_limit_hit"].any())
        self.assertTrue((bounded["projected_strength"] <= bounded["requested_strength"]).all())

    def test_zero_gate_equals_base(self):
        output = self.controller(self.features, self.mask)
        output["gate"] = torch.zeros_like(output["gate"])
        fused = fuse_policies(self.base, self.mu, self.log_var, output)
        torch.testing.assert_close(fused["probabilities"], torch.softmax(self.base, -1))

    def test_fixed_gate_override_preserves_strength(self):
        output = self.controller(self.features, self.mask)
        original_strength = output["strength"].clone()
        args = SimpleNamespace(ablation="none", fixed_gate=0.75, fixed_lambda=None)
        overridden = apply_inference_overrides(output, args)
        torch.testing.assert_close(overridden["gate"], torch.full((3,), 0.75))
        torch.testing.assert_close(overridden["strength"], original_strength)

    def test_no_gate_override_takes_full_guided_policy(self):
        output = self.controller(self.features, self.mask)
        args = SimpleNamespace(ablation="no-gate", fixed_gate=None, fixed_lambda=3.0)
        overridden = apply_inference_overrides(output, args)
        torch.testing.assert_close(overridden["gate"], torch.ones(3))
        torch.testing.assert_close(overridden["strength"], torch.full((3,), 3.0))

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
        self.assertGreaterEqual(float(pieces["utility"]), 0.0)
        self.assertIsNotNone(calibrator.networks[0][0].weight.grad)
        self.assertIsNotNone(controller.selector[-1].weight.grad)

    def test_paper_loss_ignores_gold_outside_natural_top_k(self):
        config, _ = load_config("Method/CURA/configs/sentiment_paper.json")
        rows = [{
            "base_logits": torch.randn(6), "raw_scores": torch.randn(3, 6),
            "signal_mask": torch.ones(3, dtype=torch.bool), "gold_index": 0,
            "gold_in_top_k": False, "position": 0, "step": 0, "prefix_length": 8,
            "target_utilities": torch.linspace(0, 1, 6),
        } for _ in range(2)]
        batch = collate(rows)
        calibrator = HeteroscedasticSignalCalibrator(3, hidden_dim=8)
        mu, log_var = calibrator(batch["raw_scores"], batch["signal_mask"])
        features = controller_features(batch["base_logits"], mu, log_var, batch["signal_mask"])
        controller = CuraController(3, features.size(-1), hidden_dim=16, dropout=0.0, signal_costs=[1, .8, 1.1])
        loss, pieces = compute_loss(batch, calibrator, controller, config, torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(pieces["nll"]), 0.0)
        self.assertGreater(float(pieces["utility"]), 0.0)

    def test_astar_two_signal_loss_is_finite(self):
        config, _ = load_config("Method/CURA/configs/sentiment_astar.json")
        self.assertEqual(config["signals"], ["rad", "cdq"])
        rows = [{
            "base_logits": torch.randn(8), "raw_scores": torch.randn(2, 8),
            "signal_mask": torch.ones(2, dtype=torch.bool), "gold_index": 0,
            "gold_in_top_k": True, "position": index, "step": index, "prefix_length": 10,
            "target_utilities": torch.rand(8),
        } for index in range(3)]
        batch = collate(rows)
        calibrator = HeteroscedasticSignalCalibrator(2, hidden_dim=8)
        mu, log_var = calibrator(batch["raw_scores"], batch["signal_mask"])
        features = controller_features(batch["base_logits"], mu, log_var, batch["signal_mask"])
        controller = CuraController(2, features.size(-1), hidden_dim=16, dropout=0.0, signal_costs=[1, .8])
        loss, pieces = compute_loss(batch, calibrator, controller, config, torch.device("cpu"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(pieces["utility"]), 0.0)

    def test_sentiment_genarm_config_uses_new_artifact(self):
        config, _ = load_config("Method/CURA/configs/sentiment_astar_genarm.json")
        self.assertEqual(config["signals"], ["rad", "cdq", "genarm"])
        self.assertEqual(config["signal_objectives"]["genarm"], "sentiment")
        self.assertEqual(
            config["signal_adapter_configs"]["genarm"]["arm_model"],
            "genarm-gpt2-small-sentiment",
        )
        self.assertEqual(
            artifact_path(config, "genarm").name,
            "genarm-gpt2-small-sentiment",
        )
        self.assertTrue(audit(config)["valid"])


class CuraCacheTests(unittest.TestCase):
    def test_rollout_targets_score_response_without_prompt(self):
        class BaseModel:
            def generate(self, initial, **kwargs):
                suffix = torch.full((initial.size(0), 1), 9, dtype=initial.dtype)
                return torch.cat([initial, suffix], dim=1)

        class BaseTokenizer:
            eos_token_id = 0

            def __init__(self):
                self.decoded = []

            def batch_decode(self, rows, **kwargs):
                self.decoded = [row.tolist() for row in rows]
                return [" ".join(map(str, row)) for row in self.decoded]

        class Encoded(dict):
            def to(self, device):
                return self

        class EvaluatorTokenizer:
            def __call__(self, texts, **kwargs):
                return Encoded(input_ids=torch.ones(len(texts), 2, dtype=torch.long))

        class Evaluator:
            def __call__(self, **kwargs):
                count = kwargs["input_ids"].size(0)
                return SimpleNamespace(logits=torch.tensor([[0.0, 1.0]]).repeat(count, 1))

        tokenizer = BaseTokenizer()
        row = {
            "prompt_id": "p0", "step": 2,
            "prefix_token_ids": torch.tensor([100, 101, 7, 8]),
            "candidate_token_ids": torch.tensor([1, 2]),
        }
        args = SimpleNamespace(
            candidate_batch_size=10, rollouts=1, seed=42, rollout_tokens=1,
            top_p=0.95, temperature=1.0, positive_label=1,
        )
        annotated = annotate_row(
            row, BaseModel(), tokenizer, Evaluator(), EvaluatorTokenizer(), args, torch.device("cpu")
        )
        self.assertEqual(tokenizer.decoded, [[7, 8, 1, 9], [7, 8, 2, 9]])
        self.assertEqual(annotated["target_utilities"].numel(), 2)

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
