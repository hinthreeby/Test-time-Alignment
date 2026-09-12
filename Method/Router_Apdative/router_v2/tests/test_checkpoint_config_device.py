from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from router_v2.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_payload,
    load_taro_checkpoint,
    save_taro_checkpoint,
    validate_checkpoint_payload,
)
from router_v2.config import TARORouterConfig
from router_v2.device import resolve_device
from router_v2.model import TAROTokenRouter, build_taro_router


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def small_config(**overrides: object) -> TARORouterConfig:
    values: dict[str, object] = {
        "vocab_size": 19,
        "top_k": 5,
        "token_embedding_dim": 8,
        "hidden_dim": 128,
    }
    values.update(overrides)
    return TARORouterConfig(**values)


class ConfigAndDeviceTests(unittest.TestCase):
    def test_shipped_configs_load_faithful_schema(self) -> None:
        config_dir = WORKSPACE_ROOT / "router_v2" / "configs"
        for path in sorted(config_dir.glob("taro_*.json")):
            with self.subTest(path=path.name):
                config = TARORouterConfig.load_json(path)
                self.assertEqual(config.schema_version, 2)
                self.assertEqual(config.method_label, "TARO")
                self.assertEqual(config.architecture, "taro_topk_flatten_tanh")
                self.assertEqual(config.activation, "tanh")
                self.assertEqual(config.hidden_dim, 128)

    def test_schema_files_are_valid_json_and_version_two(self) -> None:
        schema_dir = WORKSPACE_ROOT / "router_v2" / "schemas"
        for path in sorted(schema_dir.glob("taro_*.schema.json")):
            with self.subTest(path=path.name):
                with path.open("r", encoding="utf-8") as handle:
                    schema = json.load(handle)
                self.assertEqual(schema["type"], "object")
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(
                    schema["properties"]["schema_version"]["const"],
                    2,
                )

    def test_config_rejects_non_faithful_contracts(self) -> None:
        with self.assertRaises(ValueError):
            small_config(schema_version=1)
        with self.assertRaises(ValueError):
            small_config(method_label="TARO-like")
        with self.assertRaises(ValueError):
            small_config(architecture="pooled_gelu")
        with self.assertRaises(ValueError):
            small_config(activation="gelu")
        with self.assertRaises(ValueError):
            small_config(hidden_dim=64)
        with self.assertRaises(ValueError):
            small_config(mode="taro_topk_nll", entropy_weight=0.1)
        with self.assertRaises(ValueError):
            TARORouterConfig.from_dict({"unknown": True})

    def test_auto_prefers_cuda_and_falls_back_to_cpu(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), torch.device("cuda"))
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(resolve_device("cuda"), torch.device("cpu"))
            with self.assertRaises(RuntimeError):
                resolve_device("cuda", allow_cpu_fallback=False)

    def test_model_factory_honors_cpu_config(self) -> None:
        model = build_taro_router(small_config(device="cpu"))
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_output_deterministic(self) -> None:
        torch.manual_seed(88)
        config = small_config()
        model = TAROTokenRouter(config).eval()
        base_logits = torch.randn(2, config.vocab_size)
        reward_logits = torch.randn(2, config.vocab_size)
        expected = model.forward_from_full_logits(base_logits, reward_logits)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "router.pt"
            save_taro_checkpoint(
                model,
                path,
                training_state={"step": 7},
                metadata={"run": "unit-test"},
            )
            loaded, payload = load_taro_checkpoint(path)
            loaded.eval()
            actual = loaded.forward_from_full_logits(base_logits, reward_logits)

            torch.testing.assert_close(actual.alpha, expected.alpha, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.guided_logits,
                expected.guided_logits,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual.topk.base_token_ids,
                expected.topk.base_token_ids,
            )
            torch.testing.assert_close(
                actual.topk.reward_token_ids,
                expected.topk.reward_token_ids,
            )
            self.assertEqual(payload["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(payload["format"], CHECKPOINT_FORMAT)
            self.assertEqual(payload["method_label"], "TARO")
            self.assertEqual(payload["mode"], config.mode)
            self.assertEqual(payload["vocab_size"], config.vocab_size)
            self.assertEqual(payload["top_k"], config.top_k)
            self.assertEqual(
                payload["token_embedding_dim"],
                config.token_embedding_dim,
            )
            self.assertEqual(payload["hidden_dim"], 128)
            self.assertEqual(payload["entropy_weight"], config.entropy_weight)
            self.assertEqual(payload["training_state"]["step"], 7)
            self.assertEqual(payload["metadata"]["run"], "unit-test")

    def test_checkpoint_refuses_overwrite_by_default(self) -> None:
        model = TAROTokenRouter(small_config())
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "router.pt"
            save_taro_checkpoint(model, path)
            with self.assertRaises(FileExistsError):
                save_taro_checkpoint(model, path)

    def test_schema_one_checkpoint_fails_clearly(self) -> None:
        old_payload = {
            "schema_version": 1,
            "format": "router_v2.taro_checkpoint",
            "method_label": "TARO",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "old-router.pt"
            torch.save(old_payload, path)

            with self.assertRaisesRegex(
                ValueError,
                "Incompatible TARO checkpoint schema_version",
            ):
                load_taro_checkpoint(path)

    def test_mirrored_checkpoint_fields_must_match_config(self) -> None:
        payload = build_checkpoint_payload(TAROTokenRouter(small_config()))
        payload["top_k"] = payload["top_k"] + 1

        with self.assertRaisesRegex(ValueError, "does not match embedded config"):
            validate_checkpoint_payload(payload)


if __name__ == "__main__":
    unittest.main()
