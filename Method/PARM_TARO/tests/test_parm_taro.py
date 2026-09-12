from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from PARM_TARO.adapters.parm_adapter import (
    inspect_parm_adapter_config,
    named_preference_to_parm,
    set_parm_preference,
)
from PARM_TARO.adapters.router_adapter import (
    RouterSequenceState,
    SmartRouterLambdaProvider,
)
from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.config import ParmTaroConfig
from PARM_TARO.decoding.adaptive import ParmTaroDecoder
from PARM_TARO.decoding.static_regression import parm_guided_logprobs
from router_v2.smart_checkpoint import save_smart_checkpoint
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import hash_tree


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GENARM_PYTHON = Path(
    "/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python"
)


class TinyTokenizer:
    eos_token_id = 6

    def __len__(self) -> int:
        return 7

    def get_vocab(self) -> dict[str, int]:
        return {f"token_{index}": index for index in range(7)}

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        if not text:
            return {"input_ids": torch.empty((1, 0), dtype=torch.long)}
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, ids: list[int], **_: object) -> str:
        return " ".join(f"token_{token_id}" for token_id in ids)


class TinyBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(vocab_size=7)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 7, device=input_ids.device)
        logits[..., 2] = 2.0 + self.anchor
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class TinyPreferenceGuide(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pref_vec = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(vocab_size=8)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        logits[..., 3] = 1.0 + 4.0 * self.pref_vec[0] + self.anchor
        logits[..., 4] = 1.0 + 4.0 * self.pref_vec[1]
        return SimpleNamespace(logits=logits, past_key_values=None)


class SpyLambdaProvider:
    def __init__(self) -> None:
        self.preferences: list[torch.Tensor] = []
        self.score_counts_at_predict: list[int] = []

    def predict(
        self,
        base_values: torch.Tensor,
        guide_values: torch.Tensor,
        *,
        position: int,
        preference: torch.Tensor,
        state: RouterSequenceState,
    ) -> torch.Tensor:
        self.preferences.append(preference.detach().cpu().clone())
        self.score_counts_at_predict.append(len(state.selected_base_scores))
        self.assert_detached(base_values, guide_values)
        return torch.full((1, 1), 0.75, device=base_values.device)

    @staticmethod
    def assert_detached(*values: torch.Tensor) -> None:
        if any(value.requires_grad for value in values):
            raise AssertionError("Router features must be detached")

    @staticmethod
    def observe_selected_base_score(
        state: RouterSequenceState, score: float
    ) -> None:
        state.selected_base_scores.append(float(score))


def preference_router_config(**overrides: object) -> SmartRouterConfig:
    values: dict[str, object] = {
        "variant": "custom",
        "vocab_size": 7,
        "top_k": 2,
        "token_embedding_dim": 3,
        "candidate_hidden_dim": 4,
        "candidate_feature_mode": "token_aware_mean_max",
        "use_confidence": True,
        "use_position": True,
        "use_history": False,
        "use_preference": True,
        "max_position": 8,
        "history_hidden_dim": 4,
        "preference_dim": 2,
        "preference_hidden_dim": 4,
        "preference_embedding_dim": 3,
        "fusion_hidden_dim": 8,
        "fusion_bottleneck_dim": 4,
        "device": "cpu",
    }
    values.update(overrides)
    return SmartRouterConfig(**values)


class ConfigAndAlignmentTests(unittest.TestCase):
    def test_shipped_config_supports_two_objective_alpha(self) -> None:
        config = ParmTaroConfig.load_json(
            WORKSPACE_ROOT / "PARM_TARO/configs/parm_taro_tulu2.json"
        )
        self.assertEqual(config.preference_dim, 2)
        self.assertEqual(config.preference, (0.5, 0.5))

    def test_config_schema_is_strict_and_versioned(self) -> None:
        path = WORKSPACE_ROOT / "PARM_TARO/schemas/parm_taro_config.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_config_rejects_alpha_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "preference length"):
            ParmTaroConfig(preference_dim=2, preference=(1.0,))

    def test_token_alignment_allows_only_guide_suffix_tokens(self) -> None:
        alignment = PARMTokenAlignment.validate(
            TinyTokenizer(), base_vocab_size=7, guide_vocab_size=8
        )
        guide = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        torch.testing.assert_close(alignment.align_guide_logits(guide), guide[:, :7])

    def test_token_alignment_rejects_non_prefix_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot omit"):
            PARMTokenAlignment.validate(
                TinyTokenizer(), base_vocab_size=7, guide_vocab_size=6
            )


class StaticEquationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = torch.tensor([[2.0, -1.0, 0.5]])
        self.guide = torch.tensor([[-0.5, 3.0, 1.0]])

    def test_lambda_zero_is_base_distribution(self) -> None:
        actual = parm_guided_logprobs(self.base, self.guide, 0.0)
        torch.testing.assert_close(actual, F.log_softmax(self.base, dim=-1))

    def test_constant_lambda_is_static_parm_equivalent(self) -> None:
        scale = 1.7
        expected = F.log_softmax(
            F.log_softmax(self.base, dim=-1)
            + scale * F.log_softmax(self.guide, dim=-1),
            dim=-1,
        )
        actual = parm_guided_logprobs(self.base, self.guide, scale)
        torch.testing.assert_close(actual, expected)

    def test_source_logits_are_frozen_inputs(self) -> None:
        base = self.base.clone().requires_grad_()
        guide = self.guide.clone().requires_grad_()
        output = parm_guided_logprobs(base, guide, torch.tensor([[0.4]]))
        self.assertFalse(output.requires_grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(guide.grad)

    def test_static_equation_matches_vendored_operator_raw_sum(self) -> None:
        python = GENARM_PYTHON if GENARM_PYTHON.is_file() else Path(os.sys.executable)
        command = """
import json
import torch
from PARM_TARO.adapters.parm_adapter import activate_vendored_parm_runtime
activate_vendored_parm_runtime()
from model_arithmetic.operators import Constant
base = torch.log_softmax(torch.tensor([[2.0, -1.0, 0.5]]), dim=-1)
guide = torch.log_softmax(torch.tensor([[-0.5, 3.0, 1.0]]), dim=-1)
formula = Constant(base) + 1.7 * Constant(guide)
result = torch.log_softmax(formula.evaluate({}, normalize=False), dim=-1)
print(json.dumps(result.tolist()))
"""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [str(python), "-c", command],
            cwd=WORKSPACE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        vendored = torch.tensor(json.loads(completed.stdout.strip().splitlines()[-1]))
        actual = parm_guided_logprobs(self.base, self.guide, 1.7)
        torch.testing.assert_close(actual, vendored)


class PreferenceAndRouterTests(unittest.TestCase):
    def test_named_alpha_is_reordered_for_author_pblora(self) -> None:
        named = torch.tensor([[0.8, 0.2]])
        parm = named_preference_to_parm(named)
        torch.testing.assert_close(parm, torch.tensor([[0.2, 0.8]]))

    def test_alpha_changes_parm_guide(self) -> None:
        guide = TinyPreferenceGuide()
        ids = torch.tensor([[1]])
        set_parm_preference(guide, torch.tensor([1.0, 0.0]))
        first = guide(ids).logits.detach().clone()
        set_parm_preference(guide, torch.tensor([0.0, 1.0]))
        second = guide(ids).logits.detach().clone()
        self.assertFalse(torch.equal(first, second))
        self.assertGreater(first[0, 0, 3], second[0, 0, 3])
        self.assertGreater(second[0, 0, 4], first[0, 0, 4])

    def test_non_pblora_adapter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "obj_num": 2}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "PBLORA"):
                inspect_parm_adapter_config(root, expected_preference_dim=2)

    def test_alpha_reaches_router_and_changes_lambda(self) -> None:
        torch.manual_seed(17)
        provider = SmartRouterLambdaProvider(
            SmartTokenRouter(preference_router_config()), preference_dim=2
        )
        base = F.log_softmax(torch.randn(1, 7), dim=-1)
        guide = F.log_softmax(torch.randn(1, 7), dim=-1)
        first = provider.predict(
            base,
            guide,
            position=0,
            preference=torch.tensor([[1.0, 0.0]]),
            state=RouterSequenceState(),
        )
        second = provider.predict(
            base,
            guide,
            position=0,
            preference=torch.tensor([[0.0, 1.0]]),
            state=RouterSequenceState(),
        )
        self.assertFalse(torch.equal(first, second))

    def test_router_checkpoint_requires_tokenizer_hash(self) -> None:
        model = SmartTokenRouter(preference_router_config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            save_smart_checkpoint(
                model,
                path,
                metadata={"tokenizer_semantic_sha256": "correct"},
            )
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                SmartRouterLambdaProvider.from_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_vocab_size=7,
                    preference_dim=2,
                    tokenizer_semantic_sha256="wrong",
                )

    def test_rad_history_router_is_rejected_for_parm(self) -> None:
        router = SmartTokenRouter(
            preference_router_config(use_preference=False)
        )
        with self.assertRaisesRegex(ValueError, "preference-enabled"):
            SmartRouterLambdaProvider(router, preference_dim=2)


class AdaptiveGenerationTests(unittest.TestCase):
    def test_end_to_end_generation_is_adaptive_frozen_and_causal(self) -> None:
        base = TinyBaseModel()
        guide = TinyPreferenceGuide()
        provider = SpyLambdaProvider()
        decoder = ParmTaroDecoder(
            base_model=base,
            guide_model=guide,
            tokenizer=TinyTokenizer(),
            alignment=PARMTokenAlignment.validate(
                TinyTokenizer(), base_vocab_size=7, guide_vocab_size=8
            ),
            lambda_provider=provider,
            device=torch.device("cpu"),
            preference_dim=2,
            top_k=2,
            max_new_tokens=3,
            do_sample=False,
            stop_on_eos=True,
        )
        result = decoder.generate(
            "prompt", preference=torch.tensor([1.0, 0.0]), seed=2026
        )
        self.assertEqual(len(result.selected_token_ids), 3)
        self.assertEqual(result.lambda_history, [0.75, 0.75, 0.75])
        self.assertEqual(provider.score_counts_at_predict, [0, 1, 2])
        self.assertTrue(
            all(torch.equal(value, torch.tensor([[1.0, 0.0]])) for value in provider.preferences)
        )
        torch.testing.assert_close(guide.pref_vec, torch.tensor([0.0, 1.0]))
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in guide.parameters()))

    def test_router_control_alpha_is_separate_from_guide_alpha(self) -> None:
        base = TinyBaseModel()
        guide = TinyPreferenceGuide()
        provider = SpyLambdaProvider()
        decoder = ParmTaroDecoder(
            base_model=base,
            guide_model=guide,
            tokenizer=TinyTokenizer(),
            alignment=PARMTokenAlignment.validate(
                TinyTokenizer(), base_vocab_size=7, guide_vocab_size=8
            ),
            lambda_provider=provider,
            device=torch.device("cpu"),
            preference_dim=2,
            top_k=2,
            max_new_tokens=1,
            do_sample=False,
            stop_on_eos=True,
        )
        decoder.generate(
            "prompt",
            preference=torch.tensor([1.0, 0.0]),
            router_preference=torch.tensor([0.0, 1.0]),
            seed=2026,
        )
        torch.testing.assert_close(provider.preferences[0], torch.tensor([[0.0, 1.0]]))
        # PBLORA receives author order [harmlessness, helpfulness].
        torch.testing.assert_close(guide.pref_vec, torch.tensor([0.0, 1.0]))


class VendoredAndProtectedTreeTests(unittest.TestCase):
    def test_vendored_parm_dependencies_resolve_in_proven_environment(self) -> None:
        python = GENARM_PYTHON if GENARM_PYTHON.is_file() else Path(os.sys.executable)
        command = (
            "from PARM_TARO.adapters.parm_adapter import "
            "activate_vendored_parm_runtime as load; "
            "r=load(); print(r.origins['peft']); "
            "print(r.origins['model_arithmetic'])"
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [str(python), "-c", command],
            cwd=WORKSPACE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PARM/peft/src/peft", completed.stdout)
        self.assertIn(
            "PARM/language-model-arithmetic/src/model_arithmetic",
            completed.stdout,
        )

    def test_original_parm_generation_module_loads_read_only(self) -> None:
        python = GENARM_PYTHON if GENARM_PYTHON.is_file() else Path(os.sys.executable)
        command = (
            "from PARM_TARO.adapters.parm_adapter import "
            "load_original_parm_generation_module as load; "
            "m=load(); print(m.__file__); "
            "print(callable(m.get_model_arithmetic))"
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [str(python), "-c", command],
            cwd=WORKSPACE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PARM/code/evaluation/generate_outputs.py", completed.stdout)
        self.assertTrue(completed.stdout.strip().endswith("True"))

    def test_parm_tree_matches_pre_stage7_snapshot(self) -> None:
        baseline_path = WORKSPACE_ROOT / "PARM_TARO/reports/parm_baseline_hash.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = hash_tree(WORKSPACE_ROOT / "PARM")
        self.assertEqual(current["file_count"], baseline["file_count"])
        self.assertEqual(current["total_bytes"], baseline["total_bytes"])
        self.assertEqual(current["tree_sha256"], baseline["tree_sha256"])
        self.assertEqual(list((WORKSPACE_ROOT / "PARM").rglob("__pycache__")), [])


if __name__ == "__main__":
    unittest.main()
