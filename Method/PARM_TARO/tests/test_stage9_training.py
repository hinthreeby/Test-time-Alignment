from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.adapters.parm_adapter import freeze_for_inference
from PARM_TARO.training.alpha import (
    build_alpha_controls,
    is_near_tie,
    response_objective_weights,
    sample_alpha,
)
from PARM_TARO.training.alpha_collapse import (
    FIXED_LAMBDAS,
    endpoint_distribution_sensitivity,
    finite_difference_lambda_alpha,
    fixed_lambda_task_metrics,
)
from PARM_TARO.training.alpha_pilot_v2 import (
    endpoint_sensitivity_loss,
    endpoint_logit_sensitivity_loss,
    exact_score_margin_sensitivity,
    lambda_delta_to_logit_delta,
    linear_lambda_floor_loss,
    preference_gain_objective,
    reset_lambda_head,
    router_parameter_groups,
    term_gradient_norms,
    transplant_no_alpha_router,
)
from PARM_TARO.training.alpha_residual_router import (
    AlphaResidualConfig,
    alpha_path_parameter_groups,
    build_from_v2_router,
    load_alpha_residual_checkpoint,
    save_alpha_residual_checkpoint,
)
from PARM_TARO.training.alpha_production import (
    REQUIRED_PRODUCTION_CHECKS,
    _run_phase,
    production_checks,
    require_pilot_v3_pass,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.constant_control import (
    ConstantControlAccumulator,
    alpha_vectors_identical,
)
from PARM_TARO.training.constant_control_audit import (
    source_failure_is_constant_only,
)
from PARM_TARO.training.data import (
    PROMPT_TEMPLATE,
    MultiObjectiveExample,
    deterministic_example_order,
    load_examples,
    tokenize_response,
)
from PARM_TARO.training.engine import (
    _assert_optimizer_fp32,
    _validate_stage_order,
    evaluate_comparisons,
    train_stage,
)
from PARM_TARO.training.objective import (
    dual_response_objective,
    preference_sensitive_objective,
)
from PARM_TARO.training.pilot_config import AlphaPreferencePilotConfig
from PARM_TARO.training.pilot_v2_config import AlphaPreferencePilotV2Config
from PARM_TARO.training.pilot_v3_config import AlphaPreferencePilotV3Config
from PARM_TARO.training.production_config import AlphaPreferenceProductionConfig
from PARM_TARO.training.online import (
    FrozenSequenceDistributions,
    compute_frozen_guide_logprobs,
    compute_frozen_sequence_distributions,
)
from PARM_TARO.training.routing import router_parm_route, static_parm_route
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    FrozenBaseModelView,
    _load_frozen_model_pair,
    _model_kwargs,
    ParmTrainingRuntime,
    audit_training_prerequisites,
    capture_adapter_layer_state,
    resolved_paths,
)
from PARM_TARO.training.stage9_final import (
    _passed_stage,
    constant_audit_authorizes_correction,
)
from PARM_TARO.scripts.pilot_alpha_preference import (
    _evaluate as evaluate_alpha_preference_pilot,
    _require_diagnostic,
)
from PARM_TARO.scripts.pilot_alpha_preference_v2 import (
    _evaluate as evaluate_alpha_preference_pilot_v2,
    _require_failure_diagnostic,
)
from PARM_TARO.scripts.pilot_alpha_preference_v3 import (
    _require_diagnostic as require_alpha_path_diagnostic,
    _set_trainable_phase,
)
from router_v2.config import TARORouterConfig
from router_v2.model import TAROTokenRouter
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import assert_optimizer_contains_only, hash_tree
from router_v2.cache.io import sha256_file


class OffsetTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    pad_token_id = 0
    unk_token = "<unk>"
    unk_token_id = 3
    padding_side = "right"
    model_max_length = 256

    def __len__(self) -> int:
        return 11

    def get_vocab(self) -> dict[str, int]:
        return {f"token_{index}": index for index in range(len(self))}

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        ids = [4 + (ord(character) % 7) for character in text]
        return {
            "input_ids": ids,
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class ScriptedBoundaryTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, *, crossing: bool, zero_length: bool = False) -> None:
        self.crossing = crossing
        self.zero_length = zero_length

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"piece_{token_id}"

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        boundary = text.index(" ASSISTANT:") + len(" ASSISTANT:")
        response_length = len(text) - boundary
        if response_length < 3:
            raise ValueError("Scripted response must contain at least three characters")
        input_ids = [10]
        offsets = [(0, boundary)]
        if self.crossing:
            offsets[0] = (0, boundary - 1)
            input_ids.append(11)
            offsets.append((boundary - 1, boundary + 1))
            response_start = boundary + 1
        else:
            response_start = boundary
        if self.zero_length:
            input_ids.extend((12, 13))
            offsets.extend(((0, 0), (boundary, boundary)))
        for index in range(response_start, len(text)):
            input_ids.append(20 + index - response_start)
            offsets.append((index, index + 1))
        return {"input_ids": input_ids, "offset_mapping": offsets}


class TinyCausalModel(nn.Module):
    def __init__(self, vocab_size: int, *, preference_aware: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(vocab_size=vocab_size)
        if preference_aware:
            self.pref_vec = nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        vocab = self.config.vocab_size
        token_axis = torch.arange(vocab, device=input_ids.device).float()
        logits = token_axis.reshape(1, 1, -1).expand(batch, length, -1).clone()
        logits = logits * 0.05 + self.anchor
        if hasattr(self, "pref_vec"):
            logits[..., 5] += 2.0 * self.pref_vec[0]
            logits[..., 6] += 2.0 * self.pref_vec[1]
        return SimpleNamespace(logits=logits)


class TinyAdapterModel(TinyCausalModel):
    def __init__(self) -> None:
        super().__init__(11)
        self.adapter_weight = nn.Parameter(torch.tensor(3.0))
        self._disable_adapters = False
        self._active_adapter = "default"

    @property
    def adapter_enabled(self) -> bool:
        return not self._disable_adapters

    @property
    def active_adapters(self) -> list[str]:
        return [self._active_adapter]

    def enable_adapters(self, enabled: bool) -> None:
        self._disable_adapters = not enabled
        self.adapter_weight.requires_grad_(enabled)

    @contextmanager
    def disable_adapter(self):
        self.enable_adapters(False)
        try:
            yield
        finally:
            # This mirrors vendored PEFT: it re-enables the adapter and marks
            # its active weights trainable, even if the model was frozen.
            self.enable_adapters(True)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        output = super().forward(input_ids, **kwargs)
        if self.adapter_enabled:
            output.logits[..., 7] += self.adapter_weight
        return output


class AttachedTinyAdapter(nn.Module):
    def __init__(self, backbone: TinyCausalModel) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.adapter_weight = nn.Parameter(torch.tensor(3.0))
        self._disable_adapters = False
        self._active_adapter = "default"

    @property
    def adapter_enabled(self) -> bool:
        return not self._disable_adapters

    @property
    def active_adapters(self) -> list[str]:
        return [self._active_adapter]

    def enable_adapters(self, enabled: bool) -> None:
        self._disable_adapters = not enabled
        self.adapter_weight.requires_grad_(enabled)

    @contextmanager
    def disable_adapter(self):
        self.enable_adapters(False)
        try:
            yield
        finally:
            self.enable_adapters(True)

    def forward(self, *args: object, **kwargs: object) -> SimpleNamespace:
        output = self.backbone(*args, **kwargs)
        if self.adapter_enabled:
            output.logits[..., 7] += self.adapter_weight
        return output


class SpyAutoModel:
    calls: list[tuple[Path, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, path: Path, **kwargs: object) -> TinyCausalModel:
        cls.calls.append((Path(path), dict(kwargs)))
        return TinyCausalModel(11)


class SpyPeftModel:
    calls: list[tuple[nn.Module, Path, bool]] = []

    @classmethod
    def from_pretrained(
        cls,
        backbone: TinyCausalModel,
        path: Path,
        *,
        is_trainable: bool,
    ) -> AttachedTinyAdapter:
        cls.calls.append((backbone, Path(path), is_trainable))
        return AttachedTinyAdapter(backbone)


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.values = kwargs


def example(
    sample_id: str = "sample-0",
    *,
    better: int = 0,
    safer: int = 1,
) -> MultiObjectiveExample:
    return MultiObjectiveExample(
        sample_id=sample_id,
        source_index=0,
        prompt="prompt",
        responses=(" response zero", " response one"),
        better_response_id=better,
        safer_response_id=safer,
    )


def frozen_sequence(seed: int = 0) -> FrozenSequenceDistributions:
    generator = torch.Generator().manual_seed(seed)
    base = F.log_softmax(torch.randn(1, 3, 11, generator=generator), dim=-1)
    guide = F.log_softmax(torch.randn(1, 3, 11, generator=generator), dim=-1)
    gold = torch.tensor([1, 2, 3])
    selected = base[0].gather(-1, gold.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
    return FrozenSequenceDistributions(
        base_logprobs=base,
        guide_logprobs=guide,
        gold_token_ids=gold,
        base_selected_logprobs=selected,
        position=torch.arange(3).unsqueeze(0),
    )


def taro_router() -> TAROTokenRouter:
    return TAROTokenRouter(
        TARORouterConfig(
            vocab_size=11,
            top_k=3,
            token_embedding_dim=4,
            hidden_dim=128,
            device="cpu",
        )
    )


def smart_router(*, use_preference: bool) -> SmartTokenRouter:
    return SmartTokenRouter(
        SmartRouterConfig(
            variant="custom",
            vocab_size=11,
            top_k=3,
            token_embedding_dim=4,
            candidate_hidden_dim=5,
            candidate_feature_mode="token_aware_mean_max",
            use_confidence=True,
            use_position=True,
            use_history=True,
            use_preference=use_preference,
            max_position=8,
            history_hidden_dim=4,
            preference_dim=2,
            preference_hidden_dim=4,
            preference_embedding_dim=3,
            fusion_hidden_dim=8,
            fusion_bottleneck_dim=4,
            device="cpu",
        )
    )


class AlphaAndObjectiveTests(unittest.TestCase):
    def test_sampled_alpha_is_deterministic_simplex_and_resume_stable(self) -> None:
        first = sample_alpha("id-7", epoch=2, seed=19)
        second = sample_alpha("id-7", epoch=2, seed=19)
        torch.testing.assert_close(first, second)
        self.assertAlmostEqual(float(first.sum()), 1.0)
        self.assertTrue(bool(((first > 0.0) & (first < 1.0)).all()))
        self.assertFalse(torch.equal(first, sample_alpha("id-7", epoch=3, seed=19)))

    def test_conflicting_labels_keep_both_responses(self) -> None:
        weights = response_objective_weights(
            torch.tensor([0.7, 0.3]),
            better_response_id=0,
            safer_response_id=1,
        )
        torch.testing.assert_close(weights, torch.tensor([0.7, 0.3]))
        self.assertTrue(is_near_tie(torch.tensor([0.5, 0.5]), 0.05))

    def test_agreeing_labels_put_all_weight_on_shared_preference(self) -> None:
        weights = response_objective_weights(
            torch.tensor([0.2, 0.8]),
            better_response_id=1,
            safer_response_id=1,
        )
        torch.testing.assert_close(weights, torch.tensor([0.0, 1.0]))

    def test_dual_response_loss_is_weighted_mean_token_nll(self) -> None:
        logits_0 = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        logits_1 = torch.tensor([[[0.5, 1.5], [1.5, 0.5]]])
        logprobs = tuple(F.log_softmax(value, dim=-1) for value in (logits_0, logits_1))
        gold = (torch.tensor([0, 1]), torch.tensor([1, 0]))
        weights = torch.tensor([0.25, 0.75])
        result = dual_response_objective(logprobs, gold, weights)
        expected = 0.25 * F.nll_loss(logprobs[0][0], gold[0]) + 0.75 * F.nll_loss(
            logprobs[1][0], gold[1]
        )
        torch.testing.assert_close(result.total_loss, expected)

    def test_controls_change_only_router_alpha_values(self) -> None:
        correct = torch.tensor([[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]])
        controls = build_alpha_controls(correct)
        torch.testing.assert_close(controls.correct, correct)
        torch.testing.assert_close(controls.shuffled, correct.roll(1, 0))
        torch.testing.assert_close(controls.constant, torch.full_like(correct, 0.5))

    def test_preference_objective_matches_pairwise_nll_strength_equation(self) -> None:
        logits_0 = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True
        )
        logits_1 = torch.tensor(
            [[[0.5, 1.5], [1.5, 0.5]]], requires_grad=True
        )
        logprobs = tuple(
            F.log_softmax(value, dim=-1) for value in (logits_0, logits_1)
        )
        gold = (torch.tensor([0, 1]), torch.tensor([1, 0]))
        weights = torch.tensor([0.75, 0.25])
        lambdas = (torch.full((1, 2, 1), 0.01),) * 2
        result = preference_sensitive_objective(
            logprobs,
            gold,
            weights,
            lambdas,
            preference_loss_weight=1.0,
            nll_weight=0.25,
            strength_weight=0.1,
            strength_target=0.02,
            pairwise_logit_scale=2.0,
        )
        quality = dual_response_objective(logprobs, gold, weights)
        scores = -quality.response_nll
        margin = (weights[0] - weights[1]) * (scores[0] - scores[1])
        expected_preference = F.softplus(-2.0 * margin)
        expected_strength = torch.tensor((0.02 - 0.01) ** 2)
        expected_total = (
            expected_preference
            + 0.25 * quality.total_loss
            + 0.1 * expected_strength
        )
        torch.testing.assert_close(result.preference_loss, expected_preference)
        torch.testing.assert_close(result.strength_loss, expected_strength)
        torch.testing.assert_close(result.total_loss, expected_total)

    def test_pairwise_tie_has_zero_score_gradient(self) -> None:
        logits_0 = torch.randn(1, 3, 5, requires_grad=True)
        logits_1 = torch.randn(1, 3, 5, requires_grad=True)
        logprobs = tuple(
            F.log_softmax(value, dim=-1) for value in (logits_0, logits_1)
        )
        result = preference_sensitive_objective(
            logprobs,
            (torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0])),
            torch.tensor([0.5, 0.5]),
            (torch.full((1, 3, 1), 0.02),) * 2,
            preference_loss_weight=1.0,
            nll_weight=0.25,
            strength_weight=0.0,
            strength_target=0.0,
            pairwise_logit_scale=1.0,
        )
        gradients = torch.autograd.grad(
            result.preference_loss, (logits_0, logits_1)
        )
        torch.testing.assert_close(
            result.preference_loss, torch.log(torch.tensor(2.0))
        )
        self.assertTrue(all(bool((gradient == 0).all()) for gradient in gradients))

    def test_fixed_lambda_metrics_report_base_nll_and_preference_margin(self) -> None:
        pair = (frozen_sequence(2), frozen_sequence(3))
        weights = torch.tensor([0.8, 0.2])
        metrics = fixed_lambda_task_metrics(pair, weights, FIXED_LAMBDAS[0])
        routes = tuple(static_parm_route(item, 0.0) for item in pair)
        direct = dual_response_objective(
            (routes[0].guided_logprobs, routes[1].guided_logprobs),
            (pair[0].gold_token_ids, pair[1].gold_token_ids),
            weights,
        )
        self.assertAlmostEqual(metrics.dual_response_nll, float(direct.total_loss))
        self.assertIsNotNone(metrics.preferred_vs_nonpreferred_margin)
        self.assertAlmostEqual(metrics.response_weight_difference, 0.6, places=6)

    def test_endpoint_and_router_alpha_sensitivity_are_measurable(self) -> None:
        left = frozen_sequence(4)
        right = replace(
            left,
            guide_logprobs=F.log_softmax(
                left.guide_logprobs + torch.linspace(0.0, 1.0, 11),
                dim=-1,
            ),
        )
        sensitivity = endpoint_distribution_sensitivity(left, right)
        self.assertGreater(sensitivity["mean_abs_logprob_delta"], 0.0)
        self.assertGreater(sensitivity["mean_js_divergence"], 0.0)
        router = smart_router(use_preference=True)
        derivative = finite_difference_lambda_alpha(router, left, 0.5)
        self.assertGreater(derivative["mean_abs_derivative"], 0.0)

    def test_alpha_preference_pilot_config_is_strict_and_isolated(self) -> None:
        config = AlphaPreferencePilotConfig.load_json(
            PROJECT_ROOT
            / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot.json"
        )
        self.assertEqual(config.min_control_lambda_delta, 0.001)
        self.assertEqual(
            config.output_dir,
            "results/parm_taro/training/v2_alpha_preference_pilot",
        )
        self.assertGreater(config.preference_loss_weight, 0.0)
        self.assertGreater(config.nll_weight, 0.0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            AlphaPreferencePilotConfig.from_dict(
                {**config.to_dict(), "weaken_pass_threshold": True}
            )
        schema = json.loads(
            (
                PROJECT_ROOT
                / "PARM_TARO/schemas/stage9_alpha_preference_pilot.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["min_control_lambda_delta"]["exclusiveMinimum"], 0)

    def test_preference_pilot_requires_full_objective_collapse_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            config = replace(
                AlphaPreferencePilotConfig(),
                required_diagnostic_path=str(path),
            )
            path.write_text(
                json.dumps(
                    {
                        "status": "PARTIAL",
                        "validation": {
                            "examples_evaluated": 10,
                            "examples_total": 500,
                        },
                        "conclusion": {"objective_collapse": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "full PASS"):
                _require_diagnostic(config)
            path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "validation": {
                            "examples_evaluated": 500,
                            "examples_total": 500,
                        },
                        "conclusion": {"objective_collapse": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "objective_collapse"):
                _require_diagnostic(config)
            complete = {
                "status": "PASS",
                "validation": {
                    "examples_evaluated": 500,
                    "examples_total": 500,
                    "tasks_evaluated": 2500,
                    "tasks_expected": 2500,
                    "full_validation": True,
                    "skipped_sample_ids": [],
                },
                "conclusion": {"objective_collapse": True},
            }
            path.write_text(json.dumps(complete), encoding="utf-8")
            self.assertEqual(_require_diagnostic(config), complete)

    def test_preference_pilot_evaluation_reports_required_controls(self) -> None:
        training_config = ParmRouterTrainingConfig(
            stage="v2_alpha",
            router_kind="smart",
            max_length=16,
            max_continuation_tokens=4,
            validation_alpha_grid=(0.2, 0.8),
        )
        pilot_config = replace(
            AlphaPreferencePilotConfig(),
            max_validation_samples=2,
        )
        runtime = SimpleNamespace(device=torch.device("cpu"))
        pair = (frozen_sequence(2), frozen_sequence(3))
        with patch(
            "PARM_TARO.scripts.pilot_alpha_preference.frozen_response_pair",
            return_value=pair,
        ):
            metrics = evaluate_alpha_preference_pilot(
                smart_router(use_preference=True),
                smart_router(use_preference=False),
                runtime,
                training_config,
                pilot_config,
                [example("first"), example("second", better=1, safer=0)],
            )
        for name in (
            "correct_preference_loss",
            "shuffled_preference_loss",
            "constant_preference_loss",
            "correct_nll",
            "no_alpha_nll",
            "correct_lambda",
            "endpoint_alpha_lambda_delta",
            "correct_vs_shuffled_lambda_delta",
            "correct_vs_constant_lambda_delta",
        ):
            self.assertIn(name, metrics)

    def test_v2_evaluation_reports_non_identical_constant_control(self) -> None:
        training_config = ParmRouterTrainingConfig(
            stage="v2_alpha",
            router_kind="smart",
            max_length=16,
            max_continuation_tokens=4,
            validation_alpha_grid=(0.0, 0.5, 1.0),
        )
        pilot_config = replace(
            AlphaPreferencePilotV2Config(),
            max_validation_samples=2,
        )
        runtime = SimpleNamespace(device=torch.device("cpu"))
        pair = (frozen_sequence(2), frozen_sequence(3))
        with patch(
            "PARM_TARO.scripts.pilot_alpha_preference_v2.frozen_response_pair",
            return_value=pair,
        ):
            metrics = evaluate_alpha_preference_pilot_v2(
                smart_router(use_preference=True),
                smart_router(use_preference=False),
                runtime,
                training_config,
                pilot_config,
                {"recommendation": {"pairwise_logit_scale": 1.0}},
                [example("first"), example("second", better=1, safer=0)],
            )
        control = metrics["constant_control"]
        self.assertEqual(control["total_tasks"], 6)
        self.assertEqual(control["identical_tasks"], 2)
        self.assertEqual(control["non_identical_tasks"], 4)
        self.assertEqual(
            control["per_alpha_helpfulness"]["0.50"]["token_weighted"]["mean"],
            0.0,
        )
        self.assertIn("correct_vs_constant_lambda_delta", metrics)

    def test_base_relative_preference_gain_is_zero_at_lambda_zero(self) -> None:
        pair = (frozen_sequence(2), frozen_sequence(3))
        routes = tuple(static_parm_route(item, 0.0) for item in pair)
        objective = preference_gain_objective(
            routes,
            pair,
            torch.tensor([1.0, 0.0]),
            pairwise_logit_scale=3.0,
        )
        torch.testing.assert_close(
            objective.weighted_guidance_gain,
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            objective.loss,
            torch.log(torch.tensor(2.0)),
            atol=1e-6,
            rtol=0.0,
        )

    def test_increasing_preferred_guidance_gain_decreases_pairwise_loss(self) -> None:
        pair = (frozen_sequence(2), frozen_sequence(3))
        base_routes = tuple(static_parm_route(item, 0.0) for item in pair)
        improved_first = base_routes[0].guided_logprobs.clone()
        gold = pair[0].gold_token_ids
        improved_first[0, torch.arange(gold.numel()), gold] += 0.2
        improved_routes = (
            replace(base_routes[0], guided_logprobs=improved_first),
            base_routes[1],
        )
        base = preference_gain_objective(
            base_routes,
            pair,
            torch.tensor([1.0, 0.0]),
            pairwise_logit_scale=1.0,
        )
        improved = preference_gain_objective(
            improved_routes,
            pair,
            torch.tensor([1.0, 0.0]),
            pairwise_logit_scale=1.0,
        )
        self.assertLess(float(improved.loss), float(base.loss))

    def test_exact_sequence_score_sensitivity_matches_autograd(self) -> None:
        pair = (frozen_sequence(2), frozen_sequence(3))
        scale = torch.tensor(0.02, requires_grad=True)
        scores = []
        for item in pair:
            logprobs = F.log_softmax(
                item.base_logprobs + scale * item.guide_logprobs,
                dim=-1,
            )
            scores.append(
                logprobs[0]
                .gather(-1, item.gold_token_ids.unsqueeze(-1))
                .squeeze(-1)
                .mean()
            )
        expected = torch.autograd.grad(scores[0] - scores[1], scale)[0]
        measured = exact_score_margin_sensitivity(pair, 0.02)
        self.assertAlmostEqual(
            measured["d_score_margin_d_lambda"],
            float(expected),
            places=5,
        )

    def test_linear_floor_and_endpoint_sensitivity_have_direct_gradients(self) -> None:
        pair = (frozen_sequence(2), frozen_sequence(3))
        low = []
        left = []
        right = []
        for item in pair:
            shape = item.base_logprobs.shape[:-1] + (1,)
            low_value = torch.full(shape, 0.001, requires_grad=True)
            left_value = torch.full(shape, 0.01, requires_grad=True)
            right_value = torch.full(shape, 0.011, requires_grad=True)
            low.append(replace(static_parm_route(item, 0.0), lambda_t=low_value))
            left.append(replace(static_parm_route(item, 0.0), lambda_t=left_value))
            right.append(replace(static_parm_route(item, 0.0), lambda_t=right_value))
        floor_loss, mean_lambda = linear_lambda_floor_loss(tuple(low), floor=0.01)
        floor_grad = torch.autograd.grad(floor_loss, low[0].lambda_t)[0]
        self.assertAlmostEqual(float(mean_lambda), 0.001, places=6)
        self.assertTrue(bool((floor_grad < 0).all()))
        sensitivity, delta = endpoint_sensitivity_loss(
            tuple(left), tuple(right), minimum_delta=0.005
        )
        sensitivity_grad = torch.autograd.grad(sensitivity, left[0].lambda_t)[0]
        self.assertAlmostEqual(float(delta), 0.001, places=6)
        self.assertTrue(bool((sensitivity_grad > 0).all()))

    def test_no_alpha_transplant_appends_fresh_preference_and_resets_head(self) -> None:
        source = smart_router(use_preference=False)
        alpha_config = replace(
            source.config,
            variant="v2_topk_state_history_alpha",
            use_preference=True,
            preference_dim=2,
        )
        torch.manual_seed(12)
        destination = transplant_no_alpha_router(
            source,
            alpha_config,
            target_gate=0.03,
            head_weight_std=0.01,
        )
        self.assertEqual(
            destination.feature_slices["preference"].start,
            source.feature_dim,
        )
        torch.testing.assert_close(
            destination.token_embedding.weight,
            source.token_embedding.weight,
        )
        head = destination.fusion_mlp[-1]
        self.assertAlmostEqual(float(torch.sigmoid(head.bias)), 0.03, places=6)
        self.assertGreater(float(head.weight.std()), 0.0)

    def test_gradient_table_separates_preference_head_and_router_rest(self) -> None:
        router = smart_router(use_preference=True)
        pair = (frozen_sequence(2), frozen_sequence(3))
        routes = tuple(
            router_parm_route(
                router,
                item,
                router_alpha=torch.tensor([0.8, 0.2]),
            )
            for item in pair
        )
        objective = preference_gain_objective(
            routes,
            pair,
            torch.tensor([0.8, 0.2]),
            pairwise_logit_scale=1.0,
        )
        groups = router_parameter_groups(router)
        self.assertEqual(
            set(groups),
            {"preference_encoder", "final_lambda_head", "router_rest"},
        )
        norms = term_gradient_norms(objective.loss, groups, retain_graph=False)
        self.assertGreater(norms["preference_encoder"], 0.0)
        self.assertGreater(norms["final_lambda_head"], 0.0)
        self.assertGreater(norms["router_rest"], 0.0)

    def test_pilot_v2_config_preserves_threshold_and_isolated_output(self) -> None:
        config = AlphaPreferencePilotV2Config.load_json(
            PROJECT_ROOT
            / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot_v2.json"
        )
        self.assertEqual(config.min_control_lambda_delta, 0.001)
        self.assertEqual(
            config.output_dir,
            "results/parm_taro/training/v2_alpha_preference_pilot_v2",
        )
        with self.assertRaisesRegex(ValueError, "preserve"):
            replace(config, min_control_lambda_delta=1e-6)
        schema = json.loads(
            (
                PROJECT_ROOT
                / "PARM_TARO/schemas/stage9_alpha_preference_pilot_v2.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["min_control_lambda_delta"]["const"],
            0.001,
        )

    def test_pilot_v2_requires_measured_pass_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            config = replace(
                AlphaPreferencePilotV2Config(),
                required_failure_diagnostic_path=str(path),
            )
            path.write_text(
                json.dumps(
                    {
                        "status": "HOST_EXECUTION_REQUIRED",
                        "protected_unchanged": True,
                        "pairwise_direction_checks": {"passed": True},
                        "recommendation": {
                            "initialization": "fresh_alpha_target_gate",
                            "full_retraining_authorized": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "requires a PASS"):
                _require_failure_diagnostic(config)

    def test_pre_sigmoid_sensitivity_target_and_loss_are_exact(self) -> None:
        target = lambda_delta_to_logit_delta(
            reference_lambda=0.03,
            lambda_delta=0.005,
        )
        self.assertGreater(target, 0.1)
        pair = (frozen_sequence(2), frozen_sequence(3))
        left = []
        right = []
        for item in pair:
            shape = item.base_logprobs.shape[:-1] + (1,)
            left_lambda = torch.full(shape, 0.03, requires_grad=True)
            right_lambda = torch.full(shape, 0.031, requires_grad=True)
            left.append(
                replace(static_parm_route(item, 0.0), lambda_t=left_lambda)
            )
            right.append(
                replace(static_parm_route(item, 0.0), lambda_t=right_lambda)
            )
        loss, logit_delta, lambda_delta = endpoint_logit_sensitivity_loss(
            tuple(left),
            tuple(right),
            minimum_logit_delta=target,
        )
        self.assertAlmostEqual(float(lambda_delta), 0.001, places=6)
        self.assertGreater(float(logit_delta), 0.0)
        gradient = torch.autograd.grad(loss, left[0].lambda_t)[0]
        self.assertTrue(bool((gradient > 0).all()))

    def test_zero_alpha_residual_exactly_reproduces_pilot_v2_router(self) -> None:
        torch.manual_seed(41)
        source = smart_router(use_preference=True)
        residual = build_from_v2_router(
            source,
            AlphaResidualConfig(
                preference_embedding_dim=source.config.preference_embedding_dim,
                state_bottleneck_dim=source.config.fusion_bottleneck_dim,
                residual_hidden_dim=5,
            ),
        )
        final = residual.alpha_state_residual[-1]
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
        item = frozen_sequence(7)
        alpha = torch.tensor([0.8, 0.2])
        expected = router_parm_route(source, item, router_alpha=alpha)
        actual = router_parm_route(residual, item, router_alpha=alpha)
        torch.testing.assert_close(actual.lambda_t, expected.lambda_t)
        torch.testing.assert_close(actual.guided_logprobs, expected.guided_logprobs)

    def test_alpha_residual_uses_both_preference_and_decoding_state(self) -> None:
        torch.manual_seed(42)
        source = smart_router(use_preference=True)
        residual = build_from_v2_router(
            source,
            AlphaResidualConfig(
                preference_embedding_dim=source.config.preference_embedding_dim,
                state_bottleneck_dim=source.config.fusion_bottleneck_dim,
                residual_hidden_dim=5,
            ),
        )
        item = frozen_sequence(8)
        endpoint_a = router_parm_route(
            residual, item, router_alpha=torch.tensor([1.0, 0.0])
        ).lambda_t
        endpoint_b = router_parm_route(
            residual, item, router_alpha=torch.tensor([0.0, 1.0])
        ).lambda_t
        self.assertFalse(torch.equal(endpoint_a, endpoint_b))
        changed = replace(
            item,
            base_selected_logprobs=item.base_selected_logprobs + 2.0,
        )
        changed_state = router_parm_route(
            residual, changed, router_alpha=torch.tensor([1.0, 0.0])
        ).lambda_t
        self.assertFalse(torch.equal(endpoint_a, changed_state))

    def test_alpha_residual_checkpoint_roundtrip_and_no_overwrite(self) -> None:
        source = smart_router(use_preference=True)
        residual = build_from_v2_router(
            source,
            AlphaResidualConfig(
                preference_embedding_dim=source.config.preference_embedding_dim,
                state_bottleneck_dim=source.config.fusion_bottleneck_dim,
                residual_hidden_dim=5,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot_v3.pt"
            save_alpha_residual_checkpoint(
                residual,
                path,
                training_state={"step": 3},
                metadata={"task": "unit_test"},
            )
            loaded, payload = load_alpha_residual_checkpoint(path)
            self.assertEqual(payload["training_state"]["step"], 3)
            for name, value in residual.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])
            with self.assertRaises(FileExistsError):
                save_alpha_residual_checkpoint(
                    residual,
                    path,
                    training_state={},
                    metadata={},
                )

    def test_alpha_warmup_freezes_state_router_only(self) -> None:
        source = smart_router(use_preference=True)
        residual = build_from_v2_router(
            source,
            AlphaResidualConfig(
                preference_embedding_dim=source.config.preference_embedding_dim,
                state_bottleneck_dim=source.config.fusion_bottleneck_dim,
                residual_hidden_dim=5,
            ),
        )
        trainable = _set_trainable_phase(residual, alpha_only=True)
        trainable_ids = {id(parameter) for parameter in trainable}
        self.assertEqual(
            trainable_ids,
            {id(parameter) for parameter in residual.alpha_residual_parameters()},
        )
        groups = alpha_path_parameter_groups(residual)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for group in (
                    "fusion_layers",
                    "final_lambda_head",
                    "state_router_rest",
                )
                for parameter in groups[group]
            )
        )

    def test_pilot_v3_config_and_diagnostic_gate_remain_strict(self) -> None:
        config = AlphaPreferencePilotV3Config.load_json(
            PROJECT_ROOT
            / "PARM_TARO/configs/train_stage9_v2_alpha_preference_pilot_v3.json"
        )
        self.assertEqual(config.min_control_lambda_delta, 0.001)
        self.assertTrue(config.freeze_state_router_during_warmup)
        self.assertEqual(
            config.output_dir,
            "results/parm_taro/training/v2_alpha_preference_pilot_v3",
        )
        with self.assertRaisesRegex(ValueError, "preserve"):
            replace(config, min_control_lambda_delta=0.0001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            gated = replace(config, required_alpha_path_diagnostic=str(path))
            path.write_text(
                json.dumps(
                    {
                        "status": "HOST_EXECUTION_REQUIRED",
                        "protected_unchanged": True,
                        "conclusion": {
                            "sensitivity_loss_implementation_correct": True
                        },
                        "recommendation": {
                            "architecture": (
                                "normalized_alpha_state_pre_sigmoid_residual"
                            ),
                            "freeze_state_router_during_alpha_warmup": True,
                            "min_control_lambda_delta": 0.001,
                            "full_retraining_authorized": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "requires a PASS"):
                require_alpha_path_diagnostic(gated)


class TeacherForcingTests(unittest.TestCase):
    def test_shared_runtime_instantiates_exactly_one_physical_backbone(self) -> None:
        SpyAutoModel.calls = []
        SpyPeftModel.calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base"
            paths = {
                "base_model": base_path,
                "parm_adapter": root / "adapter",
                "output_dir": root / "output",
            }
            config = ParmRouterTrainingConfig(
                device="cpu",
                model_dtype="float32",
                model_placement="single_device",
                share_base_backbone=True,
                load_in_4bit=True,
            )
            base_view, guide, info = _load_frozen_model_pair(
                config,
                paths,
                base_path,
                device=torch.device("cpu"),
                auto_model_class=SpyAutoModel,
                peft_model_class=SpyPeftModel,
                bitsandbytes_config_class=FakeBitsAndBytesConfig,
            )
        self.assertEqual(len(SpyAutoModel.calls), 1)
        self.assertEqual(len(SpyPeftModel.calls), 1)
        self.assertIs(base_view.peft_model, guide)
        self.assertIs(SpyPeftModel.calls[0][0], guide.backbone)
        self.assertEqual(info["physical_backbone_count"], 1)
        freeze_for_inference(base_view, guide)
        ids = torch.tensor([[4, 5]])
        with torch.inference_mode():
            base_first = base_view(input_ids=ids).logits
            parm = guide(input_ids=ids).logits
            base_second = base_view(input_ids=ids).logits
        torch.testing.assert_close(base_first, base_second)
        self.assertFalse(torch.equal(base_first, parm))
        self.assertTrue(guide.adapter_enabled)
        self.assertTrue(
            all(parameter.grad is None for parameter in guide.parameters())
        )

    def test_cuda_model_kwargs_require_nf4_single_backbone_quantization(self) -> None:
        config = ParmRouterTrainingConfig(
            device="cuda",
            model_dtype="float16",
            share_base_backbone=True,
            load_in_4bit=True,
        )
        kwargs = _model_kwargs(
            config,
            torch.device("cuda:0"),
            Path("/tmp/stage9-output"),
            "shared_backbone",
            bitsandbytes_config_class=FakeBitsAndBytesConfig,
        )
        self.assertEqual(kwargs["device_map"], {"": 0})
        quantization = kwargs["quantization_config"]
        self.assertTrue(quantization.values["load_in_4bit"])
        self.assertEqual(quantization.values["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(quantization.values["bnb_4bit_use_double_quant"])

    def test_shared_backbone_base_view_disables_adapter_and_restores_it(self) -> None:
        model = TinyAdapterModel()
        ids = torch.tensor([[4, 5]])
        freeze_for_inference(model)
        view = FrozenBaseModelView(model)
        state_before = capture_adapter_layer_state(model)
        guide = model(ids).logits.clone()
        base = view(ids).logits
        guide_after = model(ids).logits.clone()
        transition = view.last_adapter_transition
        self.assertIsNotNone(transition)
        assert transition is not None
        self.assertFalse(torch.equal(base, guide))
        torch.testing.assert_close(guide_after, guide, rtol=0.0, atol=0.0)
        self.assertEqual(transition.before, state_before)
        self.assertTrue(transition.inside)
        self.assertTrue(all(state.disabled for state in transition.inside))
        self.assertEqual(transition.after, transition.before)
        self.assertTrue(transition.requires_grad_restored)
        self.assertEqual(capture_adapter_layer_state(model), state_before)
        self.assertTrue(model.adapter_enabled)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )
        self.assertEqual(
            {id(parameter) for parameter in view.parameters()},
            {id(parameter) for parameter in model.parameters()},
        )

    def test_adapter_state_capture_ignores_callable_transformers_api(self) -> None:
        class MethodOnlyModule(nn.Module):
            def disable_adapters(self) -> None:
                pass

        model = nn.Sequential(MethodOnlyModule(), TinyAdapterModel())
        states = capture_adapter_layer_state(model)
        self.assertEqual([state.name for state in states], ["1"])

    def test_tokenization_matches_parm_bos_and_causal_shift(self) -> None:
        tokenizer = OffsetTokenizer()
        tokenized = tokenize_response(
            tokenizer,
            example(),
            0,
            max_length=128,
            max_continuation_tokens=4,
            include_eos_target=True,
        )
        self.assertEqual(int(tokenized.input_ids[0, 0]), tokenizer.bos_token_id)
        for position, gold in zip(
            tokenized.logit_positions.tolist(), tokenized.gold_token_ids.tolist()
        ):
            self.assertEqual(int(tokenized.input_ids[0, position + 1]), gold)
        self.assertEqual(int(tokenized.gold_token_ids[-1]), tokenizer.eos_token_id)
        self.assertTrue(tokenized.truncated)

    def test_clean_boundary_matches_previous_response_mask_exactly(self) -> None:
        tokenizer = OffsetTokenizer()
        item = example()
        tokenized = tokenize_response(
            tokenizer,
            item,
            0,
            max_length=256,
            max_continuation_tokens=5,
            include_eos_target=False,
        )
        prefix = PROMPT_TEMPLATE.format(prompt=item.prompt)
        encoded = tokenizer(prefix + item.responses[0])
        old_indices = [
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if start >= len(prefix) and end > start
        ][:5]
        old_sequence = [tokenizer.bos_token_id] + encoded["input_ids"][
            : old_indices[-1] + 1
        ]
        old_positions = [index for index in old_indices]
        old_gold = [encoded["input_ids"][index] for index in old_indices]
        self.assertEqual(tokenized.input_ids.tolist(), [old_sequence])
        self.assertEqual(tokenized.logit_positions.tolist(), old_positions)
        self.assertEqual(tokenized.gold_token_ids.tolist(), old_gold)
        self.assertEqual(tokenized.boundary_diagnostic.boundary_tokens_excluded, 0)

    def test_crossing_token_is_context_only_and_first_full_token_is_target(self) -> None:
        tokenizer = ScriptedBoundaryTokenizer(crossing=True)
        item = replace(example(), responses=("Response", "Alternative"))
        tokenized = tokenize_response(
            tokenizer,
            item,
            0,
            max_length=64,
            max_continuation_tokens=3,
            include_eos_target=False,
        )
        diagnostic = tokenized.boundary_diagnostic
        self.assertEqual(diagnostic.boundary_tokens_excluded, 1)
        self.assertEqual(diagnostic.crossing_tokens[0].token_id, 11)
        self.assertEqual(diagnostic.crossing_tokens[0].text, ":R")
        self.assertEqual(diagnostic.first_response_loss_token_index, 2)
        self.assertIn(11, tokenized.input_ids.tolist()[0])
        self.assertNotIn(11, tokenized.gold_token_ids.tolist())
        self.assertEqual(int(tokenized.gold_token_ids[0]), 20)
        self.assertEqual(
            int(tokenized.input_ids[0, tokenized.logit_positions[0] + 1]),
            int(tokenized.gold_token_ids[0]),
        )

    def test_zero_length_special_tokens_remain_context_but_not_loss(self) -> None:
        tokenizer = ScriptedBoundaryTokenizer(crossing=False, zero_length=True)
        item = replace(example(), responses=("Response", "Alternative"))
        tokenized = tokenize_response(
            tokenizer,
            item,
            0,
            max_length=64,
            max_continuation_tokens=2,
            include_eos_target=False,
        )
        self.assertEqual(
            tokenized.boundary_diagnostic.zero_length_token_indices,
            (1, 2),
        )
        self.assertEqual(
            tokenized.boundary_diagnostic.first_response_loss_token_index,
            3,
        )
        self.assertIn(12, tokenized.input_ids.tolist()[0])
        self.assertIn(13, tokenized.input_ids.tolist()[0])
        self.assertTrue({12, 13, 10}.isdisjoint(tokenized.gold_token_ids.tolist()))

    def test_boundary_truncation_and_diagnostics_are_deterministic(self) -> None:
        tokenizer = ScriptedBoundaryTokenizer(crossing=True)
        item = replace(example(), responses=("Response", "Alternative"))
        kwargs = {
            "max_length": 64,
            "max_continuation_tokens": 1,
            "include_eos_target": False,
        }
        first = tokenize_response(tokenizer, item, 0, **kwargs)
        second = tokenize_response(tokenizer, item, 0, **kwargs)
        self.assertTrue(first.truncated)
        self.assertEqual(first.gold_token_ids.numel(), 1)
        self.assertTrue(torch.equal(first.input_ids, second.input_ids))
        self.assertTrue(torch.equal(first.logit_positions, second.logit_positions))
        self.assertEqual(first.boundary_diagnostic, second.boundary_diagnostic)

    def test_response_loss_never_contains_prompt_or_crossing_tokens(self) -> None:
        tokenizer = ScriptedBoundaryTokenizer(crossing=True, zero_length=True)
        item = replace(example(), responses=("Response", "Alternative"))
        tokenized = tokenize_response(
            tokenizer,
            item,
            0,
            max_length=64,
            max_continuation_tokens=16,
            include_eos_target=True,
        )
        self.assertTrue(
            {10, 11, 12, 13}.isdisjoint(tokenized.gold_token_ids.tolist())
        )
        for position, gold in zip(
            tokenized.logit_positions.tolist(), tokenized.gold_token_ids.tolist()
        ):
            self.assertEqual(int(tokenized.input_ids[0, position + 1]), gold)

    def test_example_order_is_deterministic_and_epoch_specific(self) -> None:
        examples = [example(f"sample-{index}") for index in range(8)]
        first = deterministic_example_order(examples, epoch=0, seed=12)
        second = deterministic_example_order(examples, epoch=0, seed=12)
        third = deterministic_example_order(examples, epoch=1, seed=12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_test_split_cannot_be_loaded_for_training_or_tuning(self) -> None:
        with self.assertRaisesRegex(ValueError, "only train or validation"):
            load_examples(PROJECT_ROOT / "dataset/parm_taro", "test")

    def test_online_distributions_are_frozen_and_alpha_reaches_pblora(self) -> None:
        tokenizer = OffsetTokenizer()
        tokenized = tokenize_response(
            tokenizer,
            example(),
            0,
            max_length=128,
            max_continuation_tokens=3,
            include_eos_target=False,
        )
        base = TinyCausalModel(11)
        guide = TinyCausalModel(11, preference_aware=True)
        for model in (base, guide):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        output = compute_frozen_sequence_distributions(
            base,
            guide,
            tokenized,
            PARMTokenAlignment.validate(
                tokenizer, base_vocab_size=11, guide_vocab_size=11
            ),
            torch.tensor([0.8, 0.2]),
            device=torch.device("cpu"),
        )
        self.assertFalse(output.base_logprobs.requires_grad)
        self.assertFalse(output.guide_logprobs.requires_grad)
        torch.testing.assert_close(guide.pref_vec, torch.tensor([0.2, 0.8]))

    def test_guide_only_recomputation_matches_full_online_path(self) -> None:
        tokenizer = OffsetTokenizer()
        tokenized = tokenize_response(
            tokenizer,
            example(),
            0,
            max_length=128,
            max_continuation_tokens=3,
            include_eos_target=False,
        )
        base = TinyCausalModel(11)
        guide = TinyCausalModel(11, preference_aware=True)
        for model in (base, guide):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        alignment = PARMTokenAlignment.validate(
            tokenizer, base_vocab_size=11, guide_vocab_size=11
        )
        alpha = torch.tensor([0.25, 0.75])
        full = compute_frozen_sequence_distributions(
            base,
            guide,
            tokenized,
            alignment,
            alpha,
            device=torch.device("cpu"),
        )
        guide_only = compute_frozen_guide_logprobs(
            guide,
            tokenized,
            alignment,
            alpha,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(guide_only, full.guide_logprobs)
        self.assertFalse(guide_only.requires_grad)


class RoutingAndGradientTests(unittest.TestCase):
    def test_static_route_uses_full_vocabulary_parm_equation(self) -> None:
        frozen = frozen_sequence()
        route = static_parm_route(frozen, 0.7)
        expected = F.log_softmax(
            frozen.base_logprobs + 0.7 * frozen.guide_logprobs,
            dim=-1,
        )
        torch.testing.assert_close(route.guided_logprobs, expected)

    def test_only_router_parameters_receive_gradients(self) -> None:
        base_source = torch.randn(1, 3, 11, requires_grad=True)
        guide_source = torch.randn(1, 3, 11, requires_grad=True)
        gold = torch.tensor([1, 2, 3])
        frozen = FrozenSequenceDistributions(
            base_logprobs=base_source,
            guide_logprobs=guide_source,
            gold_token_ids=gold,
            base_selected_logprobs=base_source.detach()[0]
            .gather(-1, gold.unsqueeze(-1))
            .squeeze(-1)
            .unsqueeze(0),
            position=torch.arange(3).unsqueeze(0),
        )
        router = taro_router()
        route = router_parm_route(router, frozen, router_alpha=None)
        F.nll_loss(route.guided_logprobs[0], gold).backward()
        self.assertIsNone(base_source.grad)
        self.assertIsNone(guide_source.grad)
        self.assertTrue(
            any(
                parameter.grad is not None and bool((parameter.grad != 0).any())
                for parameter in router.parameters()
            )
        )

    def test_history_score_is_strictly_causal(self) -> None:
        torch.manual_seed(4)
        router = smart_router(use_preference=False)
        frozen = frozen_sequence()
        changed = replace(
            frozen,
            base_selected_logprobs=frozen.base_selected_logprobs.clone(),
        )
        changed.base_selected_logprobs[0, 1] += 8.0
        first = router_parm_route(router, frozen, router_alpha=None).lambda_t
        second = router_parm_route(router, changed, router_alpha=None).lambda_t
        torch.testing.assert_close(first[:, :2], second[:, :2])
        self.assertFalse(torch.allclose(first[:, 2], second[:, 2]))

    def test_one_alpha_router_checkpoint_behavior_covers_many_alpha(self) -> None:
        torch.manual_seed(8)
        router = smart_router(use_preference=True)
        frozen = frozen_sequence()
        first = router_parm_route(
            router, frozen, router_alpha=torch.tensor([1.0, 0.0])
        ).lambda_t
        second = router_parm_route(
            router, frozen, router_alpha=torch.tensor([0.0, 1.0])
        ).lambda_t
        self.assertFalse(torch.equal(first, second))

    def test_optimizer_is_router_only(self) -> None:
        router = smart_router(use_preference=True)
        optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        assert_optimizer_contains_only(optimizer, router.parameters())
        _assert_optimizer_fp32(optimizer)
        external = nn.Parameter(torch.tensor(0.0))
        bad = torch.optim.AdamW([external], lr=1e-3)
        with self.assertRaisesRegex(ValueError, "non-router"):
            assert_optimizer_contains_only(bad, router.parameters())
        fp16 = nn.Parameter(torch.tensor(0.0, dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "must be FP32"):
            _assert_optimizer_fp32(torch.optim.AdamW([fp16], lr=1e-3))


class ComparisonAndContractTests(unittest.TestCase):
    def test_cpu_training_engine_updates_router_and_writes_stage_checkpoint(self) -> None:
        result_root = PROJECT_ROOT / "results/parm_taro"
        result_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=result_root) as directory:
            root = Path(directory)
            router_config_path = root / "router.json"
            TARORouterConfig(
                vocab_size=11,
                top_k=3,
                token_embedding_dim=4,
                hidden_dim=128,
                device="cpu",
            ).save_json(router_config_path)
            output_dir = root / "run"
            config = ParmRouterTrainingConfig(
                stage="taro",
                router_kind="taro",
                router_config_path=str(router_config_path),
                output_dir=str(output_dir),
                device="cpu",
                model_dtype="float32",
                model_placement="single_device",
                max_length=16,
                max_continuation_tokens=4,
                gradient_accumulation_steps=1,
                validation_alpha_grid=(0.2, 0.8),
                log_every_optimizer_steps=100,
                checkpoint_every_optimizer_steps=100,
                max_train_samples=2,
                max_validation_samples=2,
            )
            base = TinyCausalModel(11)
            guide = TinyCausalModel(11, preference_aware=True)
            for model in (base, guide):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            runtime = ParmTrainingRuntime(
                base_model=base,
                guide_model=guide,
                tokenizer=OffsetTokenizer(),
                alignment=PARMTokenAlignment.validate(
                    OffsetTokenizer(), base_vocab_size=11, guide_vocab_size=11
                ),
                device=torch.device("cpu"),
                tokenizer_descriptor={"semantic_sha256": "tiny-tokenizer"},
                adapter_config={"peft_type": "PBLORA", "obj_num": 2},
                paths={
                    "data_root": PROJECT_ROOT / "dataset/parm_taro",
                    "base_model": PROJECT_ROOT / "models/tulu-2-7b",
                    "guide_base_model": PROJECT_ROOT / "models/tulu-2-7b",
                    "parm_adapter": root / "adapter",
                    "router_config": router_config_path,
                    "output_dir": output_dir,
                },
            )
            examples = [example("first"), example("second", better=1, safer=0)]
            pair = (frozen_sequence(2), frozen_sequence(3))
            with (
                patch(
                    "PARM_TARO.training.engine.audit_training_prerequisites",
                    return_value={"ready": True, "checks": {}},
                ),
                patch(
                    "PARM_TARO.training.engine.load_training_runtime",
                    return_value=runtime,
                ),
                patch(
                    "PARM_TARO.training.engine.load_examples",
                    return_value=examples,
                ),
                patch(
                    "PARM_TARO.training.engine.frozen_response_pair",
                    return_value=pair,
                ),
                patch(
                    "PARM_TARO.training.engine._frozen_snapshot",
                    return_value={"immutable": "same"},
                ),
            ):
                report = train_stage(config)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "latest.pt").is_file())
            self.assertTrue(report["checks"]["optimizer_router_only"])
            self.assertTrue(report["checks"]["frozen_artifacts_unchanged"])

    def test_comparison_contains_static_taro_no_alpha_alpha_and_controls(self) -> None:
        config = ParmRouterTrainingConfig(
            stage="v2_alpha",
            router_kind="smart",
            max_length=16,
            max_continuation_tokens=4,
            validation_alpha_grid=(0.2, 0.8),
            min_control_lambda_delta=1e-12,
            require_shuffled_loss_degradation=False,
        )
        runtime = SimpleNamespace(device=torch.device("cpu"))
        examples = [example("first"), example("second", better=1, safer=0)]
        routers = {
            "taro": taro_router(),
            "v2_no_alpha": smart_router(use_preference=False),
            "v2_alpha": smart_router(use_preference=True),
        }
        pair = (frozen_sequence(2), frozen_sequence(3))
        with patch(
            "PARM_TARO.training.engine.frozen_response_pair",
            return_value=pair,
        ):
            report = evaluate_comparisons(config, runtime, examples, routers)
        self.assertEqual(
            set(report["methods"]),
            {
                "static_parm",
                "taro",
                "v2_no_alpha",
                "v2_alpha",
                "v2_alpha_shuffled",
                "v2_alpha_constant",
            },
        )
        self.assertTrue(report["alpha_controls"]["checks"]["shuffled_changes_lambda"])
        self.assertTrue(
            report["alpha_controls"]["checks"][
                "guide_alpha_held_correct_for_all_router_controls"
            ]
        )

    def test_shipped_configs_follow_required_training_order(self) -> None:
        paths = [
            PROJECT_ROOT / f"PARM_TARO/configs/train_stage9_{name}.json"
            for name in ("taro", "v2_no_alpha", "v2_alpha")
        ]
        configs = [ParmRouterTrainingConfig.load_json(path) for path in paths]
        self.assertEqual([config.stage for config in configs], list(configs[0].STAGES))
        self.assertEqual(configs[2].alpha_order, ("helpfulness", "harmlessness"))
        self.assertEqual(
            configs[2].scalarization_rule,
            "alpha_weighted_response_mean_nll",
        )

    def test_training_config_schema_is_strict_and_versioned(self) -> None:
        schema = json.loads(
            (
                PROJECT_ROOT
                / "PARM_TARO/schemas/stage9_training_config.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_alpha_preference_production_config_is_full_and_isolated(self) -> None:
        config = AlphaPreferenceProductionConfig.load_json(
            PROJECT_ROOT
            / "PARM_TARO/configs/train_stage9_v2_alpha_preference.json"
        )
        self.assertEqual(config.expected_train_samples, 8000)
        self.assertEqual(config.expected_validation_samples, 500)
        self.assertEqual(config.min_control_lambda_delta, 0.001)
        self.assertEqual(
            config.output_dir,
            "results/parm_taro/training/v2_alpha_preference",
        )
        self.assertNotIn("test", config.to_dict())
        with self.assertRaisesRegex(ValueError, "8,000"):
            replace(config, expected_train_samples=7999)
        with self.assertRaisesRegex(ValueError, "min_control_lambda_delta"):
            replace(config, min_control_lambda_delta=0.0001)

    def test_alpha_preference_production_schema_is_strict(self) -> None:
        schema = json.loads(
            (
                PROJECT_ROOT
                / "PARM_TARO/schemas/"
                "stage9_alpha_preference_production.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["expected_train_samples"]["const"], 8000)
        self.assertEqual(
            schema["properties"]["min_control_lambda_delta"]["const"],
            0.001,
        )

    def test_production_is_authorized_only_by_preserved_pilot_v3_pass(self) -> None:
        config = AlphaPreferenceProductionConfig.load_json(
            PROJECT_ROOT
            / "PARM_TARO/configs/train_stage9_v2_alpha_preference.json"
        )
        report = require_pilot_v3_pass(config)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(
            all(
                report["checks"][name]
                for name in REQUIRED_PRODUCTION_CHECKS
                if name in report["checks"]
            )
        )

    def test_production_checks_enforce_controls_and_frozen_artifacts(self) -> None:
        config = AlphaPreferenceProductionConfig()

        def summary(mean: float, std: float = 0.1) -> dict[str, float | int]:
            return {
                "count": 4,
                "mean": mean,
                "std": std,
                "min": mean - std,
                "max": mean + std,
            }

        before_metrics = {
            "correct_preference_loss": summary(0.7),
        }
        after_metrics = {
            "correct_lambda": summary(0.03, 0.002),
            "endpoint_alpha_lambda_delta": summary(0.005),
            "correct_vs_constant_lambda_delta": summary(0.0015),
            "correct_vs_shuffled_lambda_delta": summary(0.002),
            "correct_preference_loss": summary(0.64),
            "shuffled_preference_loss": summary(0.65),
            "correct_nll": summary(1.82),
            "no_alpha_nll": summary(1.81),
            "constant_control": {
                "gate": {"pass": True},
            },
        }
        phase = {
            "optimizer_router_only": True,
            "optimizer_parameters_fp32": True,
            "metrics": {"total_loss": summary(1.0)},
        }
        snapshot = {"hash": "same"}
        checks = production_checks(
            config,
            snapshot,
            snapshot,
            before_metrics,
            after_metrics,
            [phase, phase],
        )
        self.assertEqual(set(checks), set(REQUIRED_PRODUCTION_CHECKS))
        self.assertTrue(all(checks.values()))
        after_metrics["constant_control"]["gate"]["pass"] = False
        checks = production_checks(
            config,
            snapshot,
            snapshot,
            before_metrics,
            after_metrics,
            [phase, phase],
        )
        self.assertFalse(checks["constant_alpha_differs_materially"])
        self.assertTrue(checks["shuffled_alpha_changes_lambda_materially"])
        self.assertTrue(checks["correct_alpha_changes_lambda_materially"])

    def test_constant_control_gate_excludes_identical_alpha_tasks(self) -> None:
        constant = torch.tensor([0.5, 0.5])
        accumulator = ConstantControlAccumulator()
        values = {
            0.0: [0.0012, 0.0014],
            0.25: [0.0010, 0.0012],
            0.5: [0.0, 0.0],
            0.75: [0.0011, 0.0013],
            1.0: [0.0014, 0.0016],
        }
        for helpfulness, deltas in values.items():
            accumulator.update(
                torch.tensor([helpfulness, 1.0 - helpfulness]),
                constant,
                torch.tensor(deltas),
            )
        report = accumulator.finalize(
            threshold=0.001,
            bootstrap_samples=100,
            bootstrap_seed=17,
        )
        self.assertEqual(report["constant_alpha"], [0.5, 0.5])
        self.assertEqual(report["identical_tasks"], 1)
        self.assertEqual(report["non_identical_tasks"], 4)
        self.assertEqual(
            report["per_alpha_helpfulness"]["0.50"][
                "token_weighted"
            ]["mean"],
            0.0,
        )
        self.assertLess(
            report["all_tasks"]["token_weighted"]["mean"],
            report["non_identical_tasks_only"]["token_weighted"]["mean"],
        )
        self.assertTrue(report["gate"]["pass"])
        self.assertEqual(
            report["non_identical_tasks_only"]["bootstrap_mean_ci"][
                "unit"
            ],
            "validation_task_mean",
        )
        self.assertTrue(alpha_vectors_identical(constant, constant.clone()))
        self.assertFalse(
            alpha_vectors_identical(torch.tensor([0.25, 0.75]), constant)
        )

    def test_constant_audit_can_correct_only_the_single_control_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"immutable-checkpoint")
            production_path = root / "run_status.json"
            production = {
                "status": "NOT_PASS",
                "checks": {
                    name: name != "constant_alpha_differs_materially"
                    for name in REQUIRED_PRODUCTION_CHECKS
                },
                "checkpoint": {"sha256": sha256_file(checkpoint)},
            }
            production_path.write_text(json.dumps(production), encoding="utf-8")
            audit = {
                "status": "PASS",
                "corrected_production_status": "PASS",
                "checks": {"audit": True},
                "constant_control": {
                    "gate": {
                        "value": 0.0011,
                        "threshold": 0.001,
                        "threshold_unchanged": True,
                        "pass": True,
                    }
                },
                "source": {
                    "run_status_sha256": sha256_file(production_path),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "production_tree": hash_tree(root),
                },
            }
            self.assertTrue(source_failure_is_constant_only(production))
            self.assertTrue(
                constant_audit_authorizes_correction(
                    production,
                    audit,
                    production_path=production_path,
                    checkpoint_valid=True,
                )
            )
            audit["constant_control"]["gate"]["value"] = 0.0009
            self.assertFalse(
                constant_audit_authorizes_correction(
                    production,
                    audit,
                    production_path=production_path,
                    checkpoint_valid=True,
                )
            )

    def test_full_production_checkpoints_remain_byte_identical(self) -> None:
        root = PROJECT_ROOT / "results/parm_taro/training/v2_alpha_preference"
        self.assertEqual(
            sha256_file(root / "final.pt"),
            "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20",
        )
        self.assertEqual(
            sha256_file(root / "latest.pt"),
            "68d2a54245fde3f34981e2be7028d0ddcc487905b7290fc735c6fb2f02e34074",
        )

    def test_production_phase_checkpoint_resumes_from_saved_cursor(self) -> None:
        source = smart_router(use_preference=True)
        router = build_from_v2_router(
            source,
            AlphaResidualConfig(
                preference_embedding_dim=3,
                state_bottleneck_dim=4,
                residual_hidden_dim=5,
            ),
        )
        frozen_model = TinyCausalModel(11)
        frozen_model.requires_grad_(False)
        runtime = SimpleNamespace(
            device=torch.device("cpu"),
            base_model=frozen_model,
            guide_model=frozen_model,
        )
        config = replace(
            AlphaPreferenceProductionConfig(),
            gradient_accumulation_steps=1,
            checkpoint_every_optimizer_steps=1,
            log_every_optimizer_steps=100,
        )
        training_config = ParmRouterTrainingConfig(
            max_length=16,
            max_continuation_tokens=4,
        )
        calibration = {
            "pairwise_logit_scale": 2.0,
            "lambda_floor": 0.02,
            "minimum_logit_delta": 0.01,
            "weights": {
                "preference": 1.0,
                "nll_quality_phase": 0.1,
                "strength": 1.0,
                "logit_sensitivity": 1.0,
            },
        }
        examples = [example("resume-a"), example("resume-b", better=1, safer=0)]
        pair = (frozen_sequence(2), frozen_sequence(3))
        persistent = {
            "calibration": calibration,
            "validation_before": {"baseline": True},
            "protected_before": {"hash": "fixed"},
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "PARM_TARO.training.alpha_production.frozen_response_pair",
            return_value=pair,
        ):
            latest = Path(directory) / "latest.pt"
            report, steps = _run_phase(
                router=router,
                runtime=runtime,
                training_config=training_config,
                config=config,
                calibration=calibration,
                examples=examples,
                phase_index=0,
                global_optimizer_step=0,
                resume_state=None,
                latest_path=latest,
                metadata={"kind": "test"},
                completed_phases=[],
                persistent_state=persistent,
            )
            resumed_router, payload = load_alpha_residual_checkpoint(latest)
            resumed_report, resumed_steps = _run_phase(
                router=resumed_router,
                runtime=runtime,
                training_config=training_config,
                config=config,
                calibration=calibration,
                examples=examples,
                phase_index=0,
                global_optimizer_step=int(
                    payload["training_state"]["global_optimizer_step"]
                ),
                resume_state=payload["training_state"],
                latest_path=latest,
                metadata=payload["metadata"],
                completed_phases=[],
                persistent_state=persistent,
            )
        self.assertEqual(report["samples_used"], 2)
        self.assertEqual(resumed_report["samples_used"], 2)
        self.assertEqual(steps, resumed_steps)

    def test_final_gate_rejects_status_with_any_failed_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_status.json"
            path.write_text(
                json.dumps({"status": "PASS", "checks": {"one": True, "two": False}}),
                encoding="utf-8",
            )
            passed, _ = _passed_stage(path)
            self.assertFalse(passed)

    def test_stage_order_requires_passed_predecessor(self) -> None:
        config = ParmRouterTrainingConfig(
            stage="v2_no_alpha",
            router_kind="smart",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2_no_alpha"
            with self.assertRaisesRegex(ValueError, "Missing predecessor"):
                _validate_stage_order(config, output)
            predecessor = output.parent / "taro"
            predecessor.mkdir()
            (predecessor / "run_status.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            _validate_stage_order(config, output)

    def test_output_boundary_rejects_writes_outside_results_parm_taro(self) -> None:
        config = ParmRouterTrainingConfig(output_dir="results/router_v2/not-stage9")
        with self.assertRaisesRegex(ValueError, "results/parm_taro"):
            resolved_paths(config)

    def test_prerequisite_audit_does_not_substitute_plain_lora(self) -> None:
        config = ParmRouterTrainingConfig.load_json(
            PROJECT_ROOT / "PARM_TARO/configs/train_stage9_taro.json"
        )
        report = audit_training_prerequisites(
            replace(
                config,
                parm_adapter_path="models/genarm-tulu2-hh",
            )
        )
        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["frozen_two_objective_pblora_present"])
        self.assertEqual(report["status"], "PREREQUISITES_REQUIRED")

    def test_parm_protected_tree_hash_is_unchanged(self) -> None:
        baseline = json.loads(
            (
                PROJECT_ROOT / "PARM_TARO/reports/parm_baseline_hash.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            hash_tree(PROJECT_ROOT / "PARM")["tree_sha256"],
            baseline["tree_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
