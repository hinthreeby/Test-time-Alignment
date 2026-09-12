from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from router_v2.config import TARORouterConfig
from router_v2.evaluation.rad.config import RADEvaluationConfig
from router_v2.evaluation.rad.data import (
    PromptRecord,
    load_prompt_split,
    read_prompt_records,
)
from router_v2.evaluation.rad.decoding import (
    RADV2Decoder,
    RouteSpec,
    V2Generation,
    fixed_spec,
    heuristic_lambda,
    test_route_specs,
)
from router_v2.evaluation.rad.engine import _run_generation_jobs
from router_v2.evaluation.rad.legacy import V1_METHOD_MAP, import_v1_records
from router_v2.evaluation.rad.metrics import (
    aggregate_metrics,
    attach_ppl_degradation,
    generation_length_diagnostics,
    lambda_diagnostics,
    mean_routing_strength,
    metric_record,
    pareto_points,
    select_by_alignment,
)
from router_v2.model import TAROTokenRouter
from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class TinyTokenizer:
    eos_token_id = 4

    def __call__(self, text: str, **_: object):
        del text
        return {
            "input_ids": torch.tensor([[0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, token_ids, **_: object) -> str:
        return " ".join(f"token{value}" for value in token_ids)


class TinyCausalModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("values", torch.tensor(logits, dtype=torch.float32))
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **_: object):
        self.calls += 1
        logits = self.values.view(1, 1, -1).expand(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        return SimpleNamespace(logits=logits + self.anchor, past_key_values=None)


def smart_config(*, history: bool) -> SmartRouterConfig:
    return SmartRouterConfig(
        variant="custom",
        vocab_size=5,
        top_k=2,
        token_embedding_dim=3,
        candidate_hidden_dim=4,
        candidate_feature_mode="token_aware_mean_max",
        use_confidence=True,
        use_position=True,
        use_history=history,
        use_preference=False,
        max_position=7,
        history_hidden_dim=4,
        fusion_hidden_dim=8,
        fusion_bottleneck_dim=4,
        lambda_max=1.0,
        device="cpu",
    )


def tiny_decoder() -> tuple[RADV2Decoder, TinyCausalModel, TinyCausalModel]:
    base = TinyCausalModel([0.0, 4.0, 1.0, -1.0, -2.0])
    guide = TinyCausalModel([0.0, 1.0, 5.0, -1.0, -2.0])
    taro = TAROTokenRouter(
        TARORouterConfig(
            vocab_size=5,
            top_k=2,
            token_embedding_dim=3,
            device="cpu",
        )
    )
    decoder = RADV2Decoder(
        base_model=base,
        guide_model=guide,
        tokenizer=TinyTokenizer(),
        routers={
            "taro": taro,
            "v2_state": SmartTokenRouter(smart_config(history=False)),
            "v2_history": SmartTokenRouter(smart_config(history=True)),
        },
        device=torch.device("cpu"),
        top_k=2,
        max_new_tokens=3,
        temperature=1.0,
        do_sample=False,
        stop_on_eos=False,
    )
    return decoder, base, guide


def raw_metric(
    method: str,
    family: str,
    perplexity: float,
    alignment: int,
    probability: float,
) -> dict[str, object]:
    return metric_record(
        {
            "prompt_id": "p",
            "prompt": "prompt",
            "prompt_sentiment_class": "negative",
            "method": method,
            "method_family": family,
            "ppl_model_family": family,
            "ppl_protocol": family,
            "seed": 1,
            "generated_text": "good output words",
            "selected_token_ids": [1, 2, 3],
            "lambda_history": [0.2, 0.4, 0.8],
            "perplexity": perplexity,
            "classifier_sentiment_success": alignment,
            "classifier_target_probability": probability,
            "classifier_predicted_label": "POSITIVE",
            "classifier_predicted_label_id": 1,
            "latency": {
                "average_latency_per_token": 0.1,
                "total_generation_time": 0.3,
            },
            "failure_status": "success",
            "source": "unit",
        }
    )


class EvaluationConfigAndDataTests(unittest.TestCase):
    def test_shipped_configs_and_schema_are_valid(self) -> None:
        for name in (
            "evaluate_stage6_rad_smoke.json",
            "evaluate_stage6_rad_full.json",
        ):
            config = RADEvaluationConfig.load_json(
                WORKSPACE_ROOT / "router_v2" / "configs" / name
            )
            self.assertEqual(config.top_k, 20)
            self.assertGreaterEqual(config.max_new_tokens, 32)
            self.assertEqual(config.test_prompt_offset_per_class, 50)
        with (
            WORKSPACE_ROOT
            / "router_v2"
            / "schemas"
            / "rad_evaluation_config.schema.json"
        ).open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_config_enforces_output_boundary_and_32_tokens(self) -> None:
        with self.assertRaises(ValueError):
            RADEvaluationConfig(output_dir="router/evaluation/new")
        with self.assertRaises(ValueError):
            RADEvaluationConfig(max_new_tokens=31)

    def test_prompt_loader_uses_explicit_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "negative_prompts.jsonl"
            rows = [
                {"md5_hash": f"id-{index}", "prompt": {"text": f"p{index}"}}
                for index in range(5)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected = read_prompt_records(
                path,
                "negative",
                split="test",
                limit=2,
                offset=2,
            )
        self.assertEqual(
            [row.prompt_id for row in selected],
            ["rad:test:negative:00000002", "rad:test:negative:00000003"],
        )
        self.assertEqual(
            [row.source_prompt_id for row in selected],
            ["id-2", "id-3"],
        )
        self.assertEqual([row.source_index for row in selected], [2, 3])

    def test_full_test_split_has_collision_free_ids_without_dropping_rows(self) -> None:
        config = RADEvaluationConfig.load_json(
            WORKSPACE_ROOT
            / "router_v2"
            / "configs"
            / "evaluate_stage6_rad_full.json"
        )
        prompts = load_prompt_split(config, "test", project_root=WORKSPACE_ROOT)
        prompt_ids = [prompt.prompt_id for prompt in prompts]
        source_ids = [prompt.source_prompt_id for prompt in prompts]
        self.assertEqual(len(prompts), 600)
        self.assertEqual(len(prompt_ids), len(set(prompt_ids)))
        # The source reuses one md5 across neutral/positive; both rows remain.
        self.assertEqual(len(source_ids) - len(set(source_ids)), 1)

    def test_validation_split_has_unique_prompt_ids(self) -> None:
        config = RADEvaluationConfig.load_json(
            WORKSPACE_ROOT
            / "router_v2"
            / "configs"
            / "evaluate_stage6_rad_full.json"
        )
        prompts = load_prompt_split(
            config,
            "validation",
            project_root=WORKSPACE_ROOT,
        )
        self.assertEqual(len(prompts), 150)
        self.assertEqual(
            len({prompt.prompt_id for prompt in prompts}),
            len(prompts),
        )

    def test_prompt_ids_are_reproducible_for_same_source_and_config(self) -> None:
        config = RADEvaluationConfig.load_json(
            WORKSPACE_ROOT
            / "router_v2"
            / "configs"
            / "evaluate_stage6_rad_full.json"
        )
        first = load_prompt_split(config, "test", project_root=WORKSPACE_ROOT)
        second = load_prompt_split(config, "test", project_root=WORKSPACE_ROOT)
        self.assertEqual(first, second)

    def test_full_resume_keys_are_unique(self) -> None:
        config = RADEvaluationConfig.load_json(
            WORKSPACE_ROOT
            / "router_v2"
            / "configs"
            / "evaluate_stage6_rad_full.json"
        )
        prompts = load_prompt_split(config, "test", project_root=WORKSPACE_ROOT)
        methods = (
            "v2_base",
            "v2_fixed",
            "v2_heuristic",
            "taro",
            "v2_state",
            "v2_history",
            "taro_same_average",
            "v2_same_average",
        )
        keys = [
            (prompt.prompt_id, method, seed)
            for prompt in prompts
            for method in methods
            for seed in config.seeds
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_v1_offset_and_source_ids_remain_matchable(self) -> None:
        config = RADEvaluationConfig.load_json(
            WORKSPACE_ROOT
            / "router_v2"
            / "configs"
            / "evaluate_stage6_rad_full.json"
        )
        with (
            WORKSPACE_ROOT
            / "router"
            / "evaluation"
            / "report_run_32tokens"
            / "config.json"
        ).open("r", encoding="utf-8") as handle:
            v1_config = json.load(handle)
        prompts = load_prompt_split(config, "test", project_root=WORKSPACE_ROOT)
        self.assertEqual(
            config.test_prompt_offset_per_class,
            v1_config["validation_prompt_limit_per_class"],
        )
        first_by_class = {
            prompt.prompt_sentiment_class: prompt
            for prompt in reversed(prompts)
        }
        self.assertEqual(set(first_by_class), set(config.prompt_classes))
        self.assertTrue(
            all(
                prompt.source_index == config.test_prompt_offset_per_class
                for prompt in first_by_class.values()
            )
        )
        self.assertTrue(
            all(prompt.source_prompt_id for prompt in first_by_class.values())
        )


class DecodingTests(unittest.TestCase):
    def test_base_skips_guide_and_fixed_endpoints_route_full_vocab(self) -> None:
        decoder, base, guide = tiny_decoder()
        base_output = decoder.generate("p", RouteSpec("base", "base"), seed=1)
        self.assertEqual(base_output.selected_token_ids, [1, 1, 1])
        self.assertEqual(base_output.base_entropy_history, [])
        json.dumps(base_output.to_dict(), allow_nan=False)
        self.assertEqual(guide.calls, 0)
        guide_output = decoder.generate("p", fixed_spec(1.0), seed=1)
        self.assertEqual(guide_output.selected_token_ids, [2, 2, 2])
        self.assertEqual(guide.calls, 3)
        self.assertTrue(all(value == 1.0 for value in guide_output.lambda_history))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in guide.parameters()))

    def test_taro_state_and_history_decode_without_current_target(self) -> None:
        decoder, _, _ = tiny_decoder()
        for kind in ("taro", "v2_state", "v2_history"):
            with self.subTest(kind=kind):
                output = decoder.generate("p", RouteSpec(kind, kind), seed=2)
                self.assertEqual(len(output.selected_token_ids), 3)
                self.assertEqual(len(output.lambda_history), 3)
                self.assertTrue(all(0.0 < value < 1.0 for value in output.lambda_history))

    def test_heuristics_are_bounded_and_target_independent(self) -> None:
        base = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        guide = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        from router_v2.candidates import select_taro_topk

        topk = select_taro_topk(base, guide, top_k=2)
        for kind in ("heuristic_entropy_gap", "heuristic_js"):
            value, diagnostics = heuristic_lambda(
                kind, topk, position=0, max_position=3
            )
            self.assertEqual(value.shape, (1, 1))
            self.assertTrue(0.0 <= float(value) <= 1.0)
            self.assertIn("js", diagnostics)


class MetricsAndSelectionTests(unittest.TestCase):
    @staticmethod
    def generation_row(
        method: str,
        length: int,
        *,
        terminated_on_eos: bool,
        prompt_class: str = "negative",
    ) -> dict[str, object]:
        return {
            "prompt_id": f"{method}-{length}-{prompt_class}",
            "method": method,
            "method_family": "v1" if method.startswith("v1") else "v2",
            "prompt_sentiment_class": prompt_class,
            "generation_length": length,
            "selected_token_ids": list(range(length)),
            "terminated_on_eos": terminated_on_eos,
            "source": "unit",
        }

    def test_length_gate_accepts_short_eos_under_32_token_budget(self) -> None:
        records = [
            self.generation_row("v1_base", 6, terminated_on_eos=True),
            self.generation_row("v2_base", 32, terminated_on_eos=False),
        ]
        audit = generation_length_diagnostics(records, max_new_tokens=32)
        self.assertTrue(audit["pass"])
        self.assertFalse(
            audit["all_realized_lengths_at_least_required_budget"]
        )
        self.assertEqual(audit["short_record_count"], 1)
        stats = {row["method"]: row for row in audit["method_statistics"]}
        self.assertEqual(stats["v1_base"]["min"], 6)
        self.assertEqual(stats["v2_base"]["mean"], 32.0)

    def test_length_gate_rejects_short_non_eos_and_count_mismatch(self) -> None:
        short = self.generation_row(
            "v1_base",
            6,
            terminated_on_eos=False,
        )
        mismatch = self.generation_row(
            "v2_base",
            32,
            terminated_on_eos=False,
        )
        mismatch["selected_token_ids"] = [1] * 31
        audit = generation_length_diagnostics(
            [short, mismatch],
            max_new_tokens=32,
        )
        self.assertFalse(audit["pass"])
        self.assertEqual(
            len(audit["invalid_records"]["short_without_eos"]),
            1,
        )
        self.assertEqual(
            len(audit["invalid_records"]["length_mismatch"]),
            1,
        )

    def test_family_relative_ppl_and_pareto(self) -> None:
        records = [
            raw_metric("v1_base", "v1", 10.0, 0, 0.2),
            raw_metric("v1_router", "v1", 12.0, 1, 0.8),
            raw_metric("v2_base", "v2", 20.0, 0, 0.3),
            raw_metric("v2_history", "v2", 18.0, 1, 0.9),
        ]
        attached = attach_ppl_degradation(
            records,
            family_base_methods={"v1": "v1_base", "v2": "v2_base"},
        )
        values = {row["method"]: row["ppl_degradation"] for row in attached}
        self.assertAlmostEqual(values["v1_base"], 0.0)
        self.assertAlmostEqual(values["v1_router"], 0.2)
        self.assertAlmostEqual(values["v2_history"], -0.1)
        aggregate = aggregate_metrics(
            attached, bootstrap_seed=1, bootstrap_samples=20
        )
        points = pareto_points(aggregate)
        self.assertEqual(len(points), 4)
        self.assertIn(
            "pareto_frontier_alignment_vs_ppl_degradation", points[0]
        )

    def test_selection_same_average_and_lambda_diagnostics(self) -> None:
        low = raw_metric("fixed_low", "v2", 10.0, 0, 0.2)
        high = raw_metric("fixed_high", "v2", 11.0, 1, 0.8)
        self.assertEqual(
            select_by_alignment([low, high], ["fixed_low", "fixed_high"]),
            "fixed_high",
        )
        self.assertAlmostEqual(mean_routing_strength([high], "fixed_high"), 0.4666666667)
        summary = lambda_diagnostics([low, high])
        all_rows = [row for row in summary if row["position"] == "all"]
        self.assertEqual(len(all_rows), 2)

    def test_same_average_specs_are_distinct_controls(self) -> None:
        specs = test_route_specs(
            best_fixed_lambda=0.5,
            best_heuristic="heuristic_js",
            taro_same_average=0.8,
            v2_same_average=0.9,
        )
        by_label = {spec.label: spec for spec in specs}
        self.assertEqual(by_label["taro_same_average"].value, 0.8)
        self.assertEqual(by_label["v2_same_average"].value, 0.9)


class LegacyAndResumeTests(unittest.TestCase):
    def test_legacy_import_filters_exact_keys_without_writing_source(self) -> None:
        prompt = PromptRecord("p", "prompt", "negative")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "v1.jsonl"
            rows = []
            for old_method in V1_METHOD_MAP:
                rows.append(
                    {
                        "prompt_id": "p",
                        "prompt": "prompt",
                        "prompt_sentiment_class": "negative",
                        "method": old_method,
                        "seed": 1,
                        "generated_text": "good text",
                        "selected_token_ids": [1] * 32,
                        "beta_history": [15.0] * 32,
                        "beta_max": 30.0,
                        "base_lm_perplexity": 7.0,
                        "classifier_sentiment_success": 1,
                        "classifier_target_probability": 0.9,
                        "classifier_predicted_label": "POSITIVE",
                        "classifier_predicted_label_id": 1,
                        "latency": {
                            "total_generation_time": 1.0,
                            "average_latency_per_token": 1 / 32,
                        },
                        "failure_status": "success",
                    }
                )
            content = "".join(json.dumps(row) + "\n" for row in rows)
            path.write_text(content, encoding="utf-8")
            imported = import_v1_records(
                path,
                prompts=[prompt],
                seeds=[1],
                eos_token_id=99,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), content)
        self.assertEqual(len(imported), 4)
        self.assertTrue(all(row["lambda_history"] == [0.5] * 32 for row in imported))

    def test_legacy_import_repairs_hardcoded_false_eos_metadata(self) -> None:
        prompt = PromptRecord("p", "prompt", "negative")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "v1.jsonl"
            rows = [
                {
                    "prompt_id": "p",
                    "prompt": "prompt",
                    "prompt_sentiment_class": "negative",
                    "method": old_method,
                    "seed": 1,
                    "generated_text": "done",
                    "selected_token_ids": [1, 99],
                    "beta_history": [15.0, 15.0],
                    "beta_max": 30.0,
                    "base_lm_perplexity": 7.0,
                    "classifier_sentiment_success": 1,
                    "classifier_target_probability": 0.9,
                    "classifier_predicted_label": "POSITIVE",
                    "classifier_predicted_label_id": 1,
                    "eos_generated": False,
                    "latency": {
                        "total_generation_time": 1.0,
                        "average_latency_per_token": 0.5,
                    },
                    "failure_status": "success",
                }
                for old_method in V1_METHOD_MAP
            ]
            content = "".join(json.dumps(row) + "\n" for row in rows)
            path.write_text(content, encoding="utf-8")
            imported = import_v1_records(
                path,
                prompts=[prompt],
                seeds=[1],
                eos_token_id=99,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), content)
        self.assertTrue(all(row["eos_generated"] for row in imported))
        self.assertTrue(all(row["terminated_on_eos"] for row in imported))
        self.assertTrue(all(row["eos_metadata_mismatch"] for row in imported))
        self.assertTrue(
            all(row["source_eos_generated"] is False for row in imported)
        )

    def test_legacy_import_disambiguates_reused_source_ids(self) -> None:
        prompts = [
            PromptRecord(
                "rad:test:neutral:00000163",
                "neutral prompt",
                "neutral",
                source_prompt_id="reused-md5",
                source_index=163,
            ),
            PromptRecord(
                "rad:test:positive:00000223",
                "positive prompt",
                "positive",
                source_prompt_id="reused-md5",
                source_index=223,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "v1.jsonl"
            rows = []
            for prompt in prompts:
                for old_method in V1_METHOD_MAP:
                    rows.append(
                        {
                            "prompt_id": prompt.source_prompt_id,
                            "prompt": prompt.prompt,
                            "prompt_sentiment_class": (
                                prompt.prompt_sentiment_class
                            ),
                            "method": old_method,
                            "seed": 1,
                            "generated_text": "text",
                            "selected_token_ids": [1] * 32,
                            "beta_history": [15.0] * 32,
                            "beta_max": 30.0,
                            "base_lm_perplexity": 7.0,
                            "classifier_sentiment_success": 1,
                            "classifier_target_probability": 0.9,
                            "classifier_predicted_label": "POSITIVE",
                            "classifier_predicted_label_id": 1,
                            "latency": {
                                "total_generation_time": 1.0,
                                "average_latency_per_token": 1 / 32,
                            },
                            "failure_status": "success",
                        }
                    )
            content = "".join(json.dumps(row) + "\n" for row in rows)
            path.write_text(content, encoding="utf-8")
            imported = import_v1_records(
                path,
                prompts=prompts,
                seeds=[1],
                eos_token_id=99,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), content)
        self.assertEqual(len(imported), 8)
        self.assertEqual(
            {row["prompt_id"] for row in imported},
            {prompt.prompt_id for prompt in prompts},
        )
        self.assertEqual(
            {row["source_prompt_id"] for row in imported},
            {"reused-md5"},
        )
        self.assertEqual(
            {row["source_index"] for row in imported},
            {163, 223},
        )

    def test_generation_resume_does_not_duplicate_records(self) -> None:
        class FakeDecoder:
            calls = 0

            def generate(self, prompt: str, spec: RouteSpec, *, seed: int):
                del prompt, spec, seed
                self.calls += 1
                return V2Generation(
                    generated_text="text",
                    selected_token_ids=[1] * 32,
                    lambda_history=[0.0] * 32,
                    base_entropy_history=[0.0] * 32,
                    guide_entropy_history=[0.0] * 32,
                    js_history=[0.0] * 32,
                    topk_overlap_history=[1.0] * 32,
                    base_conditional_perplexity=2.0,
                    eos_generated=False,
                    terminated_on_eos=False,
                    latency={
                        "total_generation_time": 1.0,
                        "average_latency_per_token": 1 / 32,
                    },
                )

        decoder = FakeDecoder()
        prompt = PromptRecord("p", "prompt", "negative")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.jsonl"
            first = _run_generation_jobs(
                decoder=decoder,
                prompts=[prompt],
                seeds=[1],
                specs=[RouteSpec("base", "base")],
                output_path=path,
                log_every_jobs=1,
            )
            second = _run_generation_jobs(
                decoder=decoder,
                prompts=[prompt],
                seeds=[1],
                specs=[RouteSpec("base", "base")],
                output_path=path,
                log_every_jobs=1,
            )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(decoder.calls, 1)


if __name__ == "__main__":
    unittest.main()
