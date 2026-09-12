from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PARM_TARO.evaluation.config import METHODS, ParmTaroEvaluationConfig
from PARM_TARO.evaluation.engine import analyze_records
from PARM_TARO.evaluation.io import (
    append_jsonl,
    read_jsonl,
    unique_rows,
    write_json_atomic,
)
from PARM_TARO.evaluation.metrics import (
    aggregate_methods,
    attach_normalized_objectives,
    attach_shared_preference_regret,
    hypervolume_2d,
    mean_inner_product,
    pareto_front,
    preference_cosine_similarity,
)
from PARM_TARO.evaluation.protocol import ObjectiveNormalizer, fit_normalizer
from PARM_TARO.evaluation.scoring import scorer_prerequisites
from PARM_TARO.evaluation.statistics import (
    correct_multiple_comparisons,
    holm_adjust,
    paired_bootstrap_ci,
    paired_effect_size,
    paired_metric_statistics,
    paired_permutation_pvalue,
)


ROOT = Path(__file__).resolve().parents[2]


def synthetic_records() -> list[dict[str, object]]:
    rows = []
    for prompt_index in range(3):
        for alpha_index, alpha in enumerate(((0.0, 1.0), (0.5, 0.5), (1.0, 0.0))):
            for method_index, method in enumerate(METHODS):
                helpfulness = 0.2 + 0.06 * method_index + 0.1 * alpha_index
                harmlessness = 0.8 - 0.03 * method_index - 0.1 * alpha_index
                rows.append(
                    {
                        "phase": "smoke",
                        "prompt_id": f"p{prompt_index}",
                        "source_index": prompt_index,
                        "prompt": "question words",
                        "method": method,
                        "seed": 2026,
                        "requested_alpha": list(alpha),
                        "router_alpha": list(alpha),
                        "guide_alpha": list(alpha),
                        "generated_text": "question response words remain coherent",
                        "selected_token_ids": [1, 2, 3],
                        "lambda_history": [0.01 + method_index * 0.01] * 3,
                        "latency_seconds": 0.03 + method_index * 0.001,
                        "base_conditional_perplexity": 2.0 + method_index * 0.1,
                        "helpfulness_raw": helpfulness,
                        "cost_raw": -harmlessness,
                        "harmlessness_raw": harmlessness,
                    }
                )
    return rows


class Stage10ConfigTests(unittest.TestCase):
    def test_shipped_config_freezes_methods_grid_and_production_checkpoint(self) -> None:
        config = ParmTaroEvaluationConfig.load_json(
            ROOT / "PARM_TARO/configs/evaluate_stage10_parm_taro.json"
        )
        self.assertEqual(config.methods, METHODS)
        self.assertEqual(config.alpha_grid[2], (0.5, 0.5))
        self.assertEqual(
            config.full_alpha_checkpoint_sha256,
            "56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20",
        )
        self.assertEqual(config.evaluation_split, "test_prompt_only")

    def test_shuffled_grid_is_a_deterministic_derangement(self) -> None:
        config = ParmTaroEvaluationConfig()
        shuffled = [config.shuffled_alpha(index) for index in range(5)]
        self.assertEqual(set(shuffled), set(config.alpha_grid))
        self.assertTrue(all(left != right for left, right in zip(shuffled, config.alpha_grid)))

    def test_protocol_rejects_changed_hv_reference_or_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            ParmTaroEvaluationConfig(hv_reference_point=(-1.0, -1.0))
        with self.assertRaisesRegex(ValueError, "64-token"):
            ParmTaroEvaluationConfig(max_new_tokens=32)

    def test_missing_official_scorers_are_reported_without_proxy(self) -> None:
        audit = scorer_prerequisites(
            replace(
                ParmTaroEvaluationConfig(),
                helpfulness_model_path="models/definitely-missing-stage10-help",
                harmlessness_cost_model_path="models/definitely-missing-stage10-cost",
            )
        )
        self.assertFalse(audit["ready"])
        self.assertIn("official_backend_selected", audit["checks"])
        self.assertIn("toxic-bert as Beaver harmlessness cost", audit["prohibited_substitutions"])


class ObjectiveMetricTests(unittest.TestCase):
    def test_validation_quantile_normalizer_clips_and_round_trips(self) -> None:
        records = [
            {"helpfulness_raw": float(index), "harmlessness_raw": float(index * 2)}
            for index in range(101)
        ]
        normalizer = fit_normalizer(records, lower_quantile=0.01, upper_quantile=0.99)
        self.assertEqual(normalizer.transform(-100.0, -100.0), (0.0, 0.0))
        self.assertEqual(normalizer.transform(1000.0, 1000.0), (1.0, 1.0))
        restored = ObjectiveNormalizer.from_dict(normalizer.to_dict())
        self.assertEqual(restored, normalizer)

    def test_pareto_and_hypervolume_known_example(self) -> None:
        points = [(1.0, 0.2), (0.5, 0.8), (0.2, 0.1)]
        self.assertEqual(pareto_front(points), [(0.5, 0.8), (1.0, 0.2)])
        self.assertAlmostEqual(hypervolume_2d(points, (0.0, 0.0)), 0.5)

    def test_hv_is_order_invariant_and_rejects_invalid_points(self) -> None:
        points = [(0.2, 0.9), (0.8, 0.3), (0.5, 0.6)]
        self.assertAlmostEqual(
            hypervolume_2d(points, (0.0, 0.0)),
            hypervolume_2d(list(reversed(points)), (0.0, 0.0)),
        )
        with self.assertRaises(ValueError):
            pareto_front([(math.nan, 1.0)])

    def test_mip_and_pcs_have_expected_geometry(self) -> None:
        self.assertAlmostEqual(mean_inner_product((1.0, 0.0), (0.75, 0.25)), 0.75)
        self.assertAlmostEqual(preference_cosine_similarity((1.0, 0.0), (1.0, 0.0)), 1.0)
        self.assertAlmostEqual(preference_cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_shared_regret_oracle_uses_all_methods_per_task(self) -> None:
        normalizer = ObjectiveNormalizer(0.0, 1.0, 0.0, 1.0)
        normalized = attach_normalized_objectives(synthetic_records(), normalizer)
        attached = attach_shared_preference_regret(normalized, required_methods=METHODS)
        self.assertTrue(all(float(row["preference_regret"]) >= 0.0 for row in attached))
        best_by_task = {}
        for row in attached:
            key = (row["prompt_id"], tuple(row["requested_alpha"]))
            best_by_task[key] = min(best_by_task.get(key, float("inf")), row["preference_regret"])
        self.assertTrue(all(value == 0.0 for value in best_by_task.values()))
        with self.assertRaisesRegex(ValueError, "pool"):
            attach_shared_preference_regret(normalized[:-1], required_methods=METHODS)

    def test_aggregate_keeps_hv_mip_pcs_regret_and_quality(self) -> None:
        normalizer = ObjectiveNormalizer(0.0, 1.0, 0.0, 1.0)
        rows = attach_shared_preference_regret(
            attach_normalized_objectives(synthetic_records(), normalizer),
            required_methods=METHODS,
        )
        aggregates = aggregate_methods(rows, (0.0, 0.0))
        self.assertEqual({row["method"] for row in aggregates}, set(METHODS))
        for row in aggregates:
            for metric in ("hypervolume", "mip", "pcs", "preference_regret", "perplexity", "coherence"):
                self.assertTrue(math.isfinite(float(row[metric])))


class StatisticalTests(unittest.TestCase):
    def test_end_to_end_synthetic_analysis_covers_all_primary_statistics(self) -> None:
        config = replace(
            ParmTaroEvaluationConfig(),
            bootstrap_samples=100,
            permutation_samples=100,
        )
        lock = {
            "normalization": ObjectiveNormalizer(0.0, 1.0, 0.0, 1.0).to_dict(),
            "pareto_hypervolume": {"reference_point_normalized": [0.0, 0.0]},
        }
        report = analyze_records(config, synthetic_records(), lock)
        self.assertEqual(len(report["method_aggregates"]), 7)
        self.assertEqual(len(report["paired_statistics"]), 24)
        self.assertTrue(
            all("holm_adjusted_pvalue" in row for row in report["paired_statistics"])
        )

    def test_bootstrap_and_permutation_are_prompt_clustered_and_deterministic(self) -> None:
        differences = {"a": [1.0, 2.0], "b": [2.0, 3.0], "c": [3.0, 4.0]}
        first = paired_bootstrap_ci(differences, samples=200, seed=17)
        second = paired_bootstrap_ci(differences, samples=200, seed=17)
        self.assertEqual(first, second)
        self.assertLess(
            paired_permutation_pvalue(differences, samples=1000, seed=17), 0.3
        )
        self.assertGreater(paired_effect_size([1.0, 2.0, 3.0]), 0.0)

    def test_holm_adjustment_is_monotone_and_not_below_raw(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_adjust(raw)
        self.assertTrue(all(value >= source for value, source in zip(adjusted, raw)))
        rows = correct_multiple_comparisons(
            [{"permutation_pvalue": value} for value in raw]
        )
        self.assertTrue(all("holm_adjusted_pvalue" in row for row in rows))

    def test_paired_statistics_use_prompt_alpha_seed_keys(self) -> None:
        rows = []
        for prompt in ("a", "b", "c"):
            for alpha in ((0.0, 1.0), (1.0, 0.0)):
                rows.extend(
                    (
                        {"prompt_id": prompt, "requested_alpha": alpha, "seed": 1, "method": "t", "mip": 2.0},
                        {"prompt_id": prompt, "requested_alpha": alpha, "seed": 1, "method": "c", "mip": 1.0},
                    )
                )
        report = paired_metric_statistics(
            rows,
            treatment="t",
            comparator="c",
            metric="mip",
            bootstrap_samples=200,
            permutation_samples=500,
            seed=11,
        )
        self.assertEqual(report["paired_tasks"], 6)
        self.assertEqual(report["prompt_clusters"], 3)
        self.assertEqual(report["mean_effect"], 1.0)


class JournalTests(unittest.TestCase):
    def test_atomic_json_supports_lists_and_explicit_report_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_atomic(path, [{"value": 1}])
            with self.assertRaises(FileExistsError):
                write_json_atomic(path, [{"value": 2}])
            write_json_atomic(path, [{"value": 2}], overwrite=True)
            self.assertEqual(json.loads(path.read_text()), [{"value": 2}])

    def test_resume_journal_recovers_only_trailing_partial_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            append_jsonl(path, {"id": 1})
            with path.open("ab") as handle:
                handle.write(b'{"id":')
            self.assertEqual(read_jsonl(path, recover_trailing_partial=True), [{"id": 1}])
            self.assertEqual(path.read_text(encoding="utf-8"), '{"id":1}\n')

    def test_duplicate_resume_keys_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            unique_rows([{"id": 1}, {"id": 1}], ("id",))


if __name__ == "__main__":
    unittest.main()
