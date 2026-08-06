from __future__ import annotations

import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from scripts.evaluate_router import (
    DEFAULT_FIXED_BETAS,
    ORIGINAL_FIXED_RAD_FILES,
    DecodingProtocol,
    EvaluationConfig,
    EvaluationProgress,
    PromptRecord,
    ProgressSettings,
    ResolvedDevices,
    aggregate_metrics,
    append_jsonl_record,
    assert_no_nan_inf_rows,
    baseline_source_hashes,
    benjamini_hochberg,
    bootstrap_ci,
    build_raw_record,
    collect_device_diagnostics,
    cuda_oom_message,
    dtype_for_device,
    classifier_label_mapping_from_config,
    completed_jobs_in_plan,
    effect_size_cohens_d,
    estimate_remaining_seconds,
    evaluation_jobs,
    format_beta,
    generation_key,
    independent_sentiment_metrics,
    load_completed_keys,
    load_prompt_split,
    method_names,
    paired_bootstrap_test,
    paired_permutation_test,
    paired_rows,
    pareto_points,
    pending_evaluation_jobs,
    query_nvidia_smi,
    resolve_component_devices,
    resolve_device,
    sample_to_metric_record,
    select_best_fixed_beta,
    select_best_heuristic,
    sha256_file,
    statistical_tests,
    total_evaluation_jobs,
    validate_decoding_protocol,
    write_csv,
)
from Method.RAD.adaptive_router_decoding import AdaptiveRADSample


def sample(text: str, tokens: list[int] | None = None, beta_history: list[float] | None = None) -> AdaptiveRADSample:
    token_ids = tokens or [10, 11]
    betas = beta_history if beta_history is not None else []
    return AdaptiveRADSample(
        prompt="prompt",
        text=text,
        token_ids=token_ids,
        beta_history=betas,
        gate_history=[],
        selected_token_ids=token_ids,
        candidate_ids_history=[[1, 2, 3]] * len(token_ids),
        base_logits_history=[[0.1, 0.2, 0.3]] * len(token_ids),
        rad_reward_scores_history=[[0.0, 0.5, 1.0]] * len(token_ids),
        guided_scores_history=[[0.1, 5.2, 10.3]] * len(token_ids),
        latency={
            "latency_scope": "batch",
            "total_generation_time": 1.0,
            "average_latency_per_token": 0.5,
            "per_token_total_latency": [0.5] * len(token_ids),
            "base_lm_latency": 0.2,
            "reward_model_latency": 0.2,
            "router_latency": 0.05,
            "sampling_latency": 0.05,
        },
    )


class RouterEvaluationTests(unittest.TestCase):
    def test_same_prompts_and_seeds_across_methods(self) -> None:
        config = EvaluationConfig(seeds=[1, 2], fixed_beta_sweep=[0.0, 10.0])
        methods = method_names(config)
        prompts = [PromptRecord("p1", "bad prompt", "negative"), PromptRecord("p2", "good prompt", "positive")]
        keys = {(prompt.prompt_id, method, seed) for prompt in prompts for method in methods for seed in config.seeds}
        self.assertEqual(len(keys), len(prompts) * len(methods) * len(config.seeds))
        self.assertIn(("p1", "learned_router", 2), keys)
        self.assertIn(("p2", "fixed_beta_10", 1), keys)

    def test_validation_test_separation_and_best_beta_selected_from_validation(self) -> None:
        validation_aggregate = [
            {"prompt_sentiment_class": "negative", "method": "fixed_beta_0", "metric": "classifier_sentiment_success", "mean": 0.2},
            {"prompt_sentiment_class": "negative", "method": "fixed_beta_10", "metric": "classifier_sentiment_success", "mean": 0.8},
        ]
        self.assertEqual(select_best_fixed_beta(validation_aggregate, [0.0, 10.0]), 10.0)
        test_aggregate = [
            {"prompt_sentiment_class": "negative", "method": "fixed_beta_0", "metric": "classifier_sentiment_success", "mean": 1.0},
            {"prompt_sentiment_class": "negative", "method": "fixed_beta_10", "metric": "classifier_sentiment_success", "mean": 0.0},
        ]
        self.assertEqual(select_best_fixed_beta(validation_aggregate, [0.0, 10.0]), 10.0)
        self.assertEqual(select_best_fixed_beta(test_aggregate, [0.0, 10.0]), 0.0)

    def test_best_heuristic_selected_from_validation(self) -> None:
        validation_aggregate = [
            {"prompt_sentiment_class": "negative", "method": "heuristic_linear_increase", "metric": "classifier_sentiment_success", "mean": 0.3},
            {"prompt_sentiment_class": "negative", "method": "heuristic_reward_range_beta", "metric": "classifier_sentiment_success", "mean": 0.7},
        ]
        self.assertEqual(select_best_heuristic(validation_aggregate), "heuristic_reward_range_beta")

    def test_independent_evaluator_is_not_rad_reward_model(self) -> None:
        validate_decoding_protocol(EvaluationConfig(independent_sentiment_evaluator_id="sentiment-roberta-large-english"))
        with self.assertRaises(ValueError):
            validate_decoding_protocol(EvaluationConfig(independent_sentiment_evaluator_id="rad_reward_model"))

    def test_classifier_label_mapping(self) -> None:
        mapping = classifier_label_mapping_from_config(Path("models/sentiment-roberta-large-english"))
        self.assertEqual(mapping["label2id"]["POSITIVE"], 1)
        self.assertEqual(mapping["id2label"]["1"], "POSITIVE")

    def test_pilot_full_method_separation(self) -> None:
        pilot = EvaluationConfig(methods=["base_lm", "fixed_beta_0", "fixed_beta_10", "fixed_beta_30", "learned_router"])
        self.assertEqual(method_names(pilot), ["base_lm", "fixed_beta_0", "fixed_beta_10", "fixed_beta_30", "learned_router"])
        full = EvaluationConfig(methods=[])
        self.assertIn("best_heuristic", method_names(full))

    def test_pilot_full_split_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for klass in ["negative", "neutral", "positive"]:
                path = root / f"{klass}_prompts.jsonl"
                rows = [
                    {"md5_hash": f"{klass}-{index}", "prompt": {"text": f"{klass} prompt {index}"}}
                    for index in range(4)
                ]
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            config = EvaluationConfig(dataset_dir=str(root), validation_prompt_limit_per_class=2, test_prompt_limit_per_class=2)
            validation_ids = {record.prompt_id for record in load_prompt_split(config, "validation")}
            test_ids = {record.prompt_id for record in load_prompt_split(config, "test")}
            self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_protocol_validation(self) -> None:
        with self.assertRaises(ValueError):
            validate_decoding_protocol(EvaluationConfig(decoding=DecodingProtocol(top_k=19)))
        with self.assertRaises(ValueError):
            validate_decoding_protocol(EvaluationConfig(decoding=DecodingProtocol(max_reward_length=128)))
        with self.assertRaises(ValueError):
            validate_decoding_protocol(EvaluationConfig(decoding=DecodingProtocol(do_sample=True, temperature=0.0)))

    def test_paired_metric_alignment_by_prompt_id_and_seed(self) -> None:
        rows = [
            {"prompt_id": "p1", "seed": 1, "method": "learned_router", "score": 0.8},
            {"prompt_id": "p1", "seed": 1, "method": "base_lm", "score": 0.2},
            {"prompt_id": "p1", "seed": 2, "method": "learned_router", "score": 0.7},
            {"prompt_id": "p2", "seed": 1, "method": "base_lm", "score": 0.1},
        ]
        self.assertEqual(paired_rows(rows, "learned_router", "base_lm", "score"), [(0.8, 0.2)])

    def test_resume_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            record = {"prompt_id": "p1", "method": "base_lm", "seed": 1}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(load_completed_keys(path), {("p1", "base_lm", 1)})
            self.assertEqual(generation_key(record), ("p1", "base_lm", 1))

    def test_total_jobs_and_resume_progress_initialization(self) -> None:
        prompts = [PromptRecord("p1", "a", "negative"), PromptRecord("p2", "b", "neutral")]
        methods = ["base_lm", "learned_router"]
        seeds = [1, 2, 3]
        jobs = evaluation_jobs(prompts, methods, seeds)
        completed = {("p1", "base_lm", 1), ("outside", "base_lm", 1)}
        self.assertEqual(total_evaluation_jobs(prompts, methods, seeds), 12)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(completed_jobs_in_plan(jobs, completed), 1)
        self.assertEqual(len(pending_evaluation_jobs(jobs, completed)), 11)

    def test_completed_job_is_not_run_again(self) -> None:
        prompts = [PromptRecord("p1", "a", "negative")]
        jobs = evaluation_jobs(prompts, ["base_lm", "learned_router"], [1])
        pending = pending_evaluation_jobs(jobs, {("p1", "base_lm", 1)})
        self.assertEqual([job.method for job in pending], ["learned_router"])

    def test_failure_still_advances_progress_and_eta_non_negative(self) -> None:
        prompt = PromptRecord("p1", "a", "negative")
        tracker = EvaluationProgress(
            "test progress",
            total=2,
            initial=0,
            settings=ProgressSettings(enabled=False, log_every_jobs=1, log_every_seconds=0),
        )
        tracker.update(job=evaluation_jobs([prompt], ["base_lm"], [1])[0], success=False)
        self.assertEqual(tracker.completed, 1)
        self.assertEqual(tracker.failures, 1)
        self.assertGreaterEqual(estimate_remaining_seconds(1, 2, 10.0), 0.0)
        tracker.close()

    def test_jsonl_flush_after_each_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                append_jsonl_record(handle, {"prompt_id": "p1", "method": "base_lm", "seed": 1})
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_progress_does_not_change_evaluation_output_record(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        row = build_raw_record(prompt, "base_lm", 1, sample("great", [1]), 30.0)
        before = json.dumps(row, sort_keys=True)
        tracker = EvaluationProgress("test progress", 1, 0, ProgressSettings(enabled=False))
        tracker.update(job=evaluation_jobs([prompt], ["base_lm"], [1])[0], success=True)
        tracker.close()
        self.assertEqual(json.dumps(row, sort_keys=True), before)

    def test_confidence_interval_reproducibility(self) -> None:
        values = [0.0, 1.0, 1.0, 0.0, 1.0]
        self.assertEqual(bootstrap_ci(values, seed=7, samples=200), bootstrap_ci(values, seed=7, samples=200))

    def test_significance_tests_correctness_on_synthetic_data(self) -> None:
        pairs = [(2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]
        boot = paired_bootstrap_test(pairs, seed=1, samples=200)
        self.assertGreater(boot["mean_difference"], 0.0)
        self.assertLessEqual(paired_permutation_test(pairs, seed=1, samples=200), 1.0)
        self.assertTrue(math.isfinite(effect_size_cohens_d(pairs)))
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))

    def test_router_history_length_consistency(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        raw = build_raw_record(prompt, "learned_router", 1, sample("great", [1, 2, 3], [1.0, 2.0, 3.0]), 30.0)
        self.assertEqual(len(raw["beta_history"]), len(raw["selected_token_ids"]))

    def test_aggregate_metrics_matching_per_sample_records(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        raws = [
            {**build_raw_record(prompt, "base_lm", 1, sample("bad terrible", [1]), 30.0), "classifier_sentiment_success": 0, "classifier_target_probability": 0.1},
            {**build_raw_record(prompt, "base_lm", 2, sample("great good", [2]), 30.0), "classifier_sentiment_success": 1, "classifier_target_probability": 0.9},
        ]
        metrics = [sample_to_metric_record(row) for row in raws]
        aggregate = aggregate_metrics(metrics, bootstrap_seed=1, bootstrap_samples=100)
        success = [
            row for row in aggregate
            if row["method"] == "base_lm" and row["metric"] == "target_sentiment_success"
        ][0]
        self.assertEqual(success["n"], 2)
        self.assertAlmostEqual(success["mean"], 0.5)

    def test_no_nan_inf_in_metrics(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        raw = build_raw_record(prompt, "base_lm", 1, sample("great good", [1]), 30.0, base_lm_perplexity=12.0)
        raw.update({
            "classifier_sentiment_success": 1,
            "classifier_target_probability": 0.9,
            "classifier_predicted_label": "POSITIVE",
            "classifier_predicted_label_id": 1,
        })
        metrics = sample_to_metric_record(raw)
        numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
        self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
        with self.assertRaises(ValueError):
            assert_no_nan_inf_rows([{"bad": float("nan")}], "test")

    def test_pareto_points_and_stats_outputs(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        raws = [
            build_raw_record(prompt, "learned_router", 1, sample("great good", [1], [5.0]), 30.0),
            build_raw_record(prompt, "base_lm", 1, sample("bad", [2]), 30.0),
        ]
        metrics = [sample_to_metric_record(row) for row in raws]
        aggregate = aggregate_metrics(metrics, bootstrap_seed=1, bootstrap_samples=100)
        points = pareto_points(aggregate)
        self.assertTrue(points)
        stats = statistical_tests(metrics, ["base_lm"], seed=1, samples=100)
        self.assertTrue(stats)

    def test_fixed_beta_sweep_contains_required_grid(self) -> None:
        self.assertTrue(set(DEFAULT_FIXED_BETAS).issuperset({0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0}))
        self.assertEqual(format_beta(10.0), "10")
        self.assertEqual(format_beta(1.5), "1p5")

    def test_original_fixed_beta_rad_remains_unchanged(self) -> None:
        before = baseline_source_hashes()
        self.assertTrue(before)
        after = {str(path): sha256_file(path) for path in ORIGINAL_FIXED_RAD_FILES if path.exists()}
        self.assertEqual(before, after)

    def test_csv_write_and_sentiment_metrics(self) -> None:
        self.assertGreater(independent_sentiment_metrics("great wonderful")["lexicon_sentiment_score"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            write_csv(path, [{"a": 1, "b": 2}])
            self.assertIn("a,b", path.read_text(encoding="utf-8").splitlines()[0])

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={"available": True, "nvidia_smi_cuda_version": "12.8"})
    @mock.patch("torch.cuda.get_device_properties")
    @mock.patch("torch.cuda.synchronize")
    @mock.patch("torch.empty")
    @mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3080")
    @mock.patch("torch.cuda.current_device", return_value=0)
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_auto_uses_cuda_when_probe_succeeds(
        self,
        _is_available: mock.Mock,
        _current_device: mock.Mock,
        _device_name: mock.Mock,
        empty: mock.Mock,
        _synchronize: mock.Mock,
        properties: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        properties.return_value = type("Props", (), {"total_memory": 10 * 1024**3})()
        empty.return_value = torch.zeros(1)
        self.assertEqual(str(resolve_device("auto")), "cuda:0")
        empty.assert_called_once()
        self.assertEqual(empty.call_args.kwargs["device"], torch.device("cuda:0"))

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={"available": False})
    @mock.patch("torch.cuda.current_device")
    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_resolve_device_auto_falls_back_cpu_when_cuda_unavailable(
        self,
        _is_available: mock.Mock,
        current_device: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        self.assertEqual(str(resolve_device("auto")), "cpu")
        current_device.assert_not_called()

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={"available": True, "nvidia_smi_cuda_version": "12.8"})
    @mock.patch("torch.cuda.device_count", return_value=1)
    @mock.patch("torch.cuda.current_device", side_effect=RuntimeError("driver too old"))
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_auto_falls_back_when_probe_fails_even_with_visible_gpu(
        self,
        _is_available: mock.Mock,
        _current_device: mock.Mock,
        _device_count: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        self.assertEqual(str(resolve_device("auto")), "cpu")

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={"available": True, "nvidia_smi_cuda_version": "12.8"})
    @mock.patch("torch.cuda.current_device", side_effect=RuntimeError("found version 12080"))
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_raises_instead_of_fallback(
        self,
        _is_available: mock.Mock,
        _current_device: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA was explicitly requested but is unavailable"):
            resolve_device("cuda")

    @mock.patch("torch.cuda.current_device")
    @mock.patch("torch.cuda.is_available")
    def test_explicit_cpu_does_not_probe_cuda(self, is_available: mock.Mock, current_device: mock.Mock) -> None:
        self.assertEqual(str(resolve_device("cpu")), "cpu")
        is_available.assert_not_called()
        current_device.assert_not_called()

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={"available": False})
    @mock.patch("torch.cuda.device_count")
    @mock.patch("torch.cuda.is_available")
    def test_cpu_diagnostics_do_not_initialize_cuda(
        self,
        is_available: mock.Mock,
        device_count: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        diagnostics = collect_device_diagnostics("cpu", torch.device("cpu"))
        self.assertFalse(diagnostics["cuda_available"])
        self.assertEqual(diagnostics["cuda_device_count"], 0)
        is_available.assert_not_called()
        device_count.assert_not_called()

    @mock.patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi"))
    def test_nvidia_smi_missing_does_not_crash(self, _run: mock.Mock) -> None:
        diagnostics = query_nvidia_smi()
        self.assertFalse(diagnostics["available"])
        self.assertIn("FileNotFoundError", diagnostics["error"])

    @mock.patch("scripts.evaluate_router.query_nvidia_smi", return_value={
        "available": True,
        "driver_version": "570.133.20",
        "nvidia_smi_cuda_version": "12.8",
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "gpu_vram_gb": 10.0,
    })
    @mock.patch("torch.cuda.device_count", return_value=1)
    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_collect_device_diagnostics_records_requested_and_resolved_device(
        self,
        _is_available: mock.Mock,
        _device_count: mock.Mock,
        _smi: mock.Mock,
    ) -> None:
        diagnostics = collect_device_diagnostics("auto", torch.device("cpu"), "torch.cuda.is_available() is False")
        self.assertEqual(diagnostics["requested_device"], "auto")
        self.assertEqual(diagnostics["resolved_device"], "cpu")
        self.assertEqual(diagnostics["driver_version"], "570.133.20")
        self.assertEqual(diagnostics["nvidia_smi_cuda_version"], "12.8")
        self.assertIn("is_available", diagnostics["fallback_reason"])

    @mock.patch("scripts.evaluate_router.collect_device_diagnostics")
    @mock.patch("scripts.evaluate_router.resolve_device")
    def test_per_component_device_resolution(
        self,
        resolve: mock.Mock,
        diagnostics: mock.Mock,
    ) -> None:
        resolve.side_effect = [
            torch.device("cuda:0"),
            torch.device("cuda:0"),
            torch.device("cuda:0"),
            torch.device("cpu"),
        ]
        diagnostics.side_effect = lambda requested, device: {"requested_device": requested, "resolved_device": str(device)}
        config = EvaluationConfig(devices={
            "base_model": "auto",
            "reward_model": "auto",
            "router": "auto",
            "sentiment_classifier": "cpu",
        })
        resolved = resolve_component_devices(config)
        self.assertEqual(str(resolved.base_model), "cuda:0")
        self.assertEqual(str(resolved.reward_model), "cuda:0")
        self.assertEqual(str(resolved.router), "cuda:0")
        self.assertEqual(str(resolved.sentiment_classifier), "cpu")
        self.assertEqual(resolve.call_count, 4)

    def test_dtype_policy_for_cpu_cuda_and_router(self) -> None:
        self.assertEqual(dtype_for_device("auto", torch.device("cpu"), "base_model"), torch.float32)
        self.assertEqual(dtype_for_device("auto", torch.device("cuda:0"), "base_model"), torch.float16)
        self.assertEqual(dtype_for_device("auto", torch.device("cuda:0"), "reward_model"), torch.float16)
        self.assertEqual(dtype_for_device("auto", torch.device("cuda:0"), "router"), torch.float32)

    def test_token_ids_keep_integer_dtype_when_moved_to_device(self) -> None:
        token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        self.assertEqual(token_ids.to(torch.device("cpu")).dtype, torch.long)
        self.assertEqual(attention_mask.to(torch.device("cpu")).dtype, torch.long)

    @mock.patch("scripts.evaluate_router.collect_device_diagnostics", return_value={
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "gpu_vram_gb": 10.0,
    })
    def test_cuda_oom_message_contains_actionable_guidance(self, _diagnostics: mock.Mock) -> None:
        message = cuda_oom_message("base model load", torch.device("cpu"))
        self.assertIn("CUDA out of memory during base model load", message)
        self.assertIn("keep sentiment classifier on CPU", message)
        self.assertIn("use float16", message)

    @mock.patch("scripts.evaluate_router.resolve_component_devices")
    def test_device_metadata_is_recorded_in_report_inputs(self, _resolve: mock.Mock) -> None:
        diagnostics = {
            "base_model": {"requested_device": "auto", "resolved_device": "cpu"},
            "sentiment_classifier": {"requested_device": "cpu", "resolved_device": "cpu"},
        }
        resolved = ResolvedDevices(
            base_model=torch.device("cpu"),
            reward_model=torch.device("cpu"),
            router=torch.device("cpu"),
            sentiment_classifier=torch.device("cpu"),
            diagnostics=diagnostics,
        )
        self.assertEqual(resolved.diagnostics["base_model"]["resolved_device"], "cpu")

    def test_device_changes_do_not_change_synthetic_output_record(self) -> None:
        prompt = PromptRecord("p1", "bad", "negative")
        row = build_raw_record(prompt, "base_lm", 1, sample("great", [1]), 30.0)
        before = json.dumps(sample_to_metric_record(row), sort_keys=True)
        _ = dtype_for_device("auto", torch.device("cpu"), "base_model")
        after = json.dumps(sample_to_metric_record(row), sort_keys=True)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
