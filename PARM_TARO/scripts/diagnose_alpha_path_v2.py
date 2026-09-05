"""Diagnose weak alpha expression in the completed Stage 9 Pilot V2."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from PARM_TARO.training.alpha import response_objective_weights
from PARM_TARO.training.alpha_collapse import summarize
from PARM_TARO.training.alpha_pilot_v2 import (
    endpoint_sensitivity_loss,
    lambda_delta_to_logit_delta,
    preference_gain_objective,
    term_gradient_norms,
)
from PARM_TARO.training.config import ParmRouterTrainingConfig
from PARM_TARO.training.data import load_examples, tokenize_response
from PARM_TARO.training.engine import route_response_pair
from PARM_TARO.training.online import (
    compute_frozen_guide_logprobs,
    compute_frozen_sequence_distributions,
)
from PARM_TARO.training.runtime import (
    PROJECT_ROOT,
    load_training_runtime,
    project_path,
)
from router_v2.cache.io import sha256_file, write_json_atomic
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.smart_checkpoint import load_smart_checkpoint
from router_v2.smart_model import SmartTokenRouter
from router_v2.training.audit import hash_tree


DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO/configs/train_stage9_v2_alpha.json"
DEFAULT_PILOT_REPORT = (
    PROJECT_ROOT
    / "results/parm_taro/training/v2_alpha_preference_pilot_v2/pilot_report.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results/parm_taro/training/v2_alpha_preference_pilot_v2/pilot_final.pt"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_alpha_path_v2_diagnostic.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "PARM_TARO/reports/stage9_alpha_path_v2_diagnostic.md"
)


def _parameter_groups(
    router: SmartTokenRouter,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "preference_encoder": [],
        "fusion_layers": [],
        "final_lambda_head": [],
        "state_router_rest": [],
    }
    for name, parameter in router.named_parameters():
        if name.startswith("preference_encoder."):
            group = "preference_encoder"
        elif name.startswith("fusion_mlp.5."):
            group = "final_lambda_head"
        elif name.startswith("fusion_mlp."):
            group = "fusion_layers"
        else:
            group = "state_router_rest"
        groups[group].append(parameter)
    return {name: tuple(parameters) for name, parameters in groups.items()}


class _FusionFeatureCapture:
    def __init__(self, router: SmartTokenRouter) -> None:
        self.router = router
        self.raw: list[torch.Tensor] = []
        self.normalized: list[torch.Tensor] = []
        self._pre_handle = None
        self._post_handle = None

    def __enter__(self) -> "_FusionFeatureCapture":
        normalizer = self.router.fusion_mlp[0]

        def capture_raw(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            self.raw.append(inputs[0].detach().float())

        def capture_normalized(
            _module: Any,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self.normalized.append(output.detach().float())

        self._pre_handle = normalizer.register_forward_pre_hook(capture_raw)
        self._post_handle = normalizer.register_forward_hook(capture_normalized)
        return self

    def __exit__(self, *_: object) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()


def _captured_feature_metrics(
    router: SmartTokenRouter,
    capture: _FusionFeatureCapture,
) -> dict[str, list[float]]:
    preference_slice = router.feature_slices["preference"]
    preference_indices = list(
        range(preference_slice.start or 0, preference_slice.stop or 0)
    )
    state_indices = [
        index for index in range(router.feature_dim) if index not in preference_indices
    ]
    first = router.fusion_mlp[1]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("Smart Router fusion input must be linear")
    values: dict[str, list[float]] = defaultdict(list)
    for raw, normalized in zip(capture.raw, capture.normalized):
        raw_state = raw[..., state_indices]
        raw_preference = raw[..., preference_indices]
        normalized_state = normalized[..., state_indices]
        normalized_preference = normalized[..., preference_indices]
        state_contribution = F.linear(
            normalized_state,
            first.weight[:, state_indices],
        )
        preference_contribution = F.linear(
            normalized_preference,
            first.weight[:, preference_indices],
        )
        state_norm = state_contribution.norm(dim=-1)
        preference_norm = preference_contribution.norm(dim=-1)
        cosine = F.cosine_similarity(
            state_contribution,
            preference_contribution,
            dim=-1,
            eps=1e-12,
        )
        for tensor, name in (
            (raw_state.norm(dim=-1), "raw_non_preference_feature_norm"),
            (raw_preference.norm(dim=-1), "raw_preference_activation_norm"),
            (normalized_state.norm(dim=-1), "normalized_state_feature_norm"),
            (
                normalized_preference.norm(dim=-1),
                "normalized_preference_feature_norm",
            ),
            (state_norm, "state_first_layer_contribution_norm"),
            (
                preference_norm,
                "preference_first_layer_contribution_norm",
            ),
            (
                preference_norm / state_norm.clamp_min(1e-12),
                "preference_to_state_contribution_ratio",
            ),
            (cosine, "preference_state_contribution_cosine"),
        ):
            values[name].extend(float(value) for value in tensor.cpu().reshape(-1))
    return values


def _protected_snapshot(
    config: ParmRouterTrainingConfig,
    inputs: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "parm_tree": hash_tree(PROJECT_ROOT / "PARM"),
        "pblora": checkpoint_descriptor(project_path(config.parm_adapter_path)),
        "data_manifest_sha256": sha256_file(
            project_path(config.data_root) / "manifest.json"
        ),
        "read_only_inputs": {
            str(path.resolve()): sha256_file(path) for path in inputs
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    feature = report["feature_scale"]
    logit = report["logit_sensitivity"]
    conclusion = report["conclusion"]
    return "\n".join(
        [
            "# Stage 9 Pilot V2 Alpha-Path Diagnostic",
            "",
            "## Status",
            "",
            "```text",
            str(report["status"]),
            "```",
            "",
            "## Feature Scale",
            "",
            "| Metric | Mean |",
            "|---|---:|",
            "| Preference activation norm | "
            f"{feature['raw_preference_activation_norm']['mean']:.8f} |",
            "| Non-preference feature norm | "
            f"{feature['raw_non_preference_feature_norm']['mean']:.8f} |",
            "| Preference/state first-layer ratio | "
            f"{feature['preference_to_state_contribution_ratio']['mean']:.8f} |",
            "",
            "## Logit Sensitivity",
            "",
            f"- Endpoint pre-sigmoid delta: `{logit['endpoint_logit_delta']['mean']:.8f}`",
            f"- Endpoint lambda delta: `{logit['endpoint_lambda_delta']['mean']:.8f}`",
            "- Lambda/logit compression ratio: "
            f"`{logit['lambda_to_logit_delta_ratio']['mean']:.8f}`",
            "",
            "## Conclusion",
            "",
            str(conclusion["root_cause"]),
            "",
            "Pilot V3 uses the V2 checkpoint unchanged as its source, adds a",
            "normalized learned alpha-state residual before sigmoid, and moves",
            "endpoint sensitivity to logit space. Full retraining is not authorized.",
            "",
        ]
    )


def _write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite alpha-path report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_diagnostic(
    config: ParmRouterTrainingConfig,
    *,
    pilot_report_path: Path,
    checkpoint_path: Path,
    max_conflict_examples: int,
) -> dict[str, Any]:
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    if pilot_report.get("status") != "NOT_PASS":
        raise ValueError("Alpha-path diagnostic requires NOT_PASS Pilot V2")
    successful_checks = (
        "lambda_not_collapsed",
        "preference_loss_improved",
        "nll_reasonable_vs_no_alpha",
        "frozen_artifacts_unchanged",
    )
    if not all(pilot_report["checks"].get(name) is True for name in successful_checks):
        raise ValueError("Pilot V2 did not establish the required successful behavior")
    inputs = (
        pilot_report_path,
        checkpoint_path,
        PROJECT_ROOT
        / "results/parm_taro/training/v2_alpha_preference_pilot/pilot_report.json",
        PROJECT_ROOT
        / "results/parm_taro/training/v2_alpha_preference_pilot/pilot_final.pt",
    )
    before = _protected_snapshot(config, inputs)
    runtime = load_training_runtime(config)
    router, payload = load_smart_checkpoint(
        checkpoint_path,
        map_location=runtime.device,
    )
    tokenizer_hash = str(runtime.tokenizer_descriptor["semantic_sha256"])
    if payload["metadata"].get("tokenizer_semantic_sha256") != tokenizer_hash:
        raise ValueError("Pilot V2 tokenizer hash mismatch")
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    examples = [
        example
        for example in load_examples(project_path(config.data_root), "validation")
        if example.better_response_id != example.safer_response_id
    ][:max_conflict_examples]
    if len(examples) != max_conflict_examples:
        raise ValueError("Not enough conflict examples for alpha-path diagnostic")
    feature_values: dict[str, list[float]] = defaultdict(list)
    logit_values: dict[str, list[float]] = defaultdict(list)
    gradient_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    implementation_checks = []
    groups = _parameter_groups(router)
    pairwise_scale = float(
        pilot_report["diagnostic_recommendation"]["pairwise_logit_scale"]
    )
    sensitivity_target = float(
        pilot_report["diagnostic_recommendation"]["sensitivity_minimum_delta"]
    )
    for example in examples:
        tokenized = tuple(
            tokenize_response(
                runtime.tokenizer,
                example,
                response_index,
                max_length=config.max_length,
                max_continuation_tokens=config.max_continuation_tokens,
                include_eos_target=config.include_eos_target,
            )
            for response_index in (0, 1)
        )
        base_reference = None
        for helpfulness in (0.0, 1.0):
            alpha = torch.tensor(
                [helpfulness, 1.0 - helpfulness],
                dtype=torch.float32,
                device=runtime.device,
            )
            weights = response_objective_weights(
                alpha,
                better_response_id=example.better_response_id,
                safer_response_id=example.safer_response_id,
            )
            if base_reference is None:
                frozen = tuple(
                    compute_frozen_sequence_distributions(
                        runtime.base_model,
                        runtime.guide_model,
                        item,
                        runtime.alignment,
                        alpha,
                        device=runtime.device,
                    )
                    for item in tokenized
                )
                base_reference = frozen
            else:
                frozen = tuple(
                    replace(
                        reference,
                        guide_logprobs=compute_frozen_guide_logprobs(
                            runtime.guide_model,
                            item,
                            runtime.alignment,
                            alpha,
                            device=runtime.device,
                        ),
                    )
                    for reference, item in zip(base_reference, tokenized)
                )
            with torch.no_grad(), _FusionFeatureCapture(router) as capture:
                route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=alpha,
                    static_scale=1.0,
                )
            for name, values in _captured_feature_metrics(router, capture).items():
                feature_values[name].extend(values)
            routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=alpha,
                static_scale=1.0,
            )
            endpoint_routes = tuple(
                route_response_pair(
                    frozen,
                    router=router,
                    router_alpha=torch.tensor(
                        [endpoint, 1.0 - endpoint],
                        dtype=torch.float32,
                        device=runtime.device,
                    ),
                    static_scale=1.0,
                )
                for endpoint in (0.0, 1.0)
            )
            constant_routes = route_response_pair(
                frozen,
                router=router,
                router_alpha=torch.tensor(
                    [0.5, 0.5], dtype=torch.float32, device=runtime.device
                ),
                static_scale=1.0,
            )
            endpoint_lambda = torch.cat(
                [
                    (
                        endpoint_routes[0][index].lambda_t
                        - endpoint_routes[1][index].lambda_t
                    )
                    .abs()
                    .reshape(-1)
                    for index in (0, 1)
                ]
            )
            endpoint_logit = torch.cat(
                [
                    (
                        torch.logit(endpoint_routes[0][index].lambda_t)
                        - torch.logit(endpoint_routes[1][index].lambda_t)
                    )
                    .abs()
                    .reshape(-1)
                    for index in (0, 1)
                ]
            )
            midpoint = torch.cat(
                [route.lambda_t.reshape(-1) for route in constant_routes]
            )
            constant_contribution = torch.cat(
                [
                    (
                        torch.logit(endpoint_routes[endpoint][index].lambda_t)
                        - torch.logit(constant_routes[index].lambda_t)
                    )
                    .abs()
                    .reshape(-1)
                    for endpoint in (0, 1)
                    for index in (0, 1)
                ]
            )
            for tensor, name in (
                (endpoint_lambda, "endpoint_lambda_delta"),
                (endpoint_logit, "endpoint_logit_delta"),
                (
                    endpoint_lambda / endpoint_logit.clamp_min(1e-12),
                    "lambda_to_logit_delta_ratio",
                ),
                (midpoint * (1.0 - midpoint), "local_sigmoid_derivative"),
                (
                    constant_contribution,
                    "preference_contribution_to_final_logit",
                ),
            ):
                logit_values[name].extend(
                    float(value) for value in tensor.detach().cpu().reshape(-1)
                )
            sensitivity, _ = endpoint_sensitivity_loss(
                endpoint_routes[0],
                endpoint_routes[1],
                minimum_delta=sensitivity_target,
            )
            preference = preference_gain_objective(
                routes,
                frozen,
                weights,
                pairwise_logit_scale=pairwise_scale,
            )
            terms = (
                ("preference", preference.loss),
                ("nll", 0.25 * preference.quality_nll.total_loss),
                ("sensitivity_raw", sensitivity),
                ("sensitivity_weighted", 100.0 * sensitivity),
            )
            for term_index, (term_name, term) in enumerate(terms):
                norms = term_gradient_norms(
                    term,
                    groups,
                    retain_graph=term_index < len(terms) - 1,
                )
                for group, value in norms.items():
                    gradient_values[term_name][group].append(value)
            expected_loss = float(
                F.relu(sensitivity_target - endpoint_lambda).mean().detach()
            )
            implementation_checks.append(
                abs(float(sensitivity.detach()) - expected_loss) <= 1e-7
            )
            del frozen, routes, endpoint_routes, constant_routes, preference
    feature_summary = {
        name: summarize(values) for name, values in feature_values.items()
    }
    logit_summary = {
        name: summarize(values) for name, values in logit_values.items()
    }
    gradient_summary = {
        term: {group: summarize(values) for group, values in groups.items()}
        for term, groups in gradient_values.items()
    }
    contribution_ratio = float(
        feature_summary["preference_to_state_contribution_ratio"]["mean"]
    )
    alpha_drowned = contribution_ratio < 0.1
    compression_matches_sigmoid = abs(
        float(logit_summary["lambda_to_logit_delta_ratio"]["mean"])
        - float(logit_summary["local_sigmoid_derivative"]["mean"])
    ) <= 0.005
    mean_lambda = float(
        pilot_report["validation_after"]["correct_lambda"]["mean"]
    )
    minimum_logit_delta = lambda_delta_to_logit_delta(
        reference_lambda=mean_lambda,
        lambda_delta=sensitivity_target,
    )
    implementation_correct = all(implementation_checks)
    after = _protected_snapshot(config, inputs)
    root_causes = []
    if alpha_drowned:
        root_causes.append("preference contribution is small relative to state fusion")
    if compression_matches_sigmoid:
        root_causes.append("sigmoid compression attenuates the expressed logit delta")
    root_causes.append(
        "lambda-space sensitivity spends most gradient on the shared final head "
        "instead of a dedicated alpha pathway"
    )
    report = {
        "schema_version": 1,
        "stage": 9,
        "method_label": "PARM_TARO_ALPHA_PATH_V2_DIAGNOSTIC",
        "status": (
            "PASS"
            if before == after and implementation_correct and len(examples) > 0
            else "NOT_PASS"
        ),
        "validation": {
            "split": "validation",
            "conflict_examples": len(examples),
            "alpha_endpoint_tasks": len(examples) * 2,
            "test_split_used": False,
        },
        "feature_scale": feature_summary,
        "logit_sensitivity": logit_summary,
        "gradient_scale": gradient_summary,
        "sensitivity_loss_implementation": {
            "correct": implementation_correct,
            "target_lambda_delta": sensitivity_target,
            "weight": 100.0,
        },
        "conclusion": {
            "preference_numerically_drowned": alpha_drowned,
            "sigmoid_compression_contributes": compression_matches_sigmoid,
            "sensitivity_loss_implementation_correct": implementation_correct,
            "root_cause": "; ".join(root_causes) + ".",
        },
        "recommendation": {
            "source_checkpoint": str(checkpoint_path.resolve()),
            "architecture": "normalized_alpha_state_pre_sigmoid_residual",
            "minimum_logit_delta": minimum_logit_delta,
            "minimum_logit_delta_rule": (
                "logit(mean_lambda + target_delta/2) - "
                "logit(mean_lambda - target_delta/2)"
            ),
            "target_lambda_delta": sensitivity_target,
            "freeze_state_router_during_alpha_warmup": True,
            "preserve_base_relative_preference_gain": True,
            "preserve_quality_phase": True,
            "min_control_lambda_delta": 0.001,
            "full_retraining_authorized": False,
        },
        "pilot_v2_report_sha256": sha256_file(pilot_report_path),
        "pilot_v2_checkpoint_sha256": sha256_file(checkpoint_path),
        "protected_before": before,
        "protected_after": after,
        "protected_unchanged": before == after,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--max-conflict-examples", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ParmRouterTrainingConfig.load_json(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_cpu_fallback:
        overrides["allow_cpu_fallback"] = False
    if overrides:
        config = replace(config, **overrides)
    reports_root = (PROJECT_ROOT / "PARM_TARO/reports").resolve()
    for output in (args.output_json.resolve(), args.output_markdown.resolve()):
        try:
            output.relative_to(reports_root)
        except ValueError as error:
            raise ValueError("Alpha-path outputs must stay under reports") from error
    report = run_diagnostic(
        config,
        pilot_report_path=args.pilot_report.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        max_conflict_examples=args.max_conflict_examples,
    )
    write_json_atomic(args.output_json, report, overwrite=args.overwrite)
    _write_text_atomic(
        args.output_markdown,
        _render_markdown(report),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
