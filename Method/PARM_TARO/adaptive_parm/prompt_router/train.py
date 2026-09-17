"""Leakage-safe leave-one-prompt-out training for tiny prompt routers."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extraction import FEATURE_COLUMNS, FEATURES, OUTPUT, ROOT


ORACLE = ROOT / "results/parm_taro/adaptive_parm/01_offline_oracle/oracle_by_case.csv"
CANDIDATES = ROOT / "results/parm_taro/adaptive_parm/01_offline_oracle/candidate_utility.csv"
ORACLE_SUMMARY = ROOT / "results/parm_taro/adaptive_parm/01_offline_oracle/oracle_summary.json"
WEIGHTS = (0.0, 0.01 / 1.01, 0.05 / 1.05, 0.1 / 1.1, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)
UTILITY_RIDGE_ALPHA = 10.0
CLASSIFIER_C = 0.1
SEED = 42


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def utility_design(state: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Small interaction basis allowing state-dependent utility curves."""
    w = weight.reshape(-1, 1)
    return np.concatenate((state, w, w**2, state * w, state * (w**2)), axis=1)


def _pairwise_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    correct = total = 0
    for left in range(len(true)):
        for right in range(left + 1, len(true)):
            delta = true[left] - true[right]
            if abs(delta) <= 1e-12:
                continue
            predicted_delta = predicted[left] - predicted[right]
            correct += int(predicted_delta * delta > 0)
            total += 1
    return correct / total if total else 1.0


def _state(row: Mapping[str, str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in FEATURE_COLUMNS], dtype=np.float64)


def train_and_evaluate() -> dict[str, Any]:
    for path in (FEATURES, ORACLE, CANDIDATES, ORACLE_SUMMARY):
        if not path.is_file(): raise FileNotFoundError(path)
    feature_rows = _read_csv(FEATURES)
    oracle_rows = _read_csv(ORACLE)
    candidate_rows = _read_csv(CANDIDATES)
    baseline = json.loads(ORACLE_SUMMARY.read_text(encoding="utf-8"))
    if len(feature_rows) != 60 or len(oracle_rows) != 60 or len(candidate_rows) != 480:
        raise ValueError("Unexpected prompt-router input coverage")
    features = {row["case_id"]: row for row in feature_rows}
    oracles = {row["case_id"]: row for row in oracle_rows}
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows: candidates[row["case_id"]].append(row)
    if set(features) != set(oracles) or set(features) != set(candidates):
        raise ValueError("Feature/oracle/candidate case IDs differ")
    if any(len(group) != len(WEIGHTS) for group in candidates.values()):
        raise ValueError("Each case must contain all eight candidate weights")

    prompts = sorted({row["sample_id"] for row in feature_rows})
    if len(prompts) != 12:
        raise ValueError("Leave-one-prompt-out requires exactly 12 prompt groups")
    weight_to_class = {weight: index for index, weight in enumerate(WEIGHTS)}
    action_rows: list[dict[str, Any]] = []
    utility_candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold, held_prompt in enumerate(prompts):
        train_cases = [case for case, row in features.items() if row["sample_id"] != held_prompt]
        test_cases = [case for case, row in features.items() if row["sample_id"] == held_prompt]
        if len(train_cases) != 55 or len(test_cases) != 5:
            raise ValueError("Grouped split must be 55 train / 5 validation cases")
        if {features[case]["sample_id"] for case in train_cases} & {held_prompt}:
            raise AssertionError("Prompt leakage detected")

        classifier = Pipeline((
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=CLASSIFIER_C, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ))
        classifier_x = np.stack([_state(features[case]) for case in train_cases])
        classifier_y = np.asarray([
            weight_to_class[float(oracles[case]["oracle_weight"])] for case in train_cases
        ])
        classifier.fit(classifier_x, classifier_y)

        train_utility_state = []
        train_utility_weight = []
        train_utility_y = []
        for case in train_cases:
            state = _state(features[case])
            for candidate in candidates[case]:
                train_utility_state.append(state)
                train_utility_weight.append(float(candidate["weight"]))
                train_utility_y.append(float(candidate["mip"]))
        utility = Pipeline((
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=UTILITY_RIDGE_ALPHA)),
        ))
        utility.fit(
            utility_design(np.stack(train_utility_state), np.asarray(train_utility_weight)),
            np.asarray(train_utility_y),
        )

        fold_action_rows = []
        fold_candidate_true = []
        fold_candidate_predicted = []
        fold_rank = []
        fold_pairwise = []
        for case in sorted(test_cases):
            state = _state(features[case])
            oracle = oracles[case]
            oracle_weight = float(oracle["oracle_weight"])
            oracle_ties = {float(value) for value in json.loads(oracle["oracle_tied_weights"])}
            ordered_candidates = sorted(candidates[case], key=lambda row: float(row["weight"]))
            weights = np.asarray([float(row["weight"]) for row in ordered_candidates])
            true_utility = np.asarray([float(row["mip"]) for row in ordered_candidates])
            repeated_state = np.repeat(state[None, :], len(weights), axis=0)
            predicted_utility = utility.predict(utility_design(repeated_state, weights))
            order = np.argsort(-predicted_utility, kind="stable")
            utility_index = int(order[0])
            utility_top2 = set(weights[order[:2]].tolist())
            utility_weight = float(weights[utility_index])
            utility_mip = float(true_utility[utility_index])
            correlation = spearmanr(true_utility, predicted_utility).statistic
            fold_rank.append(0.0 if not math.isfinite(float(correlation)) else float(correlation))
            fold_pairwise.append(_pairwise_accuracy(true_utility, predicted_utility))
            fold_candidate_true.extend(true_utility.tolist())
            fold_candidate_predicted.extend(predicted_utility.tolist())
            for candidate, prediction in zip(ordered_candidates, predicted_utility):
                utility_candidate_rows.append({
                    "fold": fold,
                    "held_out_prompt": held_prompt,
                    "case_id": case,
                    "sample_id": features[case]["sample_id"],
                    "weight": float(candidate["weight"]),
                    "true_mip": float(candidate["mip"]),
                    "predicted_mip": float(prediction),
                    "selected": float(candidate["weight"]) == utility_weight,
                })

            probabilities = classifier.predict_proba(state.reshape(1, -1))[0]
            class_order = np.argsort(-probabilities, kind="stable")
            classifier_class = int(classifier.classes_[class_order[0]])
            classifier_weight = float(WEIGHTS[classifier_class])
            classifier_top2 = {float(WEIGHTS[int(classifier.classes_[index])]) for index in class_order[:2]}
            classifier_candidate = next(row for row in ordered_candidates if math.isclose(float(row["weight"]), classifier_weight, abs_tol=1e-12))
            classifier_mip = float(classifier_candidate["mip"])
            action = {
                "fold": fold,
                "held_out_prompt": held_prompt,
                "case_id": case,
                "sample_id": features[case]["sample_id"],
                "alpha_helpfulness": float(features[case]["alpha_helpfulness"]),
                "alpha_harmlessness": float(features[case]["alpha_harmlessness"]),
                "oracle_weight": oracle_weight,
                "oracle_tied_weights": oracle["oracle_tied_weights"],
                "oracle_mip": float(oracle["oracle_mip"]),
                "classifier_weight": classifier_weight,
                "classifier_policy_mip": classifier_mip,
                "classifier_oracle_action_correct": classifier_weight == oracle_weight,
                "classifier_oracle_set_correct": classifier_weight in oracle_ties,
                "classifier_top2_action_correct": oracle_weight in classifier_top2,
                "utility_weight": utility_weight,
                "utility_policy_mip": utility_mip,
                "utility_oracle_action_correct": utility_weight == oracle_weight,
                "utility_oracle_set_correct": utility_weight in oracle_ties,
                "utility_top2_action_correct": oracle_weight in utility_top2,
            }
            action_rows.append(action); fold_action_rows.append(action)

        for model_name, prefix in (("classifier_logistic", "classifier"), ("utility_ridge", "utility")):
            fold_rows.append({
                "fold": fold,
                "held_out_prompt": held_prompt,
                "model": model_name,
                "train_prompts": 11,
                "validation_prompts": 1,
                "validation_cases": 5,
                "policy_mip": fmean(row[f"{prefix}_policy_mip"] for row in fold_action_rows),
                "oracle_action_accuracy": fmean(float(row[f"{prefix}_oracle_action_correct"]) for row in fold_action_rows),
                "oracle_set_accuracy": fmean(float(row[f"{prefix}_oracle_set_correct"]) for row in fold_action_rows),
                "top2_action_accuracy": fmean(float(row[f"{prefix}_top2_action_correct"]) for row in fold_action_rows),
                "utility_mse": mean_squared_error(fold_candidate_true, fold_candidate_predicted) if prefix == "utility" else "",
                "utility_mae": mean_absolute_error(fold_candidate_true, fold_candidate_predicted) if prefix == "utility" else "",
                "mean_case_spearman": fmean(fold_rank) if prefix == "utility" else "",
                "mean_case_pairwise_accuracy": fmean(fold_pairwise) if prefix == "utility" else "",
            })

    best_global = float(baseline["best_global_mip"])
    best_per_alpha = float(baseline["best_per_alpha_mip"])
    oracle_mip = float(baseline["oracle_mip"])
    gap = oracle_mip - best_global

    def policy_summary(prefix: str) -> dict[str, Any]:
        policy_mip = fmean(row[f"{prefix}_policy_mip"] for row in action_rows)
        selected = Counter(str(row[f"{prefix}_weight"]) for row in action_rows)
        return {
            "offline_policy_mip": policy_mip,
            "gain_over_global": policy_mip - best_global,
            "oracle_gap_capture": (policy_mip - best_global) / gap,
            "oracle_action_accuracy": fmean(float(row[f"{prefix}_oracle_action_correct"]) for row in action_rows),
            "oracle_set_accuracy": fmean(float(row[f"{prefix}_oracle_set_correct"]) for row in action_rows),
            "top2_action_accuracy": fmean(float(row[f"{prefix}_top2_action_correct"]) for row in action_rows),
            "selected_weight_distribution": dict(sorted(selected.items(), key=lambda item: float(item[0]))),
            "num_distinct_selected_weights": len(selected),
            "collapsed_to_one_weight": len(selected) == 1,
        }

    classifier_summary = policy_summary("classifier")
    utility_summary = policy_summary("utility")
    true_values = np.asarray([row["true_mip"] for row in utility_candidate_rows])
    predicted_values = np.asarray([row["predicted_mip"] for row in utility_candidate_rows])
    utility_summary.update({
        "utility_mse": mean_squared_error(true_values, predicted_values),
        "utility_mae": mean_absolute_error(true_values, predicted_values),
        "mean_case_spearman": fmean(float(row["mean_case_spearman"]) for row in fold_rows if row["model"] == "utility_ridge"),
        "mean_case_pairwise_accuracy": fmean(float(row["mean_case_pairwise_accuracy"]) for row in fold_rows if row["model"] == "utility_ridge"),
    })

    if utility_summary["gain_over_global"] > 0 and utility_summary["oracle_gap_capture"] >= 0.10 and not utility_summary["collapsed_to_one_weight"]:
        verdict = "PASS"
    elif utility_summary["gain_over_global"] > 0 and not utility_summary["collapsed_to_one_weight"]:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    confusion = []
    for prefix, model_name in (("classifier", "classifier_logistic"), ("utility", "utility_ridge")):
        counts = Counter((str(row["oracle_weight"]), str(row[f"{prefix}_weight"])) for row in action_rows)
        for true_weight in WEIGHTS:
            for predicted_weight in WEIGHTS:
                confusion.append({
                    "model": model_name,
                    "oracle_weight": true_weight,
                    "predicted_weight": predicted_weight,
                    "count": counts[(str(true_weight), str(predicted_weight))],
                })

    state_all = np.stack([_state(features[case]) for case in sorted(features)])
    label_all = np.asarray([weight_to_class[float(oracles[case]["oracle_weight"])] for case in sorted(features)])
    final_classifier = Pipeline((("scale", StandardScaler()), ("model", LogisticRegression(C=CLASSIFIER_C, class_weight="balanced", max_iter=5000, random_state=SEED))))
    final_classifier.fit(state_all, label_all)
    utility_states=[]; utility_weights=[]; utility_targets=[]
    for case in sorted(features):
        for candidate in candidates[case]:
            utility_states.append(_state(features[case])); utility_weights.append(float(candidate["weight"])); utility_targets.append(float(candidate["mip"]))
    final_utility = Pipeline((("scale", StandardScaler()), ("model", Ridge(alpha=UTILITY_RIDGE_ALPHA))))
    final_utility.fit(utility_design(np.stack(utility_states), np.asarray(utility_weights)), np.asarray(utility_targets))
    _atomic_joblib(OUTPUT / "models/classifier_logistic.joblib", final_classifier)
    _atomic_joblib(OUTPUT / "models/utility_ridge.joblib", final_utility)

    configs = {
        "grouping": "leave-one-unique-prompt-out (12 folds)",
        "random_row_split": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "inference_features_exclude_scorer_values": True,
        "candidate_weights": list(WEIGHTS),
        "classifier": {"type": "StandardScaler + LogisticRegression", "C": CLASSIFIER_C, "class_weight": "balanced", "max_iter": 5000},
        "utility_predictor": {"type": "StandardScaler + Ridge", "alpha": UTILITY_RIDGE_ALPHA, "design": "state,w,w^2,state*w,state*w^2"},
        "seed": SEED,
        "verdict_rule": "PASS iff utility gain > 0, gap capture >= 10%, and selected weights > 1; WEAK iff gain > 0 and non-collapsed; otherwise FAIL",
    }
    summary = {
        "status": "COMPLETE",
        "verdict": verdict,
        "num_prompts": len(prompts),
        "num_cases": len(features),
        "num_folds": len(prompts),
        "split": "leave-one-prompt-out",
        "prompt_leakage": False,
        "global_best_fixed_mip": best_global,
        "per_alpha_fixed_mip": best_per_alpha,
        "oracle_mip": oracle_mip,
        "oracle_gap": gap,
        "classifier": classifier_summary,
        "utility_predictor": utility_summary,
        "preferred_model": "utility_ridge",
        "limitations": ["12 unique prompts", "offline candidate-policy evaluation", "no statistical significance claim", "architecture/config selected a priori without random row split"],
    }
    _atomic_json(OUTPUT / "model_configs.json", configs)
    _atomic_csv(OUTPUT / "fold_results.csv", fold_rows)
    _atomic_csv(OUTPUT / "predictions.csv", action_rows)
    _atomic_csv(OUTPUT / "utility_predictions.csv", utility_candidate_rows)
    _atomic_csv(OUTPUT / "weight_confusion.csv", confusion)
    _atomic_json(OUTPUT / "summary.json", summary)
    _atomic_text(OUTPUT / "report.md", render_report(summary))
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    utility = summary["utility_predictor"]
    classifier = summary["classifier"]
    return f"""# Prompt-level Adaptive PARM router

Verdict: **{summary['verdict']}**

## Leakage boundary

All features are computed from the raw prompt and requested alpha before the
first generated token. Reward/cost scorer values are labels only and never
inference features. Evaluation is leave-one-unique-prompt-out: all five alpha
cases for a held-out prompt stay in the same fold. No random row split is used.

## Offline policy result

| Policy | MIP | Gain vs global | Oracle-gap capture | Distinct weights |
|---|---:|---:|---:|---:|
| Global fixed | {summary['global_best_fixed_mip']:.9f} | 0 | 0% | 1 |
| Per-alpha fixed | {summary['per_alpha_fixed_mip']:.9f} | {summary['per_alpha_fixed_mip']-summary['global_best_fixed_mip']:.9f} | {(summary['per_alpha_fixed_mip']-summary['global_best_fixed_mip'])/summary['oracle_gap']:.2%} | — |
| Logistic classifier | {classifier['offline_policy_mip']:.9f} | {classifier['gain_over_global']:.9f} | {classifier['oracle_gap_capture']:.2%} | {classifier['num_distinct_selected_weights']} |
| Ridge utility predictor | {utility['offline_policy_mip']:.9f} | {utility['gain_over_global']:.9f} | {utility['oracle_gap_capture']:.2%} | {utility['num_distinct_selected_weights']} |
| Oracle | {summary['oracle_mip']:.9f} | {summary['oracle_gap']:.9f} | 100% | — |

## Utility prediction diagnostics

- MSE: `{utility['utility_mse']:.9g}`
- MAE: `{utility['utility_mae']:.9g}`
- Mean within-case Spearman: `{utility['mean_case_spearman']:.6f}`
- Mean within-case pairwise ranking accuracy: `{utility['mean_case_pairwise_accuracy']:.6f}`
- Oracle action accuracy: `{utility['oracle_action_accuracy']:.2%}`
- Oracle-set accuracy (tie-aware): `{utility['oracle_set_accuracy']:.2%}`
- Top-2 action accuracy: `{utility['top2_action_accuracy']:.2%}`

## Limitations

Only 12 prompt groups are available. These are offline candidate-policy results,
not statistical or online generation evidence. The preferred method is the
predeclared small Ridge utility predictor; no large hidden-state MLP or token
router was trained.
"""
