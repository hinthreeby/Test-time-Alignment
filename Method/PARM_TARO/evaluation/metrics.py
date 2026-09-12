"""Pareto, HV, preference matching, quality, and routing metrics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from PARM_TARO.evaluation.protocol import ObjectiveNormalizer
from router_v2.evaluation.rad.metrics import (
    coherence_proxy,
    distinct_n,
    repeated_ngram_rate,
    tokenize_words,
)


def pareto_front(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    clean = [tuple(float(value) for value in point) for point in points]
    if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in clean):
        raise ValueError("Pareto points must be finite two-objective vectors")
    result = []
    for index, point in enumerate(clean):
        dominated = any(
            other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for other_index, other in enumerate(clean)
            if other_index != index
        )
        if not dominated and point not in result:
            result.append(point)
    return sorted(result)


def hypervolume_2d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    """Exact union area for a maximization front above a shared reference."""

    if len(reference) != 2:
        raise ValueError("2-D hypervolume requires a two-value reference")
    ref_x, ref_y = (float(value) for value in reference)
    eligible = [
        point
        for point in pareto_front(points)
        if point[0] >= ref_x and point[1] >= ref_y
    ]
    area = 0.0
    best_y = ref_y
    for x, y in sorted(eligible, reverse=True):
        if y > best_y:
            area += (x - ref_x) * (y - best_y)
            best_y = y
    if not math.isfinite(area) or area < -1e-12:
        raise FloatingPointError("Invalid hypervolume")
    return max(area, 0.0)


def mean_inner_product(alpha: Sequence[float], objective: Sequence[float]) -> float:
    if len(alpha) != len(objective) or not alpha:
        raise ValueError("MIP vectors must have equal non-zero dimension")
    return float(sum(float(a) * float(r) for a, r in zip(alpha, objective)))


def preference_cosine_similarity(
    alpha: Sequence[float], objective: Sequence[float]
) -> float:
    if len(alpha) != len(objective) or not alpha:
        raise ValueError("PCS vectors must have equal non-zero dimension")
    left = [float(value) for value in alpha]
    right = [float(value) for value in objective]
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0.0:
        return 0.0
    return float(sum(a * r for a, r in zip(left, right)) / denominator)


def attach_normalized_objectives(
    records: Sequence[Mapping[str, Any]], normalizer: ObjectiveNormalizer
) -> list[dict[str, Any]]:
    output = []
    for row in records:
        helpfulness, harmlessness = normalizer.transform(
            float(row["helpfulness_raw"]), float(row["harmlessness_raw"])
        )
        alpha = [float(value) for value in row["requested_alpha"]]
        objective = [helpfulness, harmlessness]
        text = str(row["generated_text"])
        tokens = tokenize_words(text)
        output.append(
            {
                **dict(row),
                "helpfulness_normalized": helpfulness,
                "harmlessness_normalized": harmlessness,
                "mip": mean_inner_product(alpha, objective),
                "pcs": preference_cosine_similarity(alpha, objective),
                "coherence": coherence_proxy(str(row["prompt"]), text),
                "distinct_1": distinct_n(text, 1),
                "distinct_2": distinct_n(text, 2),
                "distinct_3": distinct_n(text, 3),
                "repeated_4gram_rate": repeated_ngram_rate(tokens, 4),
                "generation_length": len(row["selected_token_ids"]),
                "latency_per_token": float(row["latency_seconds"])
                / max(len(row["selected_token_ids"]), 1),
                "perplexity": float(row["base_conditional_perplexity"]),
            }
        )
    return output


def attach_shared_preference_regret(
    records: Sequence[Mapping[str, Any]], *, required_methods: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[float, ...], int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row["prompt_id"]),
            tuple(float(value) for value in row["requested_alpha"]),
            int(row["seed"]),
        )
        groups[key].append(row)
    expected = set(required_methods)
    oracle: dict[tuple[str, tuple[float, ...], int], float] = {}
    for key, rows in groups.items():
        methods = [str(row["method"]) for row in rows]
        if len(methods) != len(expected) or set(methods) != expected:
            raise ValueError(f"Incomplete/duplicate shared regret pool for {key}")
        oracle[key] = max(float(row["mip"]) for row in rows)
    output = []
    for row in records:
        key = (
            str(row["prompt_id"]),
            tuple(float(value) for value in row["requested_alpha"]),
            int(row["seed"]),
        )
        regret = oracle[key] - float(row["mip"])
        output.append({**dict(row), "preference_regret": max(0.0, regret)})
    return output


def method_alpha_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[float, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["method"]), tuple(row["requested_alpha"]))].append(row)
    points = []
    for (method, alpha), rows in sorted(groups.items()):
        points.append(
            {
                "method": method,
                "requested_alpha": list(alpha),
                "n": len(rows),
                "helpfulness_normalized": statistics.fmean(
                    float(row["helpfulness_normalized"]) for row in rows
                ),
                "harmlessness_normalized": statistics.fmean(
                    float(row["harmlessness_normalized"]) for row in rows
                ),
                "helpfulness_raw": statistics.fmean(float(row["helpfulness_raw"]) for row in rows),
                "harmlessness_raw": statistics.fmean(float(row["harmlessness_raw"]) for row in rows),
            }
        )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_method[point["method"]].append(point)
    for method_points in by_method.values():
        front = set(
            pareto_front(
                [
                    (point["helpfulness_normalized"], point["harmlessness_normalized"])
                    for point in method_points
                ]
            )
        )
        for point in method_points:
            point["pareto"] = (
                point["helpfulness_normalized"], point["harmlessness_normalized"]
            ) in front
    return points


def aggregate_methods(
    records: Sequence[Mapping[str, Any]], reference: Sequence[float]
) -> list[dict[str, Any]]:
    points = method_alpha_points(records)
    point_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        point_groups[str(point["method"])].append(point)
    for row in records:
        row_groups[str(row["method"])].append(row)
    output = []
    for method, rows in sorted(row_groups.items()):
        lambda_values = [float(value) for row in rows for value in row["lambda_history"]]
        method_points = point_groups[method]
        output.append(
            {
                "method": method,
                "records": len(rows),
                "hypervolume": hypervolume_2d(
                    [
                        (point["helpfulness_normalized"], point["harmlessness_normalized"])
                        for point in method_points
                    ],
                    reference,
                ),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in rows)
                    for metric in (
                        "mip", "pcs", "preference_regret", "perplexity", "coherence",
                        "latency_per_token", "generation_length", "distinct_1", "distinct_2",
                        "distinct_3", "repeated_4gram_rate",
                    )
                },
                "lambda": {
                    "count": len(lambda_values),
                    "mean": statistics.fmean(lambda_values),
                    "std": statistics.pstdev(lambda_values) if len(lambda_values) > 1 else 0.0,
                    "min": min(lambda_values),
                    "max": max(lambda_values),
                },
            }
        )
    return output
