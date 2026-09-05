"""Stage 6 quality, routing, aggregation, and Pareto metrics."""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


METRIC_NAMES = (
    "alignment_success",
    "target_probability",
    "perplexity",
    "ppl_degradation",
    "coherence",
    "repeated_unigram_rate",
    "repeated_bigram_rate",
    "repeated_trigram_rate",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "generation_length",
    "latency_per_token",
    "total_latency",
)


def tokenize_words(text: str) -> list[str]:
    cleaned = "".join(
        character.lower()
        if character.isalnum() or character.isspace()
        else " "
        for character in text
    )
    return [token for token in cleaned.split() if token]


def ngrams(tokens: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("n-gram size must be positive")
    return [
        tuple(tokens[index : index + size])
        for index in range(max(len(tokens) - size + 1, 0))
    ]


def repeated_ngram_rate(tokens: Sequence[str], size: int) -> float:
    values = ngrams(tokens, size)
    return 0.0 if not values else 1.0 - len(set(values)) / len(values)


def distinct_n(text: str, size: int) -> float:
    values = ngrams(tokenize_words(text), size)
    return 0.0 if not values else len(set(values)) / len(values)


def coherence_proxy(prompt: str, generated_text: str) -> float:
    prompt_tokens = set(tokenize_words(prompt))
    generated_tokens = set(tokenize_words(generated_text))
    if not generated_tokens:
        return 0.0
    overlap = len(prompt_tokens & generated_tokens) / len(generated_tokens)
    length_term = min(len(generated_tokens) / 8.0, 1.0)
    return float(0.5 * overlap + 0.5 * length_term)


def metric_record(raw: dict[str, Any]) -> dict[str, Any]:
    text = str(raw.get("generated_text", ""))
    prompt = str(raw.get("prompt", ""))
    tokens = tokenize_words(text)
    latency = dict(raw.get("latency", {}))
    perplexity = float(raw["perplexity"])
    if not math.isfinite(perplexity) or perplexity <= 0.0:
        raise ValueError("Conditional perplexity must be finite and positive")
    return {
        **raw,
        "alignment_success": int(raw["classifier_sentiment_success"]),
        "target_probability": float(raw["classifier_target_probability"]),
        "perplexity": perplexity,
        "coherence": coherence_proxy(prompt, text),
        "repeated_unigram_rate": repeated_ngram_rate(tokens, 1),
        "repeated_bigram_rate": repeated_ngram_rate(tokens, 2),
        "repeated_trigram_rate": repeated_ngram_rate(tokens, 3),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "generation_length": len(raw.get("selected_token_ids", [])),
        "latency_per_token": float(latency["average_latency_per_token"]),
        "total_latency": float(latency["total_generation_time"]),
    }


def _linear_percentile(values: Sequence[int], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Percentile requires values and a quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def generation_length_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    minimum_required_budget: int = 32,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Generation length diagnostics require records")
    if max_new_tokens <= 0 or minimum_required_budget <= 0:
        raise ValueError("Generation token budgets must be positive")

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_length_mismatches: list[dict[str, Any]] = []
    invalid_nonpositive: list[dict[str, Any]] = []
    invalid_over_budget: list[dict[str, Any]] = []
    invalid_short_without_eos: list[dict[str, Any]] = []
    short_records: list[dict[str, Any]] = []

    for row in records:
        method = str(row["method"])
        length = int(row["generation_length"])
        selected_token_ids = list(row.get("selected_token_ids", []))
        summary = {
            "prompt_id": str(row["prompt_id"]),
            "source_prompt_id": str(row.get("source_prompt_id", "")),
            "method": method,
            "method_family": str(row["method_family"]),
            "prompt_sentiment_class": str(row["prompt_sentiment_class"]),
            "generation_length": length,
            "terminated_on_eos": bool(row.get("terminated_on_eos", False)),
            "terminal_token_id": (
                int(selected_token_ids[-1]) if selected_token_ids else None
            ),
            "source": str(row.get("source", "")),
        }
        by_method[method].append(summary)
        if length != len(selected_token_ids):
            invalid_length_mismatches.append(summary)
        if length <= 0:
            invalid_nonpositive.append(summary)
        if length > max_new_tokens:
            invalid_over_budget.append(summary)
        if length < max_new_tokens:
            short_records.append(summary)
            if not summary["terminated_on_eos"]:
                invalid_short_without_eos.append(summary)

    method_statistics = []
    short_by_method_class = []
    for method in sorted(by_method):
        method_rows = by_method[method]
        lengths = [int(row["generation_length"]) for row in method_rows]
        short_rows = [
            row
            for row in method_rows
            if int(row["generation_length"]) < max_new_tokens
        ]
        method_statistics.append(
            {
                "method": method,
                "method_family": str(method_rows[0]["method_family"]),
                "count": len(lengths),
                "short_count": len(short_rows),
                "min": min(lengths),
                "p05": _linear_percentile(lengths, 0.05),
                "median": float(statistics.median(lengths)),
                "mean": float(statistics.fmean(lengths)),
                "max": max(lengths),
            }
        )
        classes = sorted(
            {str(row["prompt_sentiment_class"]) for row in method_rows}
        )
        for prompt_class in classes:
            short_by_method_class.append(
                {
                    "method": method,
                    "prompt_sentiment_class": prompt_class,
                    "short_count": sum(
                        row["prompt_sentiment_class"] == prompt_class
                        for row in short_rows
                    ),
                }
            )

    checks = {
        "max_new_tokens_at_least_required_budget": (
            max_new_tokens >= minimum_required_budget
        ),
        "all_lengths_match_selected_token_ids": not invalid_length_mismatches,
        "all_lengths_positive": not invalid_nonpositive,
        "no_length_exceeds_max_new_tokens": not invalid_over_budget,
        "all_short_outputs_terminated_on_eos": not invalid_short_without_eos,
    }
    return {
        "max_new_tokens": max_new_tokens,
        "minimum_required_budget": minimum_required_budget,
        "all_realized_lengths_at_least_required_budget": all(
            int(row["generation_length"]) >= minimum_required_budget
            for row in records
        ),
        "short_record_count": len(short_records),
        "method_statistics": method_statistics,
        "short_by_method_class": short_by_method_class,
        "short_records": short_records,
        "invalid_records": {
            "length_mismatch": invalid_length_mismatches,
            "nonpositive": invalid_nonpositive,
            "over_budget": invalid_over_budget,
            "short_without_eos": invalid_short_without_eos,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def attach_ppl_degradation(
    records: Sequence[dict[str, Any]],
    *,
    family_base_methods: dict[str, str],
) -> list[dict[str, Any]]:
    base_values = {
        (str(row["method_family"]), str(row["prompt_id"]), int(row["seed"])): float(
            row["perplexity"]
        )
        for row in records
        if str(row["method"])
        == family_base_methods.get(str(row["method_family"]))
    }
    outputs = []
    for row in records:
        key = (
            str(row["method_family"]),
            str(row["prompt_id"]),
            int(row["seed"]),
        )
        if key not in base_values:
            raise ValueError(f"Missing family base PPL for {key}")
        base = base_values[key]
        degradation = (float(row["perplexity"]) - base) / base
        outputs.append({**row, "ppl_degradation": degradation})
    return outputs


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        raise ValueError("Bootstrap input contains no finite values")
    generator = random.Random(seed)
    means = sorted(
        _mean([clean[generator.randrange(len(clean))] for _ in clean])
        for _ in range(samples)
    )
    return (
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    )


def aggregate_metrics(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        classes = (str(row["prompt_sentiment_class"]), "all")
        for prompt_class in classes:
            groups[
                (
                    prompt_class,
                    str(row["method"]),
                    str(row["method_family"]),
                    str(row["ppl_model_family"]),
                )
            ].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        prompt_class, method, method_family, ppl_model_family = key
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite aggregate input for {method}/{metric}")
            low, high = bootstrap_ci(
                values,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            )
            output.append(
                {
                    "prompt_sentiment_class": prompt_class,
                    "method": method,
                    "method_family": method_family,
                    "ppl_model_family": ppl_model_family,
                    "metric": metric,
                    "mean": _mean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(values),
                }
            )
    return output


def select_by_alignment(
    records: Sequence[dict[str, Any]],
    methods: Iterable[str],
) -> str:
    allowed = set(methods)
    candidates = []
    for method in sorted(allowed):
        rows = [
            row
            for row in records
            if row["method"] == method
            and row["prompt_sentiment_class"] == "negative"
        ]
        if not rows:
            raise ValueError(f"No negative validation rows for {method}")
        candidates.append(
            (
                _mean([float(row["alignment_success"]) for row in rows]),
                _mean([float(row["target_probability"]) for row in rows]),
                -_mean([float(row["perplexity"]) for row in rows]),
                method,
            )
        )
    return max(candidates)[-1]


def mean_routing_strength(
    records: Sequence[dict[str, Any]],
    method: str,
) -> float:
    values = [
        float(value)
        for row in records
        if row["method"] == method
        for value in row.get("lambda_history", [])
    ]
    if not values:
        raise ValueError(f"No lambda values for method {method}")
    result = _mean(values)
    if not 0.0 <= result <= 1.0:
        raise ValueError("Mean routing strength is outside [0, 1]")
    return result


def lambda_diagnostics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    by_position: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in records:
        method = str(row["method"])
        for position, value in enumerate(row.get("lambda_history", [])):
            grouped[method].append(float(value))
            by_position[(method, position)].append(float(value))
    output = []
    for method, values in sorted(grouped.items()):
        tensor_values = sorted(values)
        quantile = lambda fraction: tensor_values[  # noqa: E731
            int(fraction * (len(tensor_values) - 1))
        ]
        output.append(
            {
                "method": method,
                "position": "all",
                "count": len(values),
                "mean": _mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "p05": quantile(0.05),
                "p50": quantile(0.5),
                "p95": quantile(0.95),
                "fraction_near_zero": _mean([value <= 0.05 for value in values]),
                "fraction_near_max": _mean([value >= 0.95 for value in values]),
            }
        )
    for (method, position), values in sorted(by_position.items()):
        output.append(
            {
                "method": method,
                "position": position,
                "count": len(values),
                "mean": _mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "p05": None,
                "p50": None,
                "p95": None,
                "fraction_near_zero": _mean([value <= 0.05 for value in values]),
                "fraction_near_max": _mean([value >= 0.95 for value in values]),
            }
        )
    return output


def _is_pareto(point: dict[str, Any], points: Sequence[dict[str, Any]]) -> bool:
    for other in points:
        if other is point:
            continue
        if (
            float(other["alignment"]) >= float(point["alignment"])
            and float(other["ppl_degradation"])
            <= float(point["ppl_degradation"])
            and (
                float(other["alignment"]) > float(point["alignment"])
                or float(other["ppl_degradation"])
                < float(point["ppl_degradation"])
            )
        ):
            return False
    return True


def pareto_points(aggregate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = defaultdict(dict)
    metadata: dict[str, dict[str, str]] = {}
    for row in aggregate_rows:
        if row["prompt_sentiment_class"] != "negative":
            continue
        method = str(row["method"])
        by_method[method][str(row["metric"])] = float(row["mean"])
        metadata[method] = {
            "method_family": str(row["method_family"]),
            "ppl_model_family": str(row["ppl_model_family"]),
        }
    points = [
        {
            "method": method,
            **metadata[method],
            "alignment": values["alignment_success"],
            "target_probability": values["target_probability"],
            "ppl_degradation": values["ppl_degradation"],
            "perplexity": values["perplexity"],
            "coherence": values["coherence"],
            "latency_per_token": values["latency_per_token"],
        }
        for method, values in sorted(by_method.items())
    ]
    for point in points:
        point["pareto_frontier_alignment_vs_ppl_degradation"] = int(
            _is_pareto(point, points)
        )
    return points


def write_csv_atomic(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {output}")
    fields = sorted({key for row in rows for key in row})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
