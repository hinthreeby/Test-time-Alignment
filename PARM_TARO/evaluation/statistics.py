"""Prompt-clustered paired inference for Stage 10 primary comparisons."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


LOWER_IS_BETTER = frozenset({"preference_regret", "perplexity"})


def paired_effect_size(differences: Sequence[float]) -> float:
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("Effect size requires paired differences")
    if len(values) == 1:
        return 0.0
    standard_deviation = statistics.stdev(values)
    return 0.0 if standard_deviation == 0.0 else statistics.fmean(values) / standard_deviation


def paired_bootstrap_ci(
    differences_by_prompt: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    prompts = sorted(differences_by_prompt)
    if not prompts:
        raise ValueError("Bootstrap requires prompt clusters")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [prompts[generator.randrange(len(prompts))] for _ in prompts]
        values = [value for prompt in selected for value in differences_by_prompt[prompt]]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def paired_permutation_pvalue(
    differences_by_prompt: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> float:
    prompt_means = [statistics.fmean(differences_by_prompt[key]) for key in sorted(differences_by_prompt)]
    if not prompt_means:
        raise ValueError("Permutation test requires prompt clusters")
    observed = abs(statistics.fmean(prompt_means))
    generator = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        permuted = statistics.fmean(
            value if generator.getrandbits(1) else -value for value in prompt_means
        )
        exceed += abs(permuted) >= observed
    return (exceed + 1.0) / (samples + 1.0)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    if any(not 0.0 <= float(value) <= 1.0 for value in pvalues):
        raise ValueError("P-values must be in [0, 1]")
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_metric_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    comparator: str,
    metric: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_key: dict[tuple[str, tuple[float, ...], int, str], float] = {}
    for row in records:
        method = str(row["method"])
        if method not in {treatment, comparator}:
            continue
        key = (
            str(row["prompt_id"]), tuple(row["requested_alpha"]), int(row["seed"]), method
        )
        if key in by_key:
            raise ValueError(f"Duplicate statistical pair key: {key}")
        by_key[key] = float(row[metric])
    prompts = sorted({key[0] for key in by_key})
    differences: dict[str, list[float]] = defaultdict(list)
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    for prompt in prompts:
        task_keys = sorted(
            {(alpha, seed) for pid, alpha, seed, _ in by_key if pid == prompt}
        )
        for alpha, item_seed in task_keys:
            left_key = (prompt, alpha, item_seed, treatment)
            right_key = (prompt, alpha, item_seed, comparator)
            if left_key not in by_key or right_key not in by_key:
                raise ValueError(f"Missing paired record for {prompt}/{alpha}/{item_seed}")
            differences[prompt].append(sign * (by_key[left_key] - by_key[right_key]))
    flat = [value for values in differences.values() for value in values]
    low, high = paired_bootstrap_ci(differences, samples=bootstrap_samples, seed=seed)
    return {
        "treatment": treatment,
        "comparator": comparator,
        "metric": metric,
        "direction": "positive_favors_treatment",
        "paired_tasks": len(flat),
        "prompt_clusters": len(differences),
        "mean_effect": statistics.fmean(flat),
        "ci95_low": low,
        "ci95_high": high,
        "cohen_dz": paired_effect_size(flat),
        "permutation_pvalue": paired_permutation_pvalue(
            differences, samples=permutation_samples, seed=seed
        ),
    }


def correct_multiple_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    adjusted = holm_adjust([float(row["permutation_pvalue"]) for row in rows])
    return [
        {**dict(row), "holm_adjusted_pvalue": value, "significant_at_0_05": value < 0.05}
        for row, value in zip(rows, adjusted)
    ]


def clustered_hv_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    comparator: str,
    hv_function: Callable[[Sequence[Mapping[str, Any]], str], float],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired prompt bootstrap and label-swap permutation for non-additive HV."""

    relevant = [row for row in records if row["method"] in {treatment, comparator}]
    prompts = sorted({str(row["prompt_id"]) for row in relevant})
    if not prompts:
        raise ValueError("HV inference requires paired prompts")
    by_prompt = {prompt: [row for row in relevant if row["prompt_id"] == prompt] for prompt in prompts}
    observed = hv_function(relevant, treatment) - hv_function(relevant, comparator)
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        sampled = [prompts[generator.randrange(len(prompts))] for _ in prompts]
        rows = []
        for replicate, prompt in enumerate(sampled):
            rows.extend({**dict(row), "prompt_id": f"{replicate}:{prompt}"} for row in by_prompt[prompt])
        bootstrap.append(hv_function(rows, treatment) - hv_function(rows, comparator))
    bootstrap.sort()
    exceed = 0
    for _ in range(permutation_samples):
        swapped = []
        for prompt in prompts:
            swap = bool(generator.getrandbits(1))
            for row in by_prompt[prompt]:
                method = str(row["method"])
                if swap:
                    method = comparator if method == treatment else treatment
                swapped.append({**dict(row), "method": method})
        difference = hv_function(swapped, treatment) - hv_function(swapped, comparator)
        exceed += abs(difference) >= abs(observed)
    return {
        "treatment": treatment,
        "comparator": comparator,
        "metric": "hypervolume",
        "direction": "positive_favors_treatment",
        "paired_tasks": len(relevant) // 2,
        "prompt_clusters": len(prompts),
        "mean_effect": observed,
        "ci95_low": bootstrap[int(0.025 * (bootstrap_samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (bootstrap_samples - 1))],
        "cohen_dz": None,
        "permutation_pvalue": (exceed + 1.0) / (permutation_samples + 1.0),
    }
