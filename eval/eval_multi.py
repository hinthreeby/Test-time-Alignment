#!/usr/bin/env python3
"""Paper-oriented multi-objective evaluation for DynaCAP-PARM.

Core metrics:
1. Hypervolume (HV): convergence and diversity of the Pareto front.
2. Mean Inner Product (MIP): agreement between alpha and objective values.
3. Per-objective mean, global worst-objective, and mean sample minimum.
4. Pareto-front size/ratio and preference-control Spearman correlation.
5. Efficiency: latency, tokens/s, PARM call rate, processed/replayed tokens.
6. Dynamic-control diagnostics: alpha KL/shift/switches and mean w.

The script never normalizes each method independently. Objective values must:
- already be in [0,1] in normalized_rewards/objective_scores; or
- use fixed train/validation bounds supplied by --bounds-file; or
- be probabilities produced by optional evaluator models.

Example using pre-scored outputs:

    python eval_multiobjective_sota.py \
      --method PARM=results/parm/scored_records.jsonl \
      --method DynaCAP=results/dynacap/scored_records.jsonl \
      --output-dir results/sota_eval

Example scoring raw outputs:

    python eval_multiobjective_sota.py \
      --method PARM=results/parm/generation_records.jsonl \
      --method DynaCAP=results/dynacap/generation_records.jsonl \
      --helpfulness-model models/helpfulness-deberta-v3-large-v2 \
      --toxicity-model models/toxic-bert \
      --output-dir results/sota_eval

For more than two objectives, install pymoo for exact HV:

    pip install pymoo
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_OBJECTIVES = ("helpfulness", "harmlessness")
EPS = 1e-12


@dataclass
class ScoredRecord:
    method: str
    source_path: str
    sample_id: str
    prompt: str
    response: str
    alpha: tuple[float, ...] | None
    raw_objectives: tuple[float, ...]
    objectives: tuple[float, ...]
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-objective alignment with HV, MIP, Pareto, and compute metrics."
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Method name and JSON/JSONL file. Repeat for every method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(DEFAULT_OBJECTIVES),
        help="Ordered objective names matching alpha.",
    )
    parser.add_argument(
        "--bounds-file",
        type=Path,
        help="JSON mapping objective to fixed {min,max} bounds from train/validation.",
    )
    parser.add_argument(
        "--reference-point",
        type=float,
        nargs="+",
        help="Fixed maximization HV reference after normalization. Default: all zeros.",
    )
    parser.add_argument(
        "--pairing",
        choices=("intersection", "per-method"),
        default="intersection",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--helpfulness-model",
        help="Local classifier/reward model used if helpfulness is absent.",
    )
    parser.add_argument(
        "--toxicity-model",
        help="Local toxicity classifier used if harmlessness is absent.",
    )
    parser.add_argument("--helpfulness-label")
    parser.add_argument("--toxicity-label")
    parser.add_argument(
        "--helpfulness-input",
        choices=("prompt_response", "response"),
        default="prompt_response",
    )
    parser.add_argument(
        "--toxicity-input",
        choices=("prompt_response", "response"),
        default="response",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_numbers(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        number = finite_number(value)
        if number is not None:
            output.append(number)
    return output


def mean(values: Iterable[Any]) -> float | None:
    clean = clean_numbers(values)
    return statistics.fmean(clean) if clean else None


def percentile(values: Sequence[Any], q: float) -> float | None:
    clean = sorted(clean_numbers(values))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return clean[low]
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def get_device(name: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required only when evaluator models must score missing objectives."
        ) from error
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_method_specs(specs: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method '{spec}'. Expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise ValueError(f"Empty method name in '{spec}'.")
        if name in output:
            raise ValueError(f"Duplicate method name: {name}")
        if not path.exists():
            raise FileNotFoundError(f"{name}: file not found: {path}")
        output[name] = path
    return output


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            for key in ("records", "outputs", "data", "samples"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"{path}: top level must be a list.")
        rows = value
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {error}") from error

    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status", "success") not in ("success", "completed", None):
            continue
        response = row.get("response", row.get("output", row.get("generated_text")))
        if not isinstance(response, str) or not response.strip():
            continue
        item = dict(row)
        item["prompt"] = str(row.get("prompt", row.get("instruction", ""))).strip()
        item["response"] = response.strip()
        records.append(item)
    return records


def nested_mapping(record: dict[str, Any], keys: Sequence[str]) -> dict[str, Any] | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return None


def alpha_from_record(
    record: dict[str, Any],
    objectives: Sequence[str],
) -> tuple[float, ...] | None:
    value = None
    for key in (
        "alpha_user",
        "user_preference",
        "alpha",
        "preference",
        "preference_vector",
        "weights",
    ):
        if key in record:
            value = record[key]
            break

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if isinstance(value, dict):
        aliases = {
            "helpfulness": ("helpfulness", "helpful"),
            "harmlessness": ("harmlessness", "safety", "safe"),
        }
        values = []
        for objective in objectives:
            candidates = aliases.get(objective, (objective,))
            found = next((value.get(key) for key in candidates if key in value), None)
            number = finite_number(found)
            if number is None:
                return None
            values.append(number)
    elif isinstance(value, (list, tuple)) and len(value) >= len(objectives):
        values = []
        for item in value[: len(objectives)]:
            number = finite_number(item)
            if number is None:
                return None
            values.append(number)
    else:
        # Generation scripts in this repository store the two public PARM
        # preferences as scalar columns rather than as an alpha vector.
        scalar_aliases = {
            "helpfulness": ("alpha_helpfulness", "alpha_help"),
            "harmlessness": ("alpha_harmlessness", "alpha_harm"),
        }
        values = []
        for objective in objectives:
            candidates = scalar_aliases.get(objective, ())
            found = next((record.get(key) for key in candidates if key in record), None)
            number = finite_number(found)
            if number is None:
                return None
            values.append(number)

    if any(number < 0 for number in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    return tuple(number / total for number in values)


def sample_id(record: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "md5_hash", "uid"):
        if record.get(key) is not None:
            return str(record[key])
    content = f"{record.get('prompt', '')}\n{record.get('response', '')}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def prompt_preference_key(
    record: dict[str, Any],
    objectives: Sequence[str],
) -> tuple[str, tuple[float, ...] | None]:
    prompt_id = None
    # Prefer content-stable identifiers over generation-order indices.
    for key in ("sample_id", "md5_hash", "uid", "id"):
        if record.get(key) is not None:
            prompt_id = str(record[key])
            break
    if prompt_id is None:
        prompt_id = hashlib.sha256(
            str(record.get("prompt", "")).encode("utf-8")
        ).hexdigest()
    alpha = alpha_from_record(record, objectives)
    rounded = tuple(round(value, 10) for value in alpha) if alpha else None
    return prompt_id, rounded


def restrict_to_intersection(
    records_by_method: dict[str, list[dict[str, Any]]],
    objectives: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    maps: dict[str, dict[Any, dict[str, Any]]] = {}
    for method, records in records_by_method.items():
        mapping = {}
        for record in records:
            key = prompt_preference_key(record, objectives)
            if key in mapping:
                raise ValueError(f"{method}: duplicate prompt+alpha key: {key}")
            mapping[key] = record
        maps[method] = mapping

    common: set[Any] | None = None
    for mapping in maps.values():
        common = set(mapping) if common is None else common.intersection(mapping)
    common = common or set()

    first_method = next(iter(records_by_method))
    ordered_keys = []
    seen = set()
    for record in records_by_method[first_method]:
        key = prompt_preference_key(record, objectives)
        if key in common and key not in seen:
            ordered_keys.append(key)
            seen.add(key)

    return {
        method: [mapping[key] for key in ordered_keys]
        for method, mapping in maps.items()
    }


def direct_objective_value(
    record: dict[str, Any],
    objective: str,
) -> tuple[float | None, bool]:
    """Return value and whether it is already normalized; higher is better."""
    normalized = nested_mapping(
        record,
        ("normalized_rewards", "normalized_objectives", "objective_scores"),
    )
    if normalized:
        aliases = {
            "helpfulness": ("helpfulness", "helpful"),
            "harmlessness": ("harmlessness", "safety", "safe"),
        }.get(objective, (objective,))
        for key in aliases:
            number = finite_number(normalized.get(key))
            if number is not None:
                return number, True

    raw = nested_mapping(record, ("raw_rewards", "rewards", "objectives"))
    aliases = {
        "helpfulness": (
            "helpfulness",
            "helpful",
            "helpfulness_score",
            "avg_helpfulness",
        ),
        "harmlessness": (
            "harmlessness",
            "safety",
            "safe",
            "safety_score",
        ),
    }.get(objective, (objective, f"{objective}_score"))

    if raw:
        for key in aliases:
            number = finite_number(raw.get(key))
            if number is not None:
                return number, False
        if objective == "harmlessness":
            for key in ("toxicity", "harmfulness", "toxicity_score"):
                number = finite_number(raw.get(key))
                if number is not None and 0.0 <= number <= 1.0:
                    return 1.0 - number, True

    for key in aliases:
        number = finite_number(record.get(key))
        if number is not None:
            return number, False

    if objective == "harmlessness":
        for key in ("toxicity", "toxicity_score", "harmfulness", "harmfulness_score"):
            number = finite_number(record.get(key))
            if number is not None and 0.0 <= number <= 1.0:
                return 1.0 - number, True

    return None, False


def load_bounds(path: Path | None, objectives: Sequence[str]) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for objective in objectives:
        spec = value.get(objective)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing bounds for '{objective}' in {path}.")
        low = finite_number(spec.get("min"))
        high = finite_number(spec.get("max"))
        if low is None or high is None or high <= low:
            raise ValueError(f"Invalid bounds for '{objective}': {spec}")
        output[objective] = (low, high)
    return output


def normalize_value(
    value: float,
    objective: str,
    already_normalized: bool,
    bounds: dict[str, tuple[float, float]],
) -> float:
    if already_normalized:
        if not -1e-8 <= value <= 1.0 + 1e-8:
            raise ValueError(f"{objective}: normalized value {value} is outside [0,1].")
        return min(1.0, max(0.0, value))
    if objective in bounds:
        low, high = bounds[objective]
        return min(1.0, max(0.0, (value - low) / (high - low)))
    if 0.0 <= value <= 1.0:
        return value
    raise ValueError(
        f"{objective}: raw value {value} is outside [0,1]. "
        "Supply fixed validation bounds with --bounds-file."
    )


def resolve_label_id(model: Any, requested: str | None, semantic: str) -> int | None:
    num_labels = int(getattr(model.config, "num_labels", 1))
    if num_labels == 1:
        return None

    id2label = {
        int(key): str(value)
        for key, value in dict(getattr(model.config, "id2label", {}) or {}).items()
    }
    if requested is not None:
        try:
            label_id = int(requested)
        except ValueError:
            matches = [
                index
                for index, label in id2label.items()
                if label.lower() == requested.lower()
            ]
            if len(matches) != 1:
                raise ValueError(f"Cannot resolve label '{requested}': {id2label}")
            label_id = matches[0]
        if not 0 <= label_id < num_labels:
            raise ValueError(f"Label ID {label_id} is outside [0,{num_labels}).")
        return label_id

    keywords = {
        "helpfulness": ("helpful", "positive", "preferred", "label_1"),
        "toxicity": ("toxic", "unsafe", "harmful", "label_1"),
    }[semantic]
    matches = [
        index
        for index, label in id2label.items()
        if any(keyword in label.lower() for keyword in keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    if num_labels == 2:
        return 1
    raise ValueError(
        f"Cannot auto-detect {semantic} label for id2label={id2label}. "
        f"Pass --{semantic}-label."
    )


def make_model_inputs(
    records: Sequence[dict[str, Any]],
    mode: str,
) -> list[str]:
    if mode == "response":
        return [record["response"] for record in records]
    return [
        f"{record.get('prompt', '')}\n\n{record['response']}".strip()
        for record in records
    ]


def evaluate_probability_model(
    records_by_method: dict[str, list[dict[str, Any]]],
    model_name: str,
    semantic: str,
    requested_label: str | None,
    input_mode: str,
    device: Any,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
) -> dict[str, list[float]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Scoring raw outputs requires torch and transformers. "
            "Pre-scored normalized_rewards do not require them."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    problem_type = getattr(model.config, "problem_type", None)
    is_multilabel = problem_type == "multi_label_classification"
    label_id = None if semantic == "toxicity" and is_multilabel else resolve_label_id(
        model, requested_label, semantic
    )

    slices = {}
    all_texts = []
    cursor = 0
    for method, records in records_by_method.items():
        texts = make_model_inputs(records, input_mode)
        slices[method] = (cursor, cursor + len(texts))
        all_texts.extend(texts)
        cursor += len(texts)

    scores = []
    with torch.inference_mode():
        for start in progress(
            range(0, len(all_texts), batch_size),
            desc=f"Scoring {semantic}",
        ):
            batch = all_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float().cpu()
            if semantic == "toxicity" and is_multilabel:
                # unitary/toxic-bert exposes independent toxic, threat,
                # insult, obscene, etc. probabilities.  Match the repository's
                # established evaluator by using the worst toxicity dimension.
                probabilities = torch.sigmoid(logits).max(dim=-1).values
            elif logits.ndim == 1:
                probabilities = torch.sigmoid(logits)
            elif logits.shape[-1] == 1:
                probabilities = torch.sigmoid(logits[:, 0])
            else:
                probabilities = torch.softmax(logits, dim=-1)[:, int(label_id)]
            scores.extend(float(value) for value in probabilities.tolist())

    output = {
        method: scores[left:right]
        for method, (left, right) in slices.items()
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def has_missing_objective(
    records_by_method: dict[str, list[dict[str, Any]]],
    objective: str,
) -> bool:
    return any(
        direct_objective_value(record, objective)[0] is None
        for records in records_by_method.values()
        for record in records
    )


def build_scored_records(
    records_by_method: dict[str, list[dict[str, Any]]],
    source_paths: dict[str, Path],
    objectives: Sequence[str],
    bounds: dict[str, tuple[float, float]],
    model_scores: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, list[ScoredRecord]], list[str]]:
    warnings = []
    output: dict[str, list[ScoredRecord]] = {}
    for method, records in records_by_method.items():
        scored = []
        for index, record in enumerate(records):
            raw_values = []
            normalized_values = []
            missing = False
            for objective in objectives:
                value, normalized = direct_objective_value(record, objective)
                if value is None and objective in model_scores:
                    value = model_scores[objective][method][index]
                    normalized = True
                    if objective == "harmlessness":
                        value = 1.0 - value
                if value is None:
                    warnings.append(
                        f"{method}/{sample_id(record)} missing objective '{objective}'"
                    )
                    missing = True
                    break
                raw_values.append(float(value))
                normalized_values.append(
                    normalize_value(float(value), objective, normalized, bounds)
                )

            alpha = alpha_from_record(record, objectives)
            if alpha is None:
                warnings.append(
                    f"{method}/{sample_id(record)} missing alpha; excluded from HV/MIP"
                )
            if not missing:
                scored.append(
                    ScoredRecord(
                        method=method,
                        source_path=str(source_paths[method]),
                        sample_id=sample_id(record),
                        prompt=record.get("prompt", ""),
                        response=record["response"],
                        alpha=alpha,
                        raw_objectives=tuple(raw_values),
                        objectives=tuple(normalized_values),
                        record=record,
                    )
                )
        output[method] = scored
    return output, warnings


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def pareto_front(points: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    unique = sorted(set(tuple(float(value) for value in point) for point in points))
    return [
        point
        for point in unique
        if not any(dominates(other, point) for other in unique if other != point)
    ]


def hypervolume_2d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    ref_x, ref_y = map(float, reference)
    eligible = [
        (float(point[0]), float(point[1]))
        for point in points
        if point[0] >= ref_x and point[1] >= ref_y
    ]
    front = sorted(pareto_front(eligible))
    area = 0.0
    previous_x = ref_x
    for x, y in front:
        area += max(0.0, x - previous_x) * max(0.0, y - ref_y)
        previous_x = max(previous_x, x)
    return area


def hypervolume(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> float:
    if not points:
        return 0.0
    if len(reference) == 2:
        return hypervolume_2d(points, reference)
    try:
        import numpy as np
        from pymoo.indicators.hv import HV
    except ImportError as error:
        raise RuntimeError(
            "Exact HV for K>2 requires pymoo: pip install pymoo"
        ) from error
    point_array = np.asarray(points, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    eligible = point_array[(point_array >= reference_array).all(axis=1)]
    if len(eligible) == 0:
        return 0.0
    return float(HV(ref_point=-reference_array)(-eligible))


def ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            output[indexed[position][0]] = average_rank
        cursor = end
    return output


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale <= EPS or right_scale <= EPS:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def group_preference_points(
    records: Sequence[ScoredRecord],
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, ...], list[ScoredRecord]] = defaultdict(list)
    for record in records:
        if record.alpha is not None:
            groups[tuple(round(value, 10) for value in record.alpha)].append(record)

    rows = []
    for alpha in sorted(groups):
        group = groups[alpha]
        point = tuple(
            statistics.fmean(record.objectives[index] for record in group)
            for index in range(len(alpha))
        )
        rows.append(
            {
                "alpha": alpha,
                "point": point,
                "samples": len(group),
                "mip": sum(a * q for a, q in zip(alpha, point)),
            }
        )
    return rows


def scalar_from_record(record: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value: Any = record
        valid = True
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                valid = False
                break
            value = value[part]
        if valid:
            number = finite_number(value)
            if number is not None:
                return number
    return None


def vector_kl(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        a * math.log((a + EPS) / (b + EPS))
        for a, b in zip(left, right)
        if a > 0
    )


def trace_diagnostics(scored: ScoredRecord) -> dict[str, float | None]:
    embedded = scored.record.get("trace", scored.record.get("token_trace"))
    trace = embedded if isinstance(embedded, list) else []
    alpha_kl = []
    alpha_shift = []
    w_values = []
    intervention = []
    dominant = []
    previous_alpha = None

    if scored.alpha is not None:
        for step in trace:
            if not isinstance(step, dict):
                continue
            alpha_t = step.get("alpha_t", step.get("alpha"))
            if isinstance(alpha_t, (list, tuple)) and len(alpha_t) == len(scored.alpha):
                values = [finite_number(value) for value in alpha_t]
                if all(value is not None and value >= 0 for value in values):
                    total = sum(values)
                    if total > 0:
                        normalized = tuple(float(value) / total for value in values)
                        alpha_kl.append(vector_kl(normalized, scored.alpha))
                        dominant.append(max(range(len(normalized)), key=normalized.__getitem__))
                        if previous_alpha is not None:
                            alpha_shift.append(
                                sum(abs(a - b) for a, b in zip(normalized, previous_alpha))
                            )
                        previous_alpha = normalized

            w = finite_number(step.get("w_t", step.get("weight")))
            if w is not None:
                w_values.append(w)
                intervention.append(float(w > 0))
            elif "parm_called" in step:
                intervention.append(float(bool(step["parm_called"])))

    switches = sum(
        current != previous
        for previous, current in zip(dominant, dominant[1:])
    )
    return {
        "alpha_kl": mean(alpha_kl),
        "alpha_shift": mean(alpha_shift),
        "alpha_switches": float(switches) if dominant else None,
        "mean_w": mean(w_values),
        "trace_intervention_rate": mean(intervention),
    }


def efficiency_metrics(records: Sequence[ScoredRecord]) -> dict[str, float | None]:
    latencies = []
    tokens = []
    parm_calls = []
    call_rate_pairs = []
    parm_processed = []
    replay_tokens = []
    alpha_kl = []
    alpha_shift = []
    alpha_switches = []
    mean_w = []
    explicit_rates = []

    for scored in records:
        record = scored.record
        latency = scalar_from_record(
            record,
            ("latency_seconds", "latency", "generation_time", "elapsed_seconds"),
        )
        if latency is None:
            milliseconds = scalar_from_record(record, ("latency_ms",))
            latency = milliseconds / 1000.0 if milliseconds is not None else None
        if latency is not None:
            latencies.append(latency)

        token_count = scalar_from_record(
            record,
            ("num_new_tokens", "output_tokens", "token_count", "total_steps"),
        )
        if token_count is None and isinstance(record.get("response_token_ids"), list):
            token_count = float(len(record["response_token_ids"]))
        if token_count is not None:
            tokens.append(token_count)

        calls = scalar_from_record(
            record,
            ("parm_calls", "reward_model_calls", "guide_calls"),
        )
        steps = scalar_from_record(
            record,
            ("total_steps", "num_new_tokens", "output_tokens", "token_count"),
        )
        if calls is not None:
            parm_calls.append(calls)
        if calls is not None and steps is not None:
            call_rate_pairs.append((calls, steps))

        processed = scalar_from_record(
            record,
            ("parm_processed_tokens", "reward_model_processed_tokens"),
        )
        replayed = scalar_from_record(record, ("replay_tokens", "parm_replay_tokens"))
        if processed is not None:
            parm_processed.append(processed)
        if replayed is not None:
            replay_tokens.append(replayed)

        rate = scalar_from_record(record, ("intervention_rate", "parm_call_rate"))
        if rate is not None:
            explicit_rates.append(rate)

        diagnostics = trace_diagnostics(scored)
        for target, key in (
            (alpha_kl, "alpha_kl"),
            (alpha_shift, "alpha_shift"),
            (alpha_switches, "alpha_switches"),
            (mean_w, "mean_w"),
            (explicit_rates, "trace_intervention_rate"),
        ):
            value = diagnostics[key]
            if value is not None:
                target.append(value)

        for target, keys in (
            (alpha_kl, ("mean_alpha_kl", "avg_alpha_kl")),
            (alpha_shift, ("mean_alpha_shift", "avg_alpha_shift")),
            (alpha_switches, ("alpha_switches", "dominant_objective_switches")),
            (mean_w, ("mean_w", "avg_w")),
        ):
            value = scalar_from_record(record, keys)
            if value is not None:
                target.append(value)

    total_latency = sum(latencies)
    total_tokens = sum(tokens)
    paired_steps = sum(steps for _, steps in call_rate_pairs)
    call_rate = (
        sum(calls for calls, _ in call_rate_pairs) / paired_steps
        if call_rate_pairs and paired_steps > 0
        else mean(explicit_rates)
    )
    return {
        "Latency Mean": mean(latencies),
        "Latency P95": percentile(latencies, 0.95),
        "Tokens/s": total_tokens / total_latency if total_latency > 0 else None,
        "PARM Calls": sum(parm_calls) if parm_calls else None,
        "PARM Call Rate": call_rate,
        "PARM Processed Tokens": sum(parm_processed) if parm_processed else None,
        "Replay Tokens": sum(replay_tokens) if replay_tokens else None,
        "Mean Alpha KL": mean(alpha_kl),
        "Mean Alpha Shift": mean(alpha_shift),
        "Mean Alpha Switches": mean(alpha_switches),
        "Mean w": mean(mean_w),
    }


def core_metrics(
    records: Sequence[ScoredRecord],
    objectives: Sequence[str],
    reference: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preference_rows = group_preference_points(records)
    points = [row["point"] for row in preference_rows]
    front = pareto_front(points)

    result: dict[str, Any] = {
        "Samples": len(records),
        "Preference Points": len(preference_rows),
        "Pareto Points": len(front),
        "Pareto Ratio": len(front) / len(points) if points else None,
        "Hypervolume": hypervolume(points, reference) if points else None,
        # MIP is the sample-level expectation E[alpha_user^T q].  Averaging
        # per-alpha points equally is only equivalent for a perfectly balanced
        # alpha grid.
        "MIP": mean(
            sum(a * q for a, q in zip(record.alpha, record.objectives))
            for record in records
            if record.alpha is not None
        ),
        "Mean User Utility": mean(
            sum(a * q for a, q in zip(record.alpha, record.objectives))
            for record in records
            if record.alpha is not None
        ),
        "Mean Sample Worst-Objective": mean(min(record.objectives) for record in records),
    }

    objective_means = []
    for index, objective in enumerate(objectives):
        objective_mean = mean(record.objectives[index] for record in records)
        objective_means.append(objective_mean)
        result[f"Mean {objective}"] = objective_mean
    valid_means = [value for value in objective_means if value is not None]
    result["Global Worst-Objective"] = min(valid_means) if valid_means else None

    correlations = []
    for index, objective in enumerate(objectives):
        alpha_values = [row["alpha"][index] for row in preference_rows]
        reward_values = [row["point"][index] for row in preference_rows]
        correlation = spearman(alpha_values, reward_values)
        result[f"Preference Spearman {objective}"] = correlation
        if correlation is not None:
            correlations.append(correlation)
    result["Mean Preference Spearman"] = mean(correlations)
    result.update(efficiency_metrics(records))
    return result, preference_rows


def stratified_bootstrap(
    records: Sequence[ScoredRecord],
    objectives: Sequence[str],
    reference: Sequence[float],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | None]:
    if samples <= 0:
        return {}
    groups: dict[tuple[float, ...], list[ScoredRecord]] = defaultdict(list)
    for record in records:
        if record.alpha is not None:
            groups[tuple(round(value, 10) for value in record.alpha)].append(record)
    if not groups:
        return {}

    rng = random.Random(seed)
    hv_values = []
    mip_values = []
    utility_values = []
    for _ in progress(range(samples), desc="Bootstrap", leave=False):
        replicate = []
        for group in groups.values():
            replicate.extend(rng.choice(group) for _ in range(len(group)))
        metric, _ = core_metrics(replicate, objectives, reference)
        if metric["Hypervolume"] is not None:
            hv_values.append(metric["Hypervolume"])
        if metric["MIP"] is not None:
            mip_values.append(metric["MIP"])
        if metric["Mean User Utility"] is not None:
            utility_values.append(metric["Mean User Utility"])

    tail = (1.0 - confidence) / 2.0
    return {
        "HV CI Low": percentile(hv_values, tail),
        "HV CI High": percentile(hv_values, 1.0 - tail),
        "MIP CI Low": percentile(mip_values, tail),
        "MIP CI High": percentile(mip_values, 1.0 - tail),
        "Utility CI Low": percentile(utility_values, tail),
        "Utility CI High": percentile(utility_values, 1.0 - tail),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_scored_records(
    path: Path,
    scored_by_method: dict[str, list[ScoredRecord]],
    objectives: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for method, records in scored_by_method.items():
            for record in records:
                row = {
                    "method": method,
                    "source_path": record.source_path,
                    "sample_id": record.sample_id,
                    "prompt": record.prompt,
                    "response": record.response,
                    "alpha_user": record.alpha,
                    "raw_rewards": dict(zip(objectives, record.raw_objectives)),
                    "normalized_rewards": dict(zip(objectives, record.objectives)),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0,1).")
    if len(set(args.objectives)) != len(args.objectives):
        raise ValueError("Objective names must be unique.")

    method_paths = parse_method_specs(args.method)
    records_by_method = {
        method: load_json_records(path)
        for method, path in method_paths.items()
    }
    empty = [method for method, records in records_by_method.items() if not records]
    if empty:
        raise RuntimeError(f"No valid records for methods: {empty}")

    original_counts = {
        method: len(records)
        for method, records in records_by_method.items()
    }
    if args.pairing == "intersection":
        records_by_method = restrict_to_intersection(records_by_method, args.objectives)
        common_sizes = {method: len(rows) for method, rows in records_by_method.items()}
        if not all(common_sizes.values()):
            raise RuntimeError(
                "No common prompt+alpha records across methods. "
                f"Sizes: {common_sizes}"
            )
    if args.max_samples is not None:
        # Limit after pairing so file order cannot create different subsets for
        # different methods.
        records_by_method = {
            method: records[: args.max_samples]
            for method, records in records_by_method.items()
        }

    bounds = load_bounds(args.bounds_file, args.objectives)
    reference = (
        tuple(args.reference_point)
        if args.reference_point is not None
        else tuple(0.0 for _ in args.objectives)
    )
    if len(reference) != len(args.objectives):
        raise ValueError(
            f"Reference has {len(reference)} values for {len(args.objectives)} objectives."
        )

    model_scores: dict[str, dict[str, list[float]]] = {}
    device = None

    if "helpfulness" in args.objectives and has_missing_objective(
        records_by_method, "helpfulness"
    ):
        if not args.helpfulness_model:
            raise RuntimeError(
                "Missing helpfulness. Pass --helpfulness-model or precompute it."
            )
        if device is None:
            device = get_device(args.device)
        model_scores["helpfulness"] = evaluate_probability_model(
            records_by_method,
            args.helpfulness_model,
            "helpfulness",
            args.helpfulness_label,
            args.helpfulness_input,
            device,
            args.batch_size,
            args.max_length,
            args.local_files_only,
        )

    if "harmlessness" in args.objectives and has_missing_objective(
        records_by_method, "harmlessness"
    ):
        if not args.toxicity_model:
            raise RuntimeError(
                "Missing harmlessness. Pass --toxicity-model or precompute it."
            )
        if device is None:
            device = get_device(args.device)
        model_scores["harmlessness"] = evaluate_probability_model(
            records_by_method,
            args.toxicity_model,
            "toxicity",
            args.toxicity_label,
            args.toxicity_input,
            device,
            args.batch_size,
            args.max_length,
            args.local_files_only,
        )

    unsupported_missing = [
        objective
        for objective in args.objectives
        if objective not in DEFAULT_OBJECTIVES
        and has_missing_objective(records_by_method, objective)
    ]
    if unsupported_missing:
        raise RuntimeError(
            "Missing precomputed custom objectives: " + ", ".join(unsupported_missing)
        )

    scored_by_method, warnings = build_scored_records(
        records_by_method,
        method_paths,
        args.objectives,
        bounds,
        model_scores,
    )
    if any(not records for records in scored_by_method.values()):
        raise RuntimeError(
            "A method has no fully scored records: "
            + str({method: len(rows) for method, rows in scored_by_method.items()})
        )

    summary_rows = []
    per_alpha_rows = []
    for method, records in scored_by_method.items():
        metrics, preference_rows = core_metrics(records, args.objectives, reference)
        metrics.update(
            stratified_bootstrap(
                records,
                args.objectives,
                reference,
                args.bootstrap,
                args.confidence,
                args.seed,
            )
        )
        summary_rows.append({"Method": method, **metrics})

        front = set(pareto_front([row["point"] for row in preference_rows]))
        for row in preference_rows:
            output = {
                "Method": method,
                "Alpha": row["alpha"],
                "Samples": row["samples"],
                "MIP Contribution": row["mip"],
                "Pareto": row["point"] in front,
            }
            for objective, value in zip(args.objectives, row["point"]):
                output[f"Mean {objective}"] = value
            per_alpha_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "per_alpha.csv", per_alpha_rows)
    write_scored_records(
        args.output_dir / "scored_records.jsonl",
        scored_by_method,
        args.objectives,
    )

    report = {
        "objectives": list(args.objectives),
        "reference_point": list(reference),
        "bounds": {
            name: {"min": values[0], "max": values[1]}
            for name, values in bounds.items()
        },
        "pairing": args.pairing,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "confidence": args.confidence,
        "original_counts": original_counts,
        "evaluated_counts": {
            method: len(records)
            for method, records in scored_by_method.items()
        },
        "warnings": sorted(set(warnings)),
        "notes": [
            "All objectives are oriented so higher is better.",
            "MIP uses alpha_user, never dynamic alpha_t.",
            "HV uses one fixed reference shared by every method.",
            "Normalization bounds must be fixed outside the test set.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCore multi-objective evaluation completed.")
    print("Summary:", args.output_dir / "summary.csv")
    print("Per-alpha:", args.output_dir / "per_alpha.csv")
    print("Scored:", args.output_dir / "scored_records.jsonl")
    print("Protocol:", args.output_dir / "report.json")
    if warnings:
        print(f"Warnings: {len(set(warnings))} (see report.json)")


if __name__ == "__main__":
    main()
