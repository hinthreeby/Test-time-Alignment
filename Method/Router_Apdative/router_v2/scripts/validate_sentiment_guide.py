"""Validate a trained sentiment guide before any cache extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from router_v2.cache.io import (
    require_path_within,
    sha256_file,
    write_json_atomic,
)
from router_v2.device import resolve_device
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.data import (
    PositiveContinuationDataset,
    RADSample,
    load_rad_samples,
)
from router_v2.guide_model.evaluation import (
    load_negative_audit_samples,
    paired_score_summary,
    score_tokenized_samples,
)
from router_v2.guide_model.provenance import (
    checkpoint_descriptor,
    tokenizer_descriptor,
    tokenizers_exactly_compatible,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "router_v2" / "configs" / "sentiment_guide.json"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "router_v2"
    / "reports"
    / "sentiment_guide_validation.json"
)
REPORT_ROOT = PROJECT_ROOT / "router_v2" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--guide-adapter", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = SentimentGuideConfig.load_json(args.config)
    base_path = _project_path(config.base_model_path)
    guide_path = (
        args.guide_adapter.resolve()
        if args.guide_adapter
        else _project_path(config.output_dir) / "final_adapter"
    )
    output_path = require_path_within(
        args.output,
        REPORT_ROOT,
        label="guide validation report",
    )
    if not guide_path.exists():
        raise FileNotFoundError(f"Missing sentiment guide adapter: {guide_path}")

    device = resolve_device(
        args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    use_fp16 = config.precision == "fp16" and device.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    base_tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        use_fast=True,
    )
    guide_tokenizer = AutoTokenizer.from_pretrained(
        guide_path,
        local_files_only=True,
        use_fast=True,
    )
    for tokenizer in (base_tokenizer, guide_tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    tokenizer_compatible, tokenizer_checks = tokenizers_exactly_compatible(
        base_tokenizer,
        guide_tokenizer,
    )

    positive_samples = load_rad_samples(
        _project_path(config.validation_data_path),
        max_samples=config.validation_positive_samples,
    )
    positive_dataset = PositiveContinuationDataset(
        positive_samples,
        base_tokenizer,
        max_length=config.max_length,
        include_eos_target=config.include_eos_target,
    )
    negative_samples = load_negative_audit_samples(
        _project_path(config.negative_audit_data_path),
        base_tokenizer,
        max_samples=config.validation_negative_samples,
        max_length=config.max_length,
    )
    negative_dataset = PositiveContinuationDataset(
        negative_samples,
        base_tokenizer,
        max_length=config.max_length,
        include_eos_target=config.include_eos_target,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    guide_base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    guide_model = PeftModel.from_pretrained(
        guide_base,
        guide_path,
        is_trainable=False,
    ).to(device)
    for model in (base_model, guide_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    shape_batch = positive_dataset[:1]
    from router_v2.guide_model.data import collate_tokenized_samples

    shape_inputs = collate_tokenized_samples(
        shape_batch,
        pad_token_id=int(base_tokenizer.pad_token_id),
    )
    with torch.inference_mode():
        shape_logits = guide_model(
            input_ids=shape_inputs["input_ids"].to(device),
            attention_mask=shape_inputs["attention_mask"].to(device),
            use_cache=False,
        ).logits
    expected_shape = (
        1,
        shape_inputs["input_ids"].shape[1],
        len(base_tokenizer),
    )
    output_shape_valid = tuple(shape_logits.shape) == expected_shape
    output_finite = bool(torch.isfinite(shape_logits).all())

    scoring_kwargs = {
        "batch_size": config.eval_batch_size,
        "pad_token_id": int(base_tokenizer.pad_token_id),
        "device": device,
        "use_fp16": use_fp16,
    }
    base_positive = score_tokenized_samples(
        base_model,
        positive_dataset.items,
        **scoring_kwargs,
    )
    guide_positive = score_tokenized_samples(
        guide_model,
        positive_dataset.items,
        **scoring_kwargs,
    )
    base_negative = score_tokenized_samples(
        base_model,
        negative_dataset.items,
        **scoring_kwargs,
    )
    guide_negative = score_tokenized_samples(
        guide_model,
        negative_dataset.items,
        **scoring_kwargs,
    )
    positive_summary = paired_score_summary(
        base_positive,
        guide_positive,
    )
    negative_summary = paired_score_summary(
        base_negative,
        guide_negative,
    )
    positive_nll_improvement = (
        positive_summary["base_mean_nll"]
        - positive_summary["guide_mean_nll"]
    )
    direction_delta_gap = (
        positive_summary["mean_guide_minus_base"]
        - negative_summary["mean_guide_minus_base"]
    )

    base_checkpoint = checkpoint_descriptor(base_path)
    guide_checkpoint = checkpoint_descriptor(guide_path)
    parameters_frozen = all(
        not parameter.requires_grad
        for model in (base_model, guide_model)
        for parameter in model.parameters()
    )
    checks = {
        "full_vocabulary_output": output_shape_valid and output_finite,
        "exact_tokenizer_compatible": tokenizer_compatible,
        "separate_checkpoints": (
            base_checkpoint["path"] != guide_checkpoint["path"]
            and base_checkpoint["checkpoint_sha256"]
            != guide_checkpoint["checkpoint_sha256"]
        ),
        "models_frozen": parameters_frozen,
        "inference_mode_used": True,
        "positive_nll_improved": (
            positive_nll_improvement
            > config.minimum_positive_nll_improvement
        ),
        "sentiment_direction_gap_positive": (
            direction_delta_gap > config.minimum_direction_delta_gap
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    inspect_count = config.validation_inspection_samples
    positive_text = {
        sample.sample_id: sample for sample in positive_samples
    }
    negative_text = {
        sample.sample_id: sample for sample in negative_samples
    }

    def inspectable_rows(
        rows: list[dict[str, object]],
        samples_by_id: dict[str, RADSample],
    ) -> list[dict[str, object]]:
        output = []
        for row in rows[:inspect_count]:
            sample = samples_by_id[str(row["sample_id"])]
            output.append(
                {
                    **row,
                    "prompt": sample.prompt,
                    "continuation": sample.continuation,
                    "sentiment_label": sample.label,
                    "source_split": sample.source_split,
                }
            )
        return output

    report = {
        "schema_version": 1,
        "status": status,
        "task": "sentiment",
        "objective": config.objective,
        "device": str(device),
        "precision": "fp16" if use_fp16 else "fp32",
        "checks": checks,
        "output_contract": {
            "actual_shape": list(shape_logits.shape),
            "expected_shape": list(expected_shape),
            "last_token_shape": [
                shape_logits.shape[0],
                shape_logits.shape[-1],
            ],
            "finite": output_finite,
        },
        "base_model": base_checkpoint,
        "guide_model": guide_checkpoint,
        "base_tokenizer": tokenizer_descriptor(
            base_tokenizer,
            base_path,
        ),
        "guide_tokenizer": tokenizer_descriptor(
            guide_tokenizer,
            guide_path,
        ),
        "tokenizer_compatibility_checks": tokenizer_checks,
        "runtime_padding_behavior": (
            "Both tokenizers use EOS as right-padding; padding labels are masked."
        ),
        "validation_source": {
            "path": config.validation_data_path,
            "sha256": sha256_file(
                _project_path(config.validation_data_path)
            ),
        },
        "negative_audit_source": {
            "path": config.negative_audit_data_path,
            "sha256": sha256_file(
                _project_path(config.negative_audit_data_path)
            ),
        },
        "positive": {
            key: value
            for key, value in positive_summary.items()
            if key != "rows"
        },
        "negative": {
            key: value
            for key, value in negative_summary.items()
            if key != "rows"
        },
        "positive_nll_improvement": positive_nll_improvement,
        "direction_delta_gap": direction_delta_gap,
        "acceptance_thresholds": {
            "minimum_positive_nll_improvement": (
                config.minimum_positive_nll_improvement
            ),
            "minimum_direction_delta_gap": (
                config.minimum_direction_delta_gap
            ),
        },
        "inspectable_positive_rows": inspectable_rows(
            positive_summary["rows"],
            positive_text,
        ),
        "inspectable_negative_rows": inspectable_rows(
            negative_summary["rows"],
            negative_text,
        ),
    }
    write_json_atomic(
        output_path,
        report,
        overwrite=args.overwrite_report,
    )
    print(json.dumps({"status": status, "report": str(output_path)}))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
