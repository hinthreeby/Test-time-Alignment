"""Train a GPT-2 LoRA guide with positive-continuation causal NLL."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from router_v2.cache.io import require_path_within, sha256_file
from router_v2.device import resolve_device
from router_v2.guide_model.config import SentimentGuideConfig
from router_v2.guide_model.data import (
    PositiveContinuationDataset,
    collate_tokenized_samples,
    load_rad_samples,
)
from router_v2.guide_model.provenance import checkpoint_descriptor
from router_v2.guide_model.training import (
    assert_optimizer_parameters_fp32,
    canonical_config_sha256,
    cast_trainable_parameters_to_fp32,
    capture_rng_state,
    cleanup_stale_checkpoint_temps,
    deterministic_epoch_batches,
    latest_checkpoint,
    restore_rng_state,
    save_adapter_directory_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "router_v2" / "configs" / "sentiment_guide.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "router_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume a checkpoint path, or the latest checkpoint when omitted.",
    )
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    config = SentimentGuideConfig.load_json(args.config)
    output_dir = require_path_within(
        args.output_dir or _resolve_project_path(config.output_dir),
        RESULTS_ROOT,
        label="sentiment guide output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_temps = cleanup_stale_checkpoint_temps(output_dir)

    device = resolve_device(
        args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    effective_precision = (
        "fp16"
        if config.precision == "fp16" and device.type == "cuda"
        else "fp32"
    )
    model_dtype = (
        torch.float16 if effective_precision == "fp16" else torch.float32
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    base_model_path = _resolve_project_path(config.base_model_path)
    train_data_path = _resolve_project_path(config.train_data_path)
    samples = load_rad_samples(
        train_data_path,
        max_samples=args.max_train_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must define EOS")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = PositiveContinuationDataset(
        samples,
        tokenizer,
        max_length=config.max_length,
        include_eos_target=config.include_eos_target,
    )

    config_payload = config.to_dict()
    run_fingerprint_payload = {
        "config": config_payload,
        "train_data_sha256": sha256_file(train_data_path),
        "num_train_samples": len(dataset),
    }
    run_fingerprint = canonical_config_sha256(run_fingerprint_payload)

    resume_path: Path | None = None
    if args.resume:
        resume_path = (
            latest_checkpoint(output_dir)
            if args.resume == "auto"
            else Path(args.resume).resolve()
        )
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError("No sentiment guide checkpoint to resume")
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to start a new guide run in a non-empty output "
            f"directory: {output_dir}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if resume_path is None:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = PeftModel.from_pretrained(
            base_model,
            resume_path,
            is_trainable=True,
        )
    trainable_parameters = cast_trainable_parameters_to_fp32(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    assert_optimizer_parameters_fp32(optimizer)
    batches_per_epoch = math.ceil(len(dataset) / config.train_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / config.gradient_accumulation_steps
    )
    total_optimizer_steps = (
        optimizer_steps_per_epoch * config.num_train_epochs
    )
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(effective_precision == "fp16"),
    )

    cursor_epoch = 0
    cursor_batch = 0
    global_step = 0
    if resume_path is not None:
        state = torch.load(
            resume_path / "training_state.pt",
            map_location=device,
            weights_only=False,
        )
        if state["run_fingerprint"] != run_fingerprint:
            raise ValueError("Resume checkpoint run fingerprint does not match")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        restore_rng_state(state["rng_state"])
        cursor_epoch = int(state["next_epoch"])
        cursor_batch = int(state["next_batch_index"])
        global_step = int(state["global_step"])

    base_descriptor = checkpoint_descriptor(base_model_path)
    metadata_base = {
        "schema_version": 1,
        "task": config.task,
        "objective": config.objective,
        "objective_equation": (
            "-sum_{t in continuation+EOS} "
            "log pi_guide(y_t | prompt, y_<t)"
        ),
        "prompt_labels_masked": True,
        "base_model": base_descriptor,
        "train_data_path": str(train_data_path),
        "train_data_sha256": sha256_file(train_data_path),
        "num_train_samples": len(dataset),
        "run_fingerprint": run_fingerprint,
        "config": config_payload,
        "device": str(device),
        "effective_precision": effective_precision,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "removed_stale_temp_directories": removed_temps,
    }
    log_path = output_dir / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cursor_epoch, config.num_train_epochs):
        epoch_batches = deterministic_epoch_batches(
            num_samples=len(dataset),
            batch_size=config.train_batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        start_batch = cursor_batch if epoch == cursor_epoch else 0
        accumulation = 0
        accumulation_target = 0
        accumulated_loss = 0.0
        for batch_index in range(start_batch, len(epoch_batches)):
            if accumulation == 0:
                accumulation_target = min(
                    config.gradient_accumulation_steps,
                    len(epoch_batches) - batch_index,
                )
            items = [dataset[index] for index in epoch_batches[batch_index]]
            batch = collate_tokenized_samples(
                items,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(effective_precision == "fp16"),
            ):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                ).loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, "
                    f"batch={batch_index}"
                )
            accumulated_loss += float(loss.detach().cpu())
            accumulation += 1
            scaler.scale(loss / accumulation_target).backward()
            last_batch = batch_index + 1 == len(epoch_batches)
            if accumulation < accumulation_target and not last_batch:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            mean_loss = accumulated_loss / accumulation
            accumulation = 0
            accumulated_loss = 0.0
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch == len(epoch_batches):
                next_epoch = epoch + 1
                next_batch = 0

            if (
                global_step == 1
                or global_step % config.log_optimizer_steps == 0
            ):
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "next_batch_index": next_batch,
                    "loss": mean_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "device": str(device),
                }
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

            if (
                global_step == 1
                or global_step % config.save_optimizer_steps == 0
            ):
                checkpoint = output_dir / (
                    f"checkpoint-step-{global_step:07d}"
                )
                training_state = {
                    "run_fingerprint": run_fingerprint,
                    "global_step": global_step,
                    "next_epoch": next_epoch,
                    "next_batch_index": next_batch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "complete": False,
                }
                save_adapter_directory_atomic(
                    model=model,
                    tokenizer=tokenizer,
                    destination=checkpoint,
                    training_state=training_state,
                    metadata={
                        **metadata_base,
                        "global_step": global_step,
                        "complete": False,
                    },
                )

        cursor_batch = 0

    final_path = output_dir / "final_adapter"
    final_state = {
        "run_fingerprint": run_fingerprint,
        "global_step": global_step,
        "next_epoch": config.num_train_epochs,
        "next_batch_index": 0,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "complete": True,
    }
    save_adapter_directory_atomic(
        model=model,
        tokenizer=tokenizer,
        destination=final_path,
        training_state=final_state,
        metadata={
            **metadata_base,
            "global_step": global_step,
            "complete": True,
        },
    )
    print(f"Saved final sentiment guide adapter: {final_path}")


if __name__ == "__main__":
    main()
