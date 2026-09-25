#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a single method or all methods in the Method folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["all", "base", "rad", "genarm", "gsi", "cd", "parm", "multisignal", "cura"],
        default="all",
        help="Method to run.",
    )
    parser.add_argument(
        "--action",
        choices=["generate", "train", "cache", "audit-signals", "cache-status", "cache-verify", "calibrate", "annotate-targets", "oracle", "diagnose"],
        default="generate",
        help="What to execute for the selected method.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string for generate mode.")
    parser.add_argument("--input", type=str, default=None, help="Input dataset path for generate mode.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to generate.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k setting for decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy.")
    parser.add_argument("--split", choices=["train", "validation"], default="train", help="Cache split for MultiSignal.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for cache build.")
    parser.add_argument("--max-response-tokens", type=int, default=32, help="Max response tokens for cache build.")
    parser.add_argument("--cache", type=str, default=None, help="Cache file path for training.")
    parser.add_argument("--config", type=str, default="Method/CURA/configs/sentiment.json", help="CURA config path.")
    parser.add_argument("--cache-dir", type=str, default=None, help="CURA sharded cache directory.")
    parser.add_argument("--validation-cache-dir", type=str, default=None, help="CURA validation cache directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="CURA controller checkpoint.")
    parser.add_argument("--shard-size", type=int, default=100, help="CURA prompts per atomic shard.")
    parser.add_argument("--resume", action="store_true", help="Resume a compatible CURA cache.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry CURA cache errors.")
    parser.add_argument("--allow-objective-mismatch", action="store_true", help="Allow mismatched signals for CURA-MVP only.")
    parser.add_argument("--evaluator", type=str, default=None, help="Independent held-out evaluator for CURA targets.")
    parser.add_argument("--rollouts", type=int, default=2, help="Rollouts per CURA candidate target.")
    parser.add_argument("--rollout-tokens", type=int, default=16, help="Continuation tokens per CURA rollout.")
    parser.add_argument("--candidate-batch-size", type=int, default=10, help="CURA candidates annotated per rollout batch.")
    parser.add_argument("--target-save-every", type=int, default=100, help="Checkpoint CURA target shards every N token rows.")
    parser.add_argument("--signal-budget", type=int, default=None, help="Maximum CURA signals called per token.")
    parser.add_argument("--ablation", default="none", help="CURA generation ablation.")
    parser.add_argument("--fixed-lambda", type=float, default=None, help="CURA fixed-strength ablation.")
    parser.add_argument("--fixed-gate", type=float, default=None, help="CURA fixed gate in [0, 1].")
    parser.add_argument("--leave-out", type=str, default=None, help="CURA leave-one-signal-out ablation.")
    parser.add_argument("--corrupt-signal", type=str, default=None, help="CURA corrupted-signal test.")
    parser.add_argument("--corrupt-std", type=float, default=1.0, help="Noise scale for corrupted CURA signal.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--base-model", type=str, default=None, help="CURA unseen-base transfer model.")
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto", help="Base LM device for MultiSignal cache.")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu", help="Signal model device for MultiSignal cache/generate.")
    parser.add_argument("--cd-scorer", choices=["fudge", "cdq"], default="fudge", help="CD scorer type.")
    parser.add_argument("--cd-mode", choices=["tokenwise", "blockwise"], default="tokenwise", help="CD generation mode.")
    parser.add_argument("--alpha-helpfulness", type=float, default=0.5, help="PARM helpfulness preference weight.")
    parser.add_argument("--alpha-harmlessness", type=float, default=0.5, help="PARM harmlessness preference weight.")
    parser.add_argument("--parm-adapter", type=str, default=None, help="PARM PBLoRA adapter directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    return parser


def build_method_command(method: str, action: str, args):
    if method == "base":
        if action != "generate":
            raise ValueError("Base method supports only generate.")
        return [PYTHON, str(ROOT / "Method" / "generate_base.py")]

    if method == "genarm":
        if action != "generate":
            raise ValueError("GenARM method supports only generate.")
        return [PYTHON, str(ROOT / "Method" / "GenARM" / "generate_arm_gpt2_medium.py")]

    if method == "gsi":
        if action != "generate":
            raise ValueError("GSI supports only generate.")
        cmd = [PYTHON, str(ROOT / "Method" / "GSI" / "generate_gsi.py")]
        if args.input:
            cmd.extend(["--dataset-path", str(ROOT / args.input) if not Path(args.input).is_absolute() else args.input])
        if args.output:
            cmd.extend(["--output-path", str(ROOT / args.output) if not Path(args.output).is_absolute() else args.output])
        cmd.extend([
            "--num-prompts", str(args.num_prompts),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--device", args.base_device,
            "--seed", str(args.seed),
        ])
        return cmd

    if method == "rad":
        if action != "generate":
            raise ValueError("RAD method supports only generate.")
        cmd = [PYTHON, str(ROOT / "Method" / "RAD" / "generate.py")]
        cmd.extend(["--num-prompts", str(args.num_prompts), "--max-new-tokens", str(args.max_new_tokens), "--topk", str(args.top_k)])
        if args.output:
            cmd.extend(["--output-path", str(ROOT / args.output) if not Path(args.output).is_absolute() else args.output])
        if args.input:
            cmd.extend(["--input-path", str(ROOT / args.input) if not Path(args.input).is_absolute() else args.input])
        return cmd

    if method == "cd":
        if action != "generate":
            raise ValueError("CD method supports only generate.")
        return [PYTHON, str(ROOT / "Method" / "CD" / "generate_cd.py"), "--mode", args.cd_mode, "--scorer", args.cd_scorer, "--num-prompts", str(args.num_prompts)]

    if method == "parm":
        if action != "generate":
            raise ValueError("PARM supports only generate.")
        cmd = [
            PYTHON, str(ROOT / "Method" / "PARM" / "generate.py"),
            "--num-prompts", str(args.num_prompts),
            "--max-new-tokens", str(args.max_new_tokens),
            "--alpha-helpfulness", str(args.alpha_helpfulness),
            "--alpha-harmlessness", str(args.alpha_harmlessness),
            "--seed", str(args.seed),
        ]
        if args.input:
            cmd.extend(["--dataset-path", str(ROOT / args.input) if not Path(args.input).is_absolute() else args.input])
        if args.output:
            cmd.extend(["--output-path", str(ROOT / args.output) if not Path(args.output).is_absolute() else args.output])
        if args.base_model:
            cmd.extend(["--base-model", str(ROOT / args.base_model) if not Path(args.base_model).is_absolute() else args.base_model])
        if args.parm_adapter:
            cmd.extend(["--parm-adapter", str(ROOT / args.parm_adapter) if not Path(args.parm_adapter).is_absolute() else args.parm_adapter])
        return cmd

    if method == "multisignal":
        if action == "generate":
            cmd = [PYTHON, str(ROOT / "Method" / "MultiSignal" / "core" / "generate.py")]
            if args.prompt:
                cmd.extend(["--prompt", args.prompt])
            if args.input:
                cmd.extend(["--input", str(ROOT / args.input) if not Path(args.input).is_absolute() else args.input])
            if args.output:
                cmd.extend(["--output", str(ROOT / args.output) if not Path(args.output).is_absolute() else args.output])
            cmd.extend(["--max-prompts", str(args.num_prompts), "--max-new-tokens", str(args.max_new_tokens), "--top-k", str(args.top_k)])
            if args.do_sample:
                cmd.append("--do-sample")
            if args.temperature != 1.0:
                cmd.extend(["--temperature", str(args.temperature)])
            cmd.extend(["--signal-device", args.signal_device])
            return cmd

        if action == "train":
            cache_path = args.cache or "dataset/multisignal_cache/train.pt"
            return [PYTHON, str(ROOT / "Method" / "MultiSignal" / "core" / "train.py"), "--cache", cache_path]

        if action == "cache":
            cmd = [PYTHON, str(ROOT / "Method" / "MultiSignal" / "core" / "cache.py"), "--split", args.split]
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.max_response_tokens is not None:
                cmd.extend(["--max-response-tokens", str(args.max_response_tokens)])
            cmd.extend(["--top-k", str(args.top_k)])
            cmd.extend(["--base-device", args.base_device, "--signal-device", args.signal_device])
            return cmd

        raise ValueError(f"Unsupported action for MultiSignal: {action}")

    if method == "cura":
        core = ROOT / "Method" / "CURA" / "core"
        common = ["--config", args.config]
        if action == "audit-signals":
            cmd = [PYTHON, str(core / "audit_signals.py"), *common]
        elif action == "cache":
            cache_dir = args.cache_dir or f"dataset/cura_cache/{args.split}"
            cmd = [PYTHON, str(core / "cache.py"), *common, "--split", args.split,
                   "--cache-dir", cache_dir, "--shard-size", str(args.shard_size),
                   "--top-k", str(args.top_k), "--max-response-tokens", str(args.max_response_tokens),
                   "--base-device", args.base_device, "--signal-device", args.signal_device]
            if args.input: cmd.extend(["--input", args.input])
            if args.max_samples is not None: cmd.extend(["--max-samples", str(args.max_samples)])
            if args.resume: cmd.append("--resume")
            if args.retry_errors: cmd.append("--retry-errors")
        elif action in ("cache-status", "cache-verify"):
            cache_dir = args.cache_dir or f"dataset/cura_cache/{args.split}"
            cmd = [PYTHON, str(core / "cache_status.py"), "--cache-dir", cache_dir]
            if action == "cache-verify": cmd.append("--verify")
        elif action == "calibrate":
            cmd = [PYTHON, str(core / "calibrate.py"), *common,
                   "--cache-dir", args.cache_dir or "dataset/cura_cache/validation"]
        elif action == "annotate-targets":
            if not args.evaluator:
                raise ValueError("CURA annotate-targets requires --evaluator")
            cmd = [PYTHON, str(core / "annotate_targets.py"),
                   "--cache-dir", args.cache_dir or f"dataset/cura_cache/{args.split}",
                   "--evaluator", args.evaluator, "--rollouts", str(args.rollouts),
                   "--rollout-tokens", str(args.rollout_tokens), "--seed", str(args.seed),
                   "--candidate-batch-size", str(args.candidate_batch_size),
                   "--save-every-rows", str(args.target_save_every),
                   "--device", args.base_device]
        elif action == "oracle":
            cmd = [PYTHON, str(core / "oracle_study.py"),
                   "--cache-dir", args.cache_dir or "dataset/cura_cache/validation"]
        elif action == "diagnose":
            cmd = [PYTHON, str(core / "diagnose_checkpoint.py"),
                   "--cache-dir", args.cache_dir or "dataset/cura_cache/validation"]
            if args.checkpoint: cmd.extend(["--checkpoint", args.checkpoint])
            if args.output: cmd.extend(["--output", args.output])
            if args.base_device != "auto": cmd.extend(["--device", args.base_device])
        elif action == "train":
            cmd = [PYTHON, str(core / "train.py"), *common,
                   "--cache-dir", args.cache_dir or "dataset/cura_cache/train",
                   "--validation-cache-dir", args.validation_cache_dir or "dataset/cura_cache/validation",
                   "--seed", str(args.seed)]
            if args.output: cmd.extend(["--output", args.output])
            if args.resume: cmd.append("--resume")
            if args.base_device != "auto": cmd.extend(["--device", args.base_device])
        elif action == "generate":
            cmd = [PYTHON, str(core / "generate.py"),
                   "--input", args.input or "dataset/rad_benchmark/all.jsonl",
                   "--output", args.output or "results/cura.json", "--num-prompts", str(args.num_prompts),
                   "--max-new-tokens", str(args.max_new_tokens), "--top-k", str(args.top_k),
                   "--base-device", args.base_device, "--signal-device", args.signal_device]
            if args.checkpoint: cmd.extend(["--checkpoint", args.checkpoint])
            if args.base_model: cmd.extend(["--base-model", args.base_model])
            if args.do_sample: cmd.append("--do-sample")
            if args.signal_budget is not None: cmd.extend(["--signal-budget", str(args.signal_budget)])
            if args.ablation != "none": cmd.extend(["--ablation", args.ablation])
            if args.fixed_lambda is not None: cmd.extend(["--fixed-lambda", str(args.fixed_lambda)])
            if args.fixed_gate is not None: cmd.extend(["--fixed-gate", str(args.fixed_gate)])
            if args.leave_out: cmd.extend(["--leave-out", args.leave_out])
            if args.corrupt_signal: cmd.extend(["--corrupt-signal", args.corrupt_signal, "--corrupt-std", str(args.corrupt_std)])
            cmd.extend(["--seed", str(args.seed)])
        else:
            raise ValueError(f"Unsupported action for CURA: {action}")
        if args.allow_objective_mismatch and action in ("audit-signals", "cache", "generate"):
            cmd.append("--allow-objective-mismatch")
        return cmd

    raise ValueError(f"Unknown method: {method}")


def method_cmds(method: str, action: str, args):
    if method == "all":
        return [
            build_method_command(name, action, args)
            for name in ["base", "genarm", "gsi", "rad", "cd", "multisignal"]
        ]
    return [build_method_command(method, action, args)]


def run(cmd):
    print(f"\nExecuting: {shlex.join(str(part) for part in cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = build_parser()
    args = parser.parse_args()
    commands = method_cmds(args.method, args.action, args)

    if args.dry_run:
        for cmd in commands:
            print(shlex.join(str(part) for part in cmd))
        return

    if args.method == "all":
        for method_name, cmd in zip(["base", "genarm", "gsi", "rad", "cd", "multisignal"], commands):
            print(f"\n=== Running {method_name} [{args.action}] ===")
            run(cmd)
        return

    run(commands[0])


if __name__ == "__main__":
    main()
