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
        choices=["all", "base", "rad", "genarm", "cd", "multisignal"],
        default="all",
        help="Method to run.",
    )
    parser.add_argument(
        "--action",
        choices=["generate", "train", "cache"],
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
    parser.add_argument("--base-device", choices=["auto", "cuda", "cpu"], default="auto", help="Base LM device for MultiSignal cache.")
    parser.add_argument("--signal-device", choices=["auto", "cuda", "cpu"], default="cpu", help="Signal model device for MultiSignal cache/generate.")
    parser.add_argument("--cd-scorer", choices=["fudge", "cdq"], default="fudge", help="CD scorer type.")
    parser.add_argument("--cd-mode", choices=["tokenwise", "blockwise"], default="tokenwise", help="CD generation mode.")
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

    raise ValueError(f"Unknown method: {method}")


def method_cmds(method: str, action: str, args):
    if method == "all":
        return [
            build_method_command(name, action, args)
            for name in ["base", "genarm", "rad", "cd", "multisignal"]
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
        for method_name, cmd in zip(["base", "genarm", "rad", "cd", "multisignal"], commands):
            print(f"\n=== Running {method_name} [{args.action}] ===")
            run(cmd)
        return

    run(commands[0])


if __name__ == "__main__":
    main()
