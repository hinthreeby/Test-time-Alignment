#!/usr/bin/env python3
"""Generate resumable, evaluator-compatible outputs with PARM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
PARM_ROOT = Path(__file__).resolve().parent
MODEL_ARITHMETIC_SRC = PARM_ROOT / "language-model-arithmetic" / "src"
PEFT_SRC = PARM_ROOT / "peft" / "src"

BASE_MODEL = ROOT / "models" / "tulu-2-7b"
PARM_ADAPTER = (
    ROOT
    / "Method"
    / "PARM_TARO"
    / "results"
    / "parm_taro"
    / "recovery"
    / "reproduced"
    / "pblora"
    / "final_checkpoint"
)
INPUT_FILE = ROOT / "dataset" / "rad_benchmark" / "all.jsonl"
OUTPUT_FILE = ROOT / "results" / "parm.json"
CACHE_DIR = ROOT / "results" / ".parm_cache"

PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
PROMPT_USER = "USER: {input} "
PROMPT_ASSISTANT = "ASSISTANT:"
PROMPT_TEMPLATE = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable PARM generation.")
    parser.add_argument("--dataset-path", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--parm-adapter", type=Path, default=PARM_ADAPTER)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--alpha-helpfulness", type=float, default=0.5)
    parser.add_argument("--alpha-harmlessness", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-logit", action="store_true")
    args = parser.parse_args()
    if args.num_prompts < 0:
        parser.error("--num-prompts must be >= 0")
    if args.max_new_tokens <= 0 or args.max_length <= 0:
        parser.error("--max-new-tokens and --max-length must be > 0")
    if args.alpha_helpfulness < 0 or args.alpha_harmlessness < 0:
        parser.error("preference weights must be >= 0")
    return args


def text_value(value) -> str:
    return str(value.get("text", "")) if isinstance(value, dict) else str(value or "")


def load_samples(path: Path, limit: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content or limit == 0:
        return []
    try:
        payload = json.loads(content)
        rows = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict) and text_value(row.get("prompt")).strip()][:limit]


def record_key(record: dict) -> tuple[str, str]:
    if record.get("md5_hash"):
        return "md5", str(record["md5_hash"])
    # Preference datasets can contain the same prompt multiple times with
    # different response pairs. Preserve those cases instead of collapsing
    # them by prompt text during resume/save.
    if record.get("id") is not None:
        return "id", str(record["id"])
    if record.get("sample_id") is not None:
        return "sample_id", str(record["sample_id"])
    return "prompt", text_value(record.get("prompt")).strip()


def is_completed(record: dict | None) -> bool:
    return bool(
        record
        and record.get("status", "success") == "success"
        and isinstance(record.get("response"), str)
        and record["response"].strip()
    )


def load_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Resume output must be a JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def save_results(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare_adapter(source: Path, cache_root: Path, alpha_help: float, alpha_harm: float) -> Path:
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Invalid PARM adapter directory: {source}")
    name = f"h{alpha_help:g}_s{alpha_harm:g}".replace(".", "p")
    target = cache_root / name
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # PBLoRA stores preferences in [harmlessness, helpfulness] order.
    config["pref_vec_init"] = [alpha_harm, alpha_help]
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    target_weights = target / "adapter_model.safetensors"
    if not target_weights.exists() or target_weights.stat().st_size != weights_path.stat().st_size:
        shutil.copy2(weights_path, target_weights)
    return target


def build_generator(args: argparse.Namespace):
    for source in (PEFT_SRC, MODEL_ARITHMETIC_SRC):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from model_arithmetic import ModelArithmetic, PromptedLLM
    from transformers import AutoTokenizer

    adapter = prepare_adapter(
        args.parm_adapter,
        args.cache_dir,
        args.alpha_helpfulness,
        args.alpha_harmlessness,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    template = lambda _system_prompt, input_string: PROMPT_TEMPLATE.format(input=input_string)
    base = PromptedLLM(
        system_prompt="Not used",
        prompt_template=template,
        model=str(args.base_model),
        tokenizer=tokenizer,
    )
    reward = PromptedLLM(
        system_prompt="Not used",
        prompt_template=template,
        model=str(adapter),
        tokenizer=tokenizer,
    )
    model = ModelArithmetic(base + reward, max_length=args.max_length)
    model.eval()
    temperature = 1.0 if args.normalize_logit else 0

    def generate(prompt: str) -> str:
        output = model.generate_text(
            prompt,
            max_new_tokens=args.max_new_tokens,
            batch_size=None,
            temperature=temperature,
            top_p=1,
            top_k=0,
            do_speculation=False,
        )[0]
        if tokenizer.eos_token:
            output = output.removesuffix(tokenizer.eos_token)
        return output.strip()

    return generate


def main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset_path, args.num_prompts)
    existing = load_results(args.output_path)
    by_key = {record_key(row): row for row in existing}
    pending = [
        (index, sample)
        for index, sample in enumerate(samples)
        if not is_completed(by_key.get(record_key(sample)))
    ]
    print(f"Method: PARM ({args.alpha_helpfulness:g} helpfulness, {args.alpha_harmlessness:g} harmlessness)")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Target: {len(samples)} | Completed: {len(samples) - len(pending)} | Remaining: {len(pending)}")
    if not pending:
        return
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    torch.manual_seed(args.seed)
    generate = build_generator(args)
    for index, sample in tqdm(pending, desc="PARM generation", unit="prompt"):
        prompt = text_value(sample.get("prompt")).strip()
        started = time.perf_counter()
        try:
            response = generate(prompt)
            if not response:
                raise RuntimeError("PARM generated an empty response")
            status, error = "success", None
        except Exception as exc:
            response, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
        by_key[record_key(sample)] = {
            "id": sample.get("id", index),
            "md5_hash": sample.get("md5_hash"),
            "prompt": prompt,
            "reference": text_value(sample.get("continuation")),
            "response": response,
            "method": "PARM",
            "alpha_helpfulness": args.alpha_helpfulness,
            "alpha_harmlessness": args.alpha_harmlessness,
            "base_model": str(args.base_model),
            "parm_adapter": str(args.parm_adapter),
            "latency": round(time.perf_counter() - started, 4),
            "status": status,
            "error": error,
        }
        save_results(list(by_key.values()), args.output_path)
    completed = sum(is_completed(by_key.get(record_key(sample))) for sample in samples)
    print(f"Completed target: {completed}/{len(samples)}")
    print(f"Saved {len(by_key)} records: {args.output_path}")


if __name__ == "__main__":
    main()
