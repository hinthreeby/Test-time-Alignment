"""Run one production PARM-TARO generation after prerequisites pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from PARM_TARO.config import ParmTaroConfig
from PARM_TARO.decoding.adaptive import ParmTaroDecoder
from PARM_TARO.runtime import load_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "PARM_TARO" / "configs" / "parm_taro_tulu2.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-cpu-fallback", action="store_true")
    args = parser.parse_args()
    values = ParmTaroConfig.load_json(args.config).to_dict()
    if args.device is not None:
        values["device"] = args.device
    if args.no_cpu_fallback:
        values["allow_cpu_fallback"] = False
    config = ParmTaroConfig.from_dict(values)
    runtime = load_runtime(config)
    decoder = ParmTaroDecoder(
        base_model=runtime.base_model,
        guide_model=runtime.guide_model,
        tokenizer=runtime.tokenizer,
        alignment=runtime.alignment,
        lambda_provider=runtime.lambda_provider,
        device=runtime.device,
        preference_dim=config.preference_dim,
        top_k=config.top_k,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        stop_on_eos=config.stop_on_eos,
    )
    result = decoder.generate(
        args.prompt,
        preference=torch.tensor(config.preference, device=runtime.device),
        seed=config.seed,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
