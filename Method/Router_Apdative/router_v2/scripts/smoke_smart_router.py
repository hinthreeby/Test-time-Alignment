"""Run a synthetic Smart Router V2 forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import build_smart_router


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "router_v2"
    / "configs"
    / "v2_topk_state_history_alpha.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = replace(
        SmartRouterConfig.load_json(args.config),
        device=args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    model = build_smart_router(config)
    device = next(model.parameters()).device
    leading_shape = (2, 4) if config.use_history else (2,)
    full_shape = leading_shape + (config.vocab_size,)
    base_logits = torch.randn(full_shape, device=device, requires_grad=True)
    guide_logits = torch.randn(full_shape, device=device, requires_grad=True)
    position = None
    if config.use_position:
        if config.use_history:
            position = torch.arange(4, device=device).expand(2, -1)
        else:
            position = torch.tensor([0, 1], device=device)
    selected_score = (
        torch.zeros(leading_shape, device=device)
        if config.use_history
        else None
    )
    preference = None
    if config.use_preference:
        preference = torch.ones(
            leading_shape[0],
            config.preference_dim,
            device=device,
        )
    output = model.forward_from_full_logits(
        base_logits,
        guide_logits,
        position=position,
        selected_score=selected_score,
        preference=preference,
    )
    if output.guided_logits is None:
        raise RuntimeError("Smart Router smoke did not route full logits")
    loss = output.guided_logits.square().mean() + output.lambda_t.mean()
    loss.backward()
    router_gradient_sum = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if router_gradient_sum <= 0.0:
        raise RuntimeError("No gradient reached Smart Router parameters")
    if base_logits.grad is not None or guide_logits.grad is not None:
        raise RuntimeError("Gradient entered frozen base/guide logits")
    result = {
        "status": "PASS",
        "variant": config.variant,
        "device": str(device),
        "lambda_shape": list(output.lambda_t.shape),
        "lambda_min": float(output.lambda_t.detach().min()),
        "lambda_max_observed": float(output.lambda_t.detach().max()),
        "configured_lambda_max": config.lambda_max,
        "guided_logits_shape": list(output.guided_logits.shape),
        "router_state_shape": (
            list(output.router_state.shape)
            if output.router_state is not None
            else None
        ),
        "feature_dim": model.feature_dim,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "router_gradient_sum": router_gradient_sum,
        "base_logits_grad": None,
        "guide_logits_grad": None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
