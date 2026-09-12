"""Autoregressive RAD decoding for TARO and Smart Router V2."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from router_v2.cache.features import compute_confidence_disagreement
from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.model import TAROTokenRouter
from router_v2.smart_model import SmartRouterBatch, SmartTokenRouter


@dataclass(frozen=True)
class RouteSpec:
    label: str
    kind: str
    value: float | None = None

    def __post_init__(self) -> None:
        allowed = {
            "base",
            "fixed",
            "heuristic_entropy_gap",
            "heuristic_js",
            "taro",
            "v2_state",
            "v2_history",
        }
        if self.kind not in allowed:
            raise ValueError(f"Unsupported route kind: {self.kind}")
        if self.kind == "fixed":
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise ValueError("Fixed routes require value in [0, 1]")
        elif self.value is not None:
            raise ValueError(f"Route kind {self.kind} does not accept a value")


@dataclass(frozen=True)
class V2Generation:
    generated_text: str
    selected_token_ids: list[int]
    lambda_history: list[float]
    base_entropy_history: list[float]
    guide_entropy_history: list[float]
    js_history: list[float]
    topk_overlap_history: list[float]
    base_conditional_perplexity: float
    eos_generated: bool
    terminated_on_eos: bool
    latency: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_lambda(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def fixed_spec(value: float, *, label: str | None = None) -> RouteSpec:
    return RouteSpec(
        label=label or f"v2_fixed_lambda_{format_lambda(value)}",
        kind="fixed",
        value=float(value),
    )


def validation_route_specs(
    fixed_lambdas: tuple[float, ...],
    heuristics: tuple[str, ...],
) -> list[RouteSpec]:
    specs = [fixed_spec(value) for value in fixed_lambdas]
    specs.extend(RouteSpec(label=name, kind=name) for name in heuristics)
    specs.extend(
        (
            RouteSpec(label="taro", kind="taro"),
            RouteSpec(label="v2_history", kind="v2_history"),
        )
    )
    return specs


def test_route_specs(
    *,
    best_fixed_lambda: float,
    best_heuristic: str,
    taro_same_average: float,
    v2_same_average: float,
) -> list[RouteSpec]:
    return [
        RouteSpec(label="v2_base", kind="base"),
        fixed_spec(best_fixed_lambda, label="v2_fixed"),
        RouteSpec(label="v2_heuristic", kind=best_heuristic),
        RouteSpec(label="taro", kind="taro"),
        RouteSpec(label="v2_state", kind="v2_state"),
        RouteSpec(label="v2_history", kind="v2_history"),
        fixed_spec(taro_same_average, label="taro_same_average"),
        fixed_spec(v2_same_average, label="v2_same_average"),
    ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, function: Any) -> tuple[Any, float]:
    _synchronize(device)
    start = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, time.perf_counter() - start


def _model_step(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Any,
) -> tuple[torch.Tensor, Any]:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = output.logits[:, -1].float().detach()
    if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Causal model emitted invalid next-token logits")
    return logits, output.past_key_values


def heuristic_lambda(
    kind: str,
    topk: TAROTopKBatch,
    *,
    position: int,
    max_position: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    position_tensor = torch.full(
        topk.base_logits.shape[:-1],
        int(position),
        dtype=torch.long,
        device=topk.base_logits.device,
    )
    features = compute_confidence_disagreement(
        topk.base_token_ids,
        topk.base_logits,
        topk.reward_token_ids,
        topk.reward_logits,
        position_tensor,
        max_position=max_position,
    )
    if kind == "heuristic_entropy_gap":
        # Guide only when its Top-K distribution is more confident than base.
        value = (
            (features.base_entropy - features.guide_entropy)
            / math.log(topk.base_logits.shape[-1])
        ).clamp(0.0, 1.0)
    elif kind == "heuristic_js":
        value = (features.js_divergence / math.log(2.0)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported heuristic: {kind}")
    diagnostics = {
        "base_entropy": features.base_entropy,
        "guide_entropy": features.guide_entropy,
        "js": features.js_divergence,
        "topk_overlap": features.topk_overlap,
    }
    return value.unsqueeze(-1), diagnostics


class RADV2Decoder:
    """Decode one prompt while keeping all source models frozen."""

    def __init__(
        self,
        *,
        base_model: Any,
        guide_model: Any,
        tokenizer: Any,
        routers: Mapping[str, TAROTokenRouter | SmartTokenRouter],
        device: torch.device,
        top_k: int,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stop_on_eos: bool,
    ) -> None:
        if top_k <= 0 or max_new_tokens <= 0 or temperature <= 0.0:
            raise ValueError("Invalid decoding parameters")
        self.base_model = base_model.eval()
        self.guide_model = guide_model.eval()
        self.tokenizer = tokenizer
        self.routers = dict(routers)
        self.device = device
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.stop_on_eos = stop_on_eos
        for model in (self.base_model, self.guide_model, *self.routers.values()):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        required = {"taro", "v2_state", "v2_history"}
        if set(self.routers) != required:
            raise ValueError(f"Router mapping must contain exactly {sorted(required)}")

    def _router_lambda(
        self,
        spec: RouteSpec,
        current_topk: TAROTopKBatch,
        *,
        position: int,
        history_topk: list[TAROTopKBatch],
        selected_scores: list[float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, diagnostics = heuristic_lambda(
            "heuristic_js",
            current_topk,
            position=position,
            max_position=max(self.max_new_tokens - 1, 1),
        )
        if spec.kind == "fixed":
            value = torch.full(
                current_topk.base_logits.shape[:-1] + (1,),
                float(spec.value),
                dtype=current_topk.base_logits.dtype,
                device=self.device,
            )
            return value, diagnostics
        if spec.kind.startswith("heuristic_"):
            return heuristic_lambda(
                spec.kind,
                current_topk,
                position=position,
                max_position=max(self.max_new_tokens - 1, 1),
            )
        if spec.kind == "taro":
            output = self.routers["taro"].predict_alpha(current_topk)
            return output, diagnostics
        if spec.kind == "v2_state":
            router = self.routers["v2_state"]
            if not isinstance(router, SmartTokenRouter):
                raise TypeError("v2_state checkpoint is not SmartTokenRouter")
            output = router.predict_lambda(
                SmartRouterBatch(
                    topk=current_topk,
                    position=torch.tensor([position], device=self.device),
                )
            )
            return output.lambda_t, diagnostics
        if spec.kind != "v2_history":
            raise ValueError(f"Unsupported guided route: {spec.kind}")
        router = self.routers["v2_history"]
        if not isinstance(router, SmartTokenRouter):
            raise TypeError("v2_history checkpoint is not SmartTokenRouter")
        topk = TAROTopKBatch(
            base_token_ids=torch.stack(
                [value.base_token_ids for value in history_topk], dim=1
            ),
            base_logits=torch.stack(
                [value.base_logits for value in history_topk], dim=1
            ),
            reward_token_ids=torch.stack(
                [value.reward_token_ids for value in history_topk], dim=1
            ),
            reward_logits=torch.stack(
                [value.reward_logits for value in history_topk], dim=1
            ),
        )
        scores = torch.tensor(
            [selected_scores + [0.0]],
            dtype=topk.base_logits.dtype,
            device=self.device,
        )
        output = router.predict_lambda(
            SmartRouterBatch(
                topk=topk,
                position=torch.arange(
                    len(history_topk), device=self.device
                ).unsqueeze(0),
                selected_score=scores,
            )
        )
        return output.lambda_t[:, -1], diagnostics

    def generate(self, prompt: str, spec: RouteSpec, *, seed: int) -> V2Generation:
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            self.device
        )
        if input_ids.shape[1] == 0:
            raise ValueError("Prompt tokenization produced no tokens")
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        base_input = input_ids
        guide_input = input_ids
        base_past = None
        guide_past = None
        selected_ids: list[int] = []
        selected_base_logprobs: list[float] = []
        selected_scores: list[float] = []
        lambda_history: list[float] = []
        base_entropy_history: list[float] = []
        guide_entropy_history: list[float] = []
        js_history: list[float] = []
        overlap_history: list[float] = []
        history_topk: list[TAROTopKBatch] = []
        base_seconds = 0.0
        guide_seconds = 0.0
        router_seconds = 0.0
        sampling_seconds = 0.0
        eos_generated = False
        terminated_on_eos = False
        total_start = time.perf_counter()

        with torch.inference_mode():
            for position in range(self.max_new_tokens):
                base_result, elapsed = _timed_call(
                    self.device,
                    lambda: _model_step(
                        self.base_model,
                        base_input,
                        attention_mask,
                        position_ids,
                        base_past,
                    ),
                )
                base_logits, base_past = base_result
                base_seconds += elapsed
                if spec.kind == "base":
                    guide_logits = None
                    lambda_t = torch.zeros(
                        (1, 1), dtype=base_logits.dtype, device=self.device
                    )
                    diagnostics = None
                    guided_logits = base_logits
                else:
                    guide_result, elapsed = _timed_call(
                        self.device,
                        lambda: _model_step(
                            self.guide_model,
                            guide_input,
                            attention_mask,
                            position_ids,
                            guide_past,
                        ),
                    )
                    guide_logits, guide_past = guide_result
                    guide_seconds += elapsed
                    current_topk = select_taro_topk(
                        base_logits,
                        guide_logits,
                        top_k=self.top_k,
                    )
                    history_topk.append(current_topk)
                    (lambda_t, diagnostics), elapsed = _timed_call(
                        self.device,
                        lambda: self._router_lambda(
                            spec,
                            current_topk,
                            position=position,
                            history_topk=history_topk,
                            selected_scores=selected_scores,
                        ),
                    )
                    router_seconds += elapsed
                    guided_logits = base_logits + lambda_t * (
                        guide_logits - base_logits
                    )
                if not bool(torch.isfinite(guided_logits).all()):
                    raise FloatingPointError("Guided logits contain NaN or infinity")

                sampling_start = time.perf_counter()
                scores = guided_logits / self.temperature
                if self.do_sample:
                    top_values, top_ids = torch.topk(scores, self.top_k, dim=-1)
                    sampled = torch.multinomial(
                        torch.softmax(top_values, dim=-1), num_samples=1
                    )
                    next_token = top_ids.gather(-1, sampled)
                else:
                    next_token = scores.argmax(dim=-1, keepdim=True)
                _synchronize(self.device)
                sampling_seconds += time.perf_counter() - sampling_start
                token_id = int(next_token.item())
                selected_ids.append(token_id)
                base_logprob = float(
                    torch.log_softmax(base_logits, dim=-1)[0, token_id].item()
                )
                selected_base_logprobs.append(base_logprob)
                selected_scores.append(base_logprob)
                lambda_history.append(float(lambda_t[0, 0].item()))
                if diagnostics is not None:
                    base_entropy_history.append(
                        float(diagnostics["base_entropy"][0].item())
                    )
                    guide_entropy_history.append(
                        float(diagnostics["guide_entropy"][0].item())
                    )
                    js_history.append(float(diagnostics["js"][0].item()))
                    overlap_history.append(
                        float(diagnostics["topk_overlap"][0].item())
                    )
                if token_id == self.tokenizer.eos_token_id:
                    eos_generated = True
                    if self.stop_on_eos:
                        terminated_on_eos = True
                        break

                base_input = next_token
                guide_input = next_token
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=attention_mask.dtype,
                            device=self.device,
                        ),
                    ),
                    dim=-1,
                )
                position_ids = torch.full(
                    (1, 1),
                    attention_mask.shape[1] - 1,
                    dtype=torch.long,
                    device=self.device,
                )

        _synchronize(self.device)
        total_seconds = time.perf_counter() - total_start
        generated_text = self.tokenizer.decode(
            selected_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        conditional_ppl = math.exp(
            -sum(selected_base_logprobs) / len(selected_base_logprobs)
        )
        if not math.isfinite(conditional_ppl):
            raise FloatingPointError("Conditional perplexity is not finite")
        return V2Generation(
            generated_text=generated_text,
            selected_token_ids=selected_ids,
            lambda_history=lambda_history,
            base_entropy_history=base_entropy_history,
            guide_entropy_history=guide_entropy_history,
            js_history=js_history,
            topk_overlap_history=overlap_history,
            base_conditional_perplexity=conditional_ppl,
            eos_generated=eos_generated,
            terminated_on_eos=terminated_on_eos,
            latency={
                "total_generation_time": total_seconds,
                "average_latency_per_token": total_seconds / len(selected_ids),
                "base_model_seconds": base_seconds,
                "guide_model_seconds": guide_seconds,
                "router_seconds": router_seconds,
                "sampling_seconds": sampling_seconds,
                "generated_tokens": len(selected_ids),
            },
        )
