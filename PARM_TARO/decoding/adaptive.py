"""Autoregressive PARM decoding with a Router V2 adaptive guidance scale."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from PARM_TARO.adapters.parm_adapter import (
    freeze_for_inference,
    named_preference_to_parm,
    set_parm_preference,
)
from PARM_TARO.adapters.router_adapter import RouterSequenceState
from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.decoding.static_regression import parm_guided_logprobs


@dataclass(frozen=True)
class ParmTaroGeneration:
    generated_text: str
    selected_token_ids: list[int]
    lambda_history: list[float]
    selected_base_logprobs: list[float]
    base_conditional_perplexity: float
    preference: list[float]
    eos_generated: bool
    terminated_on_eos: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    return logits, getattr(output, "past_key_values", None)


class ParmTaroDecoder:
    """Keep PARM models frozen and route only the full-vocabulary guide scale."""

    def __init__(
        self,
        *,
        base_model: Any,
        guide_model: Any,
        tokenizer: Any,
        alignment: PARMTokenAlignment,
        lambda_provider: Any,
        device: torch.device,
        preference_dim: int,
        top_k: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        stop_on_eos: bool = True,
    ) -> None:
        if preference_dim <= 0:
            raise ValueError("preference_dim must be positive")
        if top_k <= 0 or max_new_tokens <= 0 or temperature <= 0.0:
            raise ValueError("Invalid PARM-TARO decoding controls")
        self.base_model = base_model
        self.guide_model = guide_model
        self.tokenizer = tokenizer
        self.alignment = alignment
        self.lambda_provider = lambda_provider
        self.device = device
        self.preference_dim = preference_dim
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.stop_on_eos = stop_on_eos
        freeze_for_inference(base_model, guide_model)

    def generate(
        self,
        prompt: str,
        *,
        preference: torch.Tensor,
        router_preference: torch.Tensor | None = None,
        seed: int,
    ) -> ParmTaroGeneration:
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        values = preference.detach().to(device=self.device, dtype=torch.float32)
        if values.shape == (self.preference_dim,):
            values = values.unsqueeze(0)
        if values.shape != (1, self.preference_dim):
            raise ValueError("preference must have shape [preference_dim] or [1, d]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("preference must be finite")
        router_values = values
        if router_preference is not None:
            router_values = router_preference.detach().to(
                device=self.device, dtype=torch.float32
            )
            if router_values.shape == (self.preference_dim,):
                router_values = router_values.unsqueeze(0)
            if router_values.shape != (1, self.preference_dim):
                raise ValueError(
                    "router_preference must have shape [preference_dim] or [1, d]"
                )
            if not bool(torch.isfinite(router_values).all()):
                raise ValueError("router_preference must be finite")
        set_parm_preference(
            self.guide_model,
            named_preference_to_parm(values[0]),
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get(
            "attention_mask", torch.ones_like(input_ids)
        ).to(self.device)
        if input_ids.shape != attention_mask.shape or input_ids.shape[1] == 0:
            raise ValueError("Prompt tokenization produced invalid tensors")
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        base_input = input_ids
        guide_input = input_ids
        base_past = None
        guide_past = None
        router_state = RouterSequenceState()
        selected_ids: list[int] = []
        selected_base_logprobs: list[float] = []
        lambda_history: list[float] = []
        eos_generated = False
        terminated_on_eos = False

        with torch.inference_mode():
            for position in range(self.max_new_tokens):
                base_logits, base_past = _model_step(
                    self.base_model,
                    base_input,
                    attention_mask,
                    position_ids,
                    base_past,
                )
                guide_logits, guide_past = _model_step(
                    self.guide_model,
                    guide_input,
                    attention_mask,
                    position_ids,
                    guide_past,
                )
                self.alignment.validate_base_logits(base_logits)
                guide_logits = self.alignment.align_guide_logits(guide_logits)
                base_logprobs = F.log_softmax(base_logits, dim=-1)
                guide_logprobs = F.log_softmax(guide_logits, dim=-1)
                lambda_t = self.lambda_provider.predict(
                    base_logprobs,
                    guide_logprobs,
                    position=position,
                    preference=router_values,
                    state=router_state,
                )
                if lambda_t.shape != (1, 1):
                    raise ValueError("Router lambda must have shape [1, 1]")
                guided_logprobs = parm_guided_logprobs(
                    base_logits,
                    guide_logits,
                    lambda_t,
                )
                if not bool(torch.isfinite(guided_logprobs).all()):
                    raise FloatingPointError("Guided log probabilities are invalid")
                scores = guided_logprobs / self.temperature
                if self.do_sample:
                    count = min(self.top_k, scores.shape[-1])
                    top_values, top_ids = torch.topk(scores, count, dim=-1)
                    sampled = torch.multinomial(
                        torch.softmax(top_values, dim=-1), num_samples=1
                    )
                    next_token = top_ids.gather(-1, sampled)
                else:
                    next_token = scores.argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                base_score = float(base_logprobs[0, token_id].item())
                selected_ids.append(token_id)
                selected_base_logprobs.append(base_score)
                lambda_history.append(float(lambda_t.item()))
                self.lambda_provider.observe_selected_base_score(
                    router_state, base_score
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
        if not selected_ids:
            raise RuntimeError("PARM-TARO generation produced no tokens")
        perplexity = math.exp(
            -sum(selected_base_logprobs) / len(selected_base_logprobs)
        )
        if not math.isfinite(perplexity):
            raise FloatingPointError("Base conditional perplexity is invalid")
        return ParmTaroGeneration(
            generated_text=self.tokenizer.decode(
                selected_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            selected_token_ids=selected_ids,
            lambda_history=lambda_history,
            selected_base_logprobs=selected_base_logprobs,
            base_conditional_perplexity=perplexity,
            preference=[float(value) for value in values[0].tolist()],
            eos_generated=eos_generated,
            terminated_on_eos=terminated_on_eos,
        )
