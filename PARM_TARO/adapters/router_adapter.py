"""Preference-aware Router V2 loader and causal sequence adapter for PARM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from router_v2.candidates import TAROTopKBatch, select_taro_topk
from router_v2.smart_checkpoint import load_smart_checkpoint
from router_v2.smart_model import SmartRouterBatch, SmartTokenRouter


@dataclass
class RouterSequenceState:
    topk_history: list[TAROTopKBatch] = field(default_factory=list)
    selected_base_scores: list[float] = field(default_factory=list)


class SmartRouterLambdaProvider:
    """Expose only Router V2 lambda; PARM routing remains in its own module."""

    def __init__(self, router: SmartTokenRouter, *, preference_dim: int) -> None:
        config = router.config
        if not config.use_preference:
            raise ValueError("PARM-TARO requires a preference-enabled Router V2")
        if config.preference_dim != preference_dim:
            raise ValueError("Router preference_dim does not match PARM alpha")
        self.router = router.eval()
        self.preference_dim = preference_dim
        for parameter in self.router.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device,
        expected_vocab_size: int,
        preference_dim: int,
        tokenizer_semantic_sha256: str,
    ) -> "SmartRouterLambdaProvider":
        router, payload = load_smart_checkpoint(path, map_location=device)
        if router.config.vocab_size != expected_vocab_size:
            raise ValueError(
                "Router vocabulary does not match the PARM base tokenizer"
            )
        metadata = payload["metadata"]
        actual_hash = metadata.get("tokenizer_semantic_sha256")
        if actual_hash != tokenizer_semantic_sha256:
            raise ValueError(
                "Router tokenizer semantic hash is missing or incompatible"
            )
        return cls(router, preference_dim=preference_dim)

    def predict(
        self,
        base_router_values: torch.Tensor,
        guide_router_values: torch.Tensor,
        *,
        position: int,
        preference: torch.Tensor,
        state: RouterSequenceState,
    ) -> torch.Tensor:
        if base_router_values.shape != guide_router_values.shape:
            raise ValueError("Router base/guide values must have matching shapes")
        if base_router_values.ndim != 2 or base_router_values.shape[0] != 1:
            raise ValueError("PARM-TARO decoding currently supports batch size one")
        if preference.shape != (1, self.preference_dim):
            raise ValueError("preference must have shape [1, preference_dim]")
        topk = select_taro_topk(
            base_router_values.detach(),
            guide_router_values.detach(),
            top_k=self.router.config.top_k,
        )
        if self.router.config.use_history:
            state.topk_history.append(topk)
            history = TAROTopKBatch(
                base_token_ids=torch.stack(
                    [item.base_token_ids for item in state.topk_history], dim=1
                ),
                base_logits=torch.stack(
                    [item.base_logits for item in state.topk_history], dim=1
                ),
                reward_token_ids=torch.stack(
                    [item.reward_token_ids for item in state.topk_history], dim=1
                ),
                reward_logits=torch.stack(
                    [item.reward_logits for item in state.topk_history], dim=1
                ),
            )
            selected_scores = torch.tensor(
                [state.selected_base_scores + [0.0]],
                dtype=history.base_logits.dtype,
                device=history.base_logits.device,
            )
            positions = torch.arange(
                len(state.topk_history), device=history.base_logits.device
            ).unsqueeze(0)
            output = self.router.predict_lambda(
                SmartRouterBatch(
                    topk=history,
                    position=positions if self.router.config.use_position else None,
                    selected_score=selected_scores,
                    preference=preference,
                )
            )
            return output.lambda_t[:, -1].detach()
        output = self.router.predict_lambda(
            SmartRouterBatch(
                topk=topk,
                position=(
                    torch.tensor([position], device=topk.base_logits.device)
                    if self.router.config.use_position
                    else None
                ),
                preference=preference,
            )
        )
        return output.lambda_t.detach()

    @staticmethod
    def observe_selected_base_score(
        state: RouterSequenceState,
        score: float,
    ) -> None:
        state.selected_base_scores.append(float(score))
