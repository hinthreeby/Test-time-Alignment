"""Learned-router integration for RAD logits processing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor

from router.model import RADTokenRouter


def rad_transform_reward(raw_reward_scores: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    reward_scores = torch.clamp(raw_reward_scores.float(), min=0.0, max=1.0)
    if inverse:
        reward_scores = 1.0 - reward_scores
    return reward_scores


def load_router_for_inference(
    checkpoint_path: Path | str,
    beta_max: float | None = None,
    top_k: int = 20,
    map_location: str | torch.device = "cpu",
) -> RADTokenRouter:
    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if "router_state_dict" in payload:
        config = dict(payload["config"])
        router_config = {
            "beta_max": float(config["beta_max"]),
            "beta_init": config.get("beta_init"),
            "top_k": 20,
            "hidden_dim": 64,
            "eps": 1e-6,
        }
        state_dict = payload["router_state_dict"]
    else:
        router_config = dict(payload["config"])
        state_dict = payload["state_dict"]

    if top_k != int(router_config.get("top_k", 20)):
        raise ValueError(f"Router checkpoint top_k={router_config.get('top_k', 20)} does not match runtime top_k={top_k}")
    if beta_max is not None and abs(float(beta_max) - float(router_config["beta_max"])) > 1e-9:
        raise ValueError(f"Router checkpoint beta_max={router_config['beta_max']} does not match runtime beta_max={beta_max}")

    router = RADTokenRouter(**router_config)
    router.load_state_dict(state_dict, strict=True)
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad = False
    return router


class ConstantBetaRouter(torch.nn.Module):
    """Router test double that returns a fixed beta for every batch row."""

    def __init__(self, beta: float, beta_max: float) -> None:
        super().__init__()
        assert 0 <= beta <= beta_max
        self.beta = float(beta)
        self.beta_max = float(beta_max)

    def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = base_logits.shape[0]
        beta = torch.full((batch, 1), self.beta, dtype=base_logits.dtype, device=base_logits.device)
        gate = torch.full((batch, 1), self.beta / self.beta_max, dtype=base_logits.dtype, device=base_logits.device)
        return beta, gate


class RouterIntegratedLogitsProcessor(LogitsProcessor):
    """Shared top-k RAD processor for fixed beta and learned router modes."""

    def __init__(
        self,
        lm_tokenizer,
        rm_tokenizer,
        reward_model,
        topk: int = 20,
        beta: float = 30.0,
        inverse: bool = False,
        router: torch.nn.Module | None = None,
        save_beta_history: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        if topk != 20:
            raise ValueError(f"Router v1 integration requires topk=20, got {topk}")
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = rm_tokenizer
        self._reward_model = reward_model.eval()
        self._topk = int(topk)
        self._beta = float(beta)
        self._inverse = bool(inverse)
        self._router = router.eval() if router is not None else None
        self._save_beta_history = save_beta_history
        self._step = 0
        self.beta_histories: list[list[dict[str, Any]]] = []
        self.latency_history: list[dict[str, float]] = []
        self._device = torch.device(device) if device is not None else None
        if self._router is not None:
            for parameter in self._router.parameters():
                parameter.requires_grad = False

    @property
    def mode(self) -> str:
        return "learned_router" if self._router is not None else "fixed_beta"

    def _ensure_history_rows(self, batch_size: int) -> None:
        if not self.beta_histories:
            self.beta_histories = [[] for _ in range(batch_size)]

    def _score_candidates(self, candidate_input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = next(self._reward_model.parameters()).device
        with torch.inference_mode():
            output = self._reward_model(
                input_ids=candidate_input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=None,
                use_cache=False,
            )
        if len(output) == 3:
            _, raw_scores, _ = output
        else:
            _, raw_scores = output
        return raw_scores[:, 0].detach().float().to(candidate_input_ids.device)

    def _candidate_reward_scores(self, input_ids: torch.LongTensor, topk_ids: torch.LongTensor) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        batch_size = input_ids.shape[0]
        expanded = input_ids.unsqueeze(1).expand(batch_size, self._topk, input_ids.shape[-1])
        candidate_input_ids = torch.cat([expanded, topk_ids.unsqueeze(-1)], dim=-1)
        flattened = candidate_input_ids.reshape(batch_size * self._topk, -1)
        pad_token_id = self._rm_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._rm_tokenizer.eos_token_id
        attention_mask = flattened.ne(int(pad_token_id)).long()
        raw_reward_scores = self._score_candidates(flattened, attention_mask).reshape(batch_size, self._topk)
        return rad_transform_reward(raw_reward_scores, inverse=self._inverse), time.perf_counter() - start

    def _router_beta(self, topk_scores: torch.Tensor, reward_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        start = time.perf_counter()
        if self._router is None:
            beta = torch.full((topk_scores.shape[0], 1), self._beta, dtype=topk_scores.dtype, device=topk_scores.device)
            gate = torch.ones_like(beta)
        else:
            beta, gate = self._router(topk_scores, reward_scores)
            beta = beta.to(topk_scores.device, dtype=topk_scores.dtype)
            gate = gate.to(topk_scores.device, dtype=topk_scores.dtype)
        return beta, gate, time.perf_counter() - start

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        total_start = time.perf_counter()
        self._ensure_history_rows(scores.shape[0])
        topk_start = time.perf_counter()
        topk_scores, topk_ids = torch.topk(scores, self._topk, dim=-1)
        base_latency = time.perf_counter() - topk_start
        reward_scores, reward_latency = self._candidate_reward_scores(input_ids, topk_ids)
        beta, gate, router_latency = self._router_beta(topk_scores.detach().float(), reward_scores.detach().float())
        guided_topk_scores = topk_scores.float() + beta.float() * reward_scores.float()

        new_scores = torch.full_like(scores, -float("inf"))
        new_scores.scatter_(dim=-1, index=topk_ids, src=guided_topk_scores.to(scores.dtype))
        selected_indices = guided_topk_scores.argmax(dim=-1)
        selected_token_ids = topk_ids.gather(dim=-1, index=selected_indices.unsqueeze(-1)).squeeze(-1)

        if self._save_beta_history:
            probs = torch.softmax(topk_scores.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            reward_std = reward_scores.float().std(dim=-1, unbiased=False)
            reward_gap = reward_scores.float().max(dim=-1).values - reward_scores.float().min(dim=-1).values
            for row in range(scores.shape[0]):
                token_id = int(selected_token_ids[row].item())
                self.beta_histories[row].append(
                    {
                        "step": self._step,
                        "beta": float(beta[row, 0].detach().cpu().item()),
                        "gate": float(gate[row, 0].detach().cpu().item()),
                        "selected_token_id": token_id,
                        "selected_token": self._lm_tokenizer.decode([token_id]),
                        "base_entropy": float(entropy[row].detach().cpu().item()),
                        "reward_mean": float(reward_scores[row].mean().detach().cpu().item()),
                        "reward_std": float(reward_std[row].detach().cpu().item()),
                        "reward_gap": float(reward_gap[row].detach().cpu().item()),
                    }
                )
        self.latency_history.append(
            {
                "step": self._step,
                "base_lm_forward_seconds": base_latency,
                "reward_scoring_seconds": reward_latency,
                "router_forward_seconds": router_latency,
                "total_step_seconds": time.perf_counter() - total_start,
            }
        )
        self._step += 1
        return new_scores


def make_fixed_beta_processor(*args, **kwargs) -> RouterIntegratedLogitsProcessor:
    kwargs["router"] = None
    return RouterIntegratedLogitsProcessor(*args, **kwargs)


def make_learned_router_processor(*args, router: torch.nn.Module, **kwargs) -> RouterIntegratedLogitsProcessor:
    kwargs["router"] = router
    return RouterIntegratedLogitsProcessor(*args, **kwargs)
