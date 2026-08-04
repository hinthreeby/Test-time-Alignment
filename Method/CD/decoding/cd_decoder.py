from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from models.prefix_scorer import PrefixScorer


@dataclass
class CDGenerationConfig:
    lambda_weight: float = 1.0
    top_k: int = 20
    max_new_tokens: int = 64
    temperature: float = 1.0
    method: Literal[
        "greedy",
        "sample",
    ] = "greedy"


class ControlledDecoder:
    def __init__(
        self,
        base_model: PreTrainedModel,
        base_tokenizer: PreTrainedTokenizerBase,
        prefix_scorer: PrefixScorer,
        scorer_tokenizer: PreTrainedTokenizerBase,
        base_device: torch.device,
        scorer_device: torch.device,
    ):
        self.base_model = base_model.eval()
        self.base_tokenizer = base_tokenizer

        self.prefix_scorer = prefix_scorer.eval()
        self.scorer_tokenizer = scorer_tokenizer

        self.base_device = base_device
        self.scorer_device = scorer_device

    @torch.inference_mode()
    def _score_texts(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        encoded = self.scorer_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.scorer_device)

        values = (
            self.prefix_scorer
            .score_last_token(
                input_ids=encoded["input_ids"],
                attention_mask=(
                    encoded["attention_mask"]
                ),
            )
        )

        return values

    @torch.inference_mode()
    def generate_tokenwise(
        self,
        prompt: str,
        config: CDGenerationConfig,
    ) -> str:
        encoded = self.base_tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.base_device)

        prompt_length = (
            encoded["input_ids"].shape[1]
        )

        generated = encoded["input_ids"]

        attention_mask = (
            encoded["attention_mask"]
        )

        past_key_values = None
        next_input_ids = generated

        for _ in range(
            config.max_new_tokens
        ):
            outputs = self.base_model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = (
                outputs.past_key_values
            )

            base_logits = (
                outputs.logits[:, -1, :]
                / max(
                    config.temperature,
                    1e-6,
                )
            )

            k = min(
                config.top_k,
                base_logits.size(-1),
            )

            candidate_logits, candidate_ids = (
                torch.topk(
                    base_logits,
                    k=k,
                    dim=-1,
                )
            )

            repeated_prefix = generated.repeat(
                k,
                1,
            )

            candidate_sequences = torch.cat(
                [
                    repeated_prefix,
                    candidate_ids[0].unsqueeze(1),
                ],
                dim=1,
            )

            candidate_texts = (
                self.base_tokenizer
                .batch_decode(
                    candidate_sequences,
                    skip_special_tokens=True,
                )
            )

            values = self._score_texts(
                candidate_texts
            ).to(self.base_device)

            aligned_logits = (
                candidate_logits[0]
                + config.lambda_weight
                * values
            )

            if config.method == "greedy":
                selected_index = torch.argmax(
                    aligned_logits
                )
            else:
                probabilities = torch.softmax(
                    aligned_logits,
                    dim=-1,
                )

                selected_index = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )[0]

            selected_token = candidate_ids[
                0,
                selected_index,
            ].view(1, 1)

            generated = torch.cat(
                [
                    generated,
                    selected_token,
                ],
                dim=1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (
                            attention_mask.size(0),
                            1,
                        ),
                        dtype=(
                            attention_mask.dtype
                        ),
                        device=(
                            attention_mask.device
                        ),
                    ),
                ],
                dim=1,
            )

            next_input_ids = selected_token

            if (
                selected_token.item()
                == self.base_tokenizer.eos_token_id
            ):
                break

        continuation_ids = generated[
            :,
            prompt_length:,
        ]

        response = self.base_tokenizer.decode(
            continuation_ids[0],
            skip_special_tokens=True,
        )

        return response.strip()

    @torch.inference_mode()
    def generate_blockwise(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        block_size: int = 8,
        num_candidates: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> str:
        encoded = self.base_tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.base_device)

        generated = encoded["input_ids"]

        prompt_length = generated.size(1)

        while (
            generated.size(1)
            - prompt_length
            < max_new_tokens
        ):
            generated_length = (
                generated.size(1)
                - prompt_length
            )

            remaining = (
                max_new_tokens
                - generated_length
            )

            current_block_size = min(
                block_size,
                remaining,
            )

            candidates = self.base_model.generate(
                input_ids=generated,
                attention_mask=torch.ones_like(
                    generated
                ),
                max_new_tokens=current_block_size,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=(
                    num_candidates
                ),
                pad_token_id=(
                    self.base_tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.base_tokenizer.eos_token_id
                ),
            )

            candidate_texts = (
                self.base_tokenizer
                .batch_decode(
                    candidates,
                    skip_special_tokens=True,
                )
            )

            values = self._score_texts(
                candidate_texts
            )

            best_index = int(
                torch.argmax(values).item()
            )

            generated = candidates[
                best_index
            ].unsqueeze(0)

            if (
                generated[0, -1].item()
                == self.base_tokenizer.eos_token_id
            ):
                break

        continuation_ids = generated[
            :,
            prompt_length:,
        ]

        response = self.base_tokenizer.decode(
            continuation_ids[0],
            skip_special_tokens=True,
        )

        return response.strip()