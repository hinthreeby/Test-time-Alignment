"""Explicit base/PARM vocabulary alignment without token remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PARMTokenAlignment:
    """Represent PARM's shared-tokenizer, base-vocabulary prefix contract."""

    base_vocab_size: int
    guide_vocab_size: int
    tokenizer_vocab_size: int
    base_token_ids_are_prefix: bool

    @classmethod
    def validate(
        cls,
        tokenizer: Any,
        *,
        base_vocab_size: int,
        guide_vocab_size: int,
    ) -> "PARMTokenAlignment":
        tokenizer_vocab_size = len(tokenizer)
        if base_vocab_size <= 1 or guide_vocab_size <= 1:
            raise ValueError("Model vocabularies must contain at least two tokens")
        if tokenizer_vocab_size != base_vocab_size:
            raise ValueError(
                "PARM-TARO requires the shared tokenizer length to equal the "
                "base model vocabulary"
            )
        if guide_vocab_size < base_vocab_size:
            raise ValueError("Guide vocabulary cannot omit base token IDs")
        vocabulary = tokenizer.get_vocab()
        token_ids = set(int(value) for value in vocabulary.values())
        expected = set(range(base_vocab_size))
        if token_ids != expected:
            raise ValueError(
                "Shared tokenizer IDs must be a dense [0, base_vocab_size) prefix"
            )
        return cls(
            base_vocab_size=base_vocab_size,
            guide_vocab_size=guide_vocab_size,
            tokenizer_vocab_size=tokenizer_vocab_size,
            base_token_ids_are_prefix=True,
        )

    def align_guide_logits(self, guide_logits: torch.Tensor) -> torch.Tensor:
        if guide_logits.shape[-1] != self.guide_vocab_size:
            raise ValueError(
                "Guide logits do not match the validated guide vocabulary"
            )
        aligned = guide_logits[..., : self.base_vocab_size]
        if aligned.shape[-1] != self.base_vocab_size:
            raise RuntimeError("Guide vocabulary alignment failed")
        return aligned

    def validate_base_logits(self, base_logits: torch.Tensor) -> None:
        if base_logits.shape[-1] != self.base_vocab_size:
            raise ValueError("Base logits do not match the validated vocabulary")
