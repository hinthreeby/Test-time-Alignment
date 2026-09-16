from __future__ import annotations

import time
from pathlib import Path

import torch

from Method.CURA.router.adapters.base import SignalOutput
from Method.MultiSignal.core.cache import load_adapter


class CuraAdapter:
    """Metadata-safe wrapper around the established MultiSignal adapters."""

    def __init__(self, name, objective, project_root: Path, device: torch.device):
        self.name = name
        self.objective = objective
        self.adapter = load_adapter(name, device)
        self.project_root = project_root

    @property
    def device(self):
        return self.adapter.device

    def _score_text_candidates(self, prefix_text, candidate_texts):
        texts = [str(prefix_text) + str(candidate) for candidate in candidate_texts]
        if self.name == "rad" and getattr(self.adapter, "rm", None) is not None:
            encoded = self.adapter.rm_tokenizer(
                texts, padding=True, truncation=True, max_length=self.adapter.max_length, return_tensors="pt"
            ).to(self.device)
            _, scores = self.adapter.rm(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False
            )
            return scores.reshape(-1).float() * self.adapter.scale / 30.0
        if self.name == "cdq" and getattr(self.adapter, "scorer", None) is not None:
            encoded = self.adapter.scorer_tokenizer(
                texts, padding=True, truncation=True, max_length=self.adapter.max_length, return_tensors="pt"
            ).to(self.device)
            return self.adapter.scorer(encoded["input_ids"], encoded["attention_mask"]).reshape(-1).float() * self.adapter.scale
        if self.name == "args" and getattr(self.adapter, "rm", None) is not None:
            encoded = self.adapter.rm_tokenizer(
                texts, padding=True, truncation=True, max_length=self.adapter.max_length, return_tensors="pt"
            ).to(self.device)
            logits = self.adapter.rm(**encoded).logits.float()
            return (logits[:, -1] if logits.size(-1) > 1 else logits[:, 0]).reshape(-1) * self.adapter.scale
        return None

    @torch.inference_mode()
    def score(self, prefix_ids, candidate_ids, candidate_texts=None, prefix_text=None) -> SignalOutput:
        started = time.perf_counter()
        text_scores = self._score_text_candidates(prefix_text, candidate_texts) if prefix_text is not None and candidate_texts is not None else None
        scores = (
            text_scores if text_scores is not None
            else self.adapter.get_signal_scores(prefix_ids, candidate_ids).float()
        ).reshape(1, -1)
        metadata = dict(self.adapter.metadata())
        if candidate_texts is not None:
            signal_tokenizer = (
                getattr(self.adapter, "rm_tokenizer", None)
                or getattr(self.adapter, "scorer_tokenizer", None)
                or getattr(self.adapter, "tokenizer", None)
            )
            mismatches = []
            for candidate_text in candidate_texts:
                encoded = signal_tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
                reconstructed = signal_tokenizer.decode(encoded, skip_special_tokens=True)
                mismatches.append(reconstructed.strip() != str(candidate_text).strip() or not encoded)
            metadata["tokenization_mismatch_rate"] = sum(mismatches) / max(1, len(mismatches))
            metadata["tokenization_mismatch"] = any(mismatches)
            metadata["candidate_mapping"] = "decode_then_retokenize" if text_scores is not None else "shared_token_ids"
        output = SignalOutput(
            name=self.name,
            objective=self.objective,
            scores=scores,
            available=True,
            source=self.adapter.metadata()["source"],
            cost_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )
        output.validate(1, candidate_ids.numel())
        return output


def load_adapters(config, project_root: Path, device: torch.device):
    objectives = config.get("signal_objectives", {})
    return [
        CuraAdapter(name, objectives.get(name, "unknown"), project_root, device)
        for name in config["signals"]
    ]
