"""Independent local sentiment scoring for Stage 6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentScorer:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device,
        target_label: str = "POSITIVE",
        max_length: int = 256,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        id_to_label = {
            int(key): str(value)
            for key, value in dict(self.model.config.id2label).items()
        }
        label_to_id = {label: index for index, label in id_to_label.items()}
        if target_label not in label_to_id:
            raise ValueError(
                f"Target label {target_label!r} missing from classifier config"
            )
        self.id_to_label = id_to_label
        self.target_label = target_label
        self.target_label_id = int(label_to_id[target_label])

    def score(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.float()
            probabilities = torch.softmax(logits, dim=-1)
        outputs = []
        for row in range(len(texts)):
            predicted_id = int(probabilities[row].argmax().item())
            outputs.append(
                {
                    "classifier_sentiment_success": int(
                        predicted_id == self.target_label_id
                    ),
                    "classifier_target_probability": float(
                        probabilities[row, self.target_label_id].item()
                    ),
                    "classifier_predicted_label": self.id_to_label[predicted_id],
                    "classifier_predicted_label_id": predicted_id,
                }
            )
        return outputs

