import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ModelOutputState:
    next_scores: torch.Tensor
    hidden: Optional[torch.Tensor]
    cache: Any

class BaseModelAdapter:
    def __init__(self, model_name_or_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            torch_dtype=dtype,
            local_files_only=True
        )
        self.model.eval()

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> ModelOutputState:
        """Return next-token logits, hidden features, and KV cache."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True
        )
        next_scores = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None
        return ModelOutputState(
            next_scores=next_scores,
            hidden=hidden,
            cache=outputs.past_key_values
        )

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, past_key_values: Any) -> ModelOutputState:
        """Advance one token and return logits/features/new cache."""
        # token_id shape should be (batch_size, 1)
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(-1)
            
        outputs = self.model(
            input_ids=token_id,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        next_scores = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None
        return ModelOutputState(
            next_scores=next_scores,
            hidden=hidden,
            cache=outputs.past_key_values
        )
