"""
PARMAdapter — wraps PBLoRA-based PARM model.

Loads the PARM adapter directly via the custom peft library (PBLoRA type).
Sets user preference by updating `pref_vec` ParameterDict in each attention
projection layer — no config file copy needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# !! Must insert custom peft BEFORE any other import that loads peft !!
_PEFT_SRC = Path(__file__).resolve().parents[4] / "PARM" / "peft" / "src"
if str(_PEFT_SRC) not in sys.path:
    sys.path.insert(0, str(_PEFT_SRC))

# Remove any already-cached conda peft so our custom one takes over
for _key in list(sys.modules.keys()):
    if _key == "peft" or _key.startswith("peft."):
        del sys.modules[_key]

from peft import PeftModel  # noqa: E402 — intentionally after sys.path fix

from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_adapter import ModelOutputState


def preference_to_pblora(alpha: List[float]) -> List[float]:
    """Map public [helpfulness, harmlessness] to PBLoRA's stored order."""
    if len(alpha) != 2:
        raise ValueError(f"Expected K=2 objectives, got {len(alpha)}")
    values = [float(value) for value in alpha]
    if any(value < 0 for value in values) or abs(sum(values) - 1.0) >= 1e-5:
        raise ValueError(f"Preference must be a non-negative simplex vector: {alpha}")
    return [values[1], values[0]]


class PARMAdapter:
    """
    Wraps the PBLoRA PARM model.

    Interface (per CAP-PARM spec):
      set_preference(alpha)           -> configures PBLoRA pref_vec
      prefill(input_ids, alpha, ...)  -> ModelOutputState
      step(token_id, alpha, cache)    -> ModelOutputState
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_alpha: Optional[List[float]] = None

        print(f"  [PARMAdapter] Loading base model from {base_model_path} ...")
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=self.device,
            local_files_only=True,
        )

        print(f"  [PARMAdapter] Attaching PBLoRA adapter from {adapter_path} ...")
        self.model: PeftModel = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            local_files_only=True,
        )
        self.model.eval()

        # Cache all modules that own a pref_vec ParameterDict
        self._pref_modules = [
            module
            for _, module in self.model.named_modules()
            if hasattr(module, "pref_vec")
        ]
        print(f"  [PARMAdapter] Found {len(self._pref_modules)} PBLoRA layers with pref_vec.")

    # ------------------------------------------------------------------
    # Preference setting
    # ------------------------------------------------------------------

    def set_preference(self, alpha: List[float]) -> None:
        """
        Validate alpha and update pref_vec in all PBLoRA layers.

        The public order is [helpfulness, harmlessness]. The recovered PBLoRA
        checkpoint stores pref_vec as [harmlessness, helpfulness].

        alpha must lie on the K-simplex:
          all(a >= 0) and sum(alpha) == 1.0
        """
        internal_alpha = preference_to_pblora(alpha)

        if self.current_alpha == list(alpha):
            return  # No change needed

        alpha_tensor = torch.tensor(internal_alpha, dtype=torch.float32, device=self.device)
        for module in self._pref_modules:
            module.pref_vec["default"].data.copy_(alpha_tensor)
        self.current_alpha = list(alpha)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        alpha: List[float],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutputState:
        """Return PARM logits and KV cache after processing the full prompt."""
        self.set_preference(alpha)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        next_scores = outputs.logits[:, -1, :]
        return ModelOutputState(
            next_scores=next_scores,
            hidden=None,
            cache=outputs.past_key_values,
        )

    @torch.no_grad()
    def step(
        self,
        token_id: torch.Tensor,
        alpha: List[float],
        past_key_values: Any,
    ) -> ModelOutputState:
        """Advance PARM cache by one generated token."""
        self.set_preference(alpha)
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(-1)
        outputs = self.model(
            input_ids=token_id,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )
        next_scores = outputs.logits[:, -1, :]
        return ModelOutputState(
            next_scores=next_scores,
            hidden=None,
            cache=outputs.past_key_values,
        )

    @torch.no_grad()
    def probe_objectives(
        self,
        token_id: torch.Tensor,
        objective_vectors: List[List[float]],
        past_key_values: Any,
    ) -> List[torch.Tensor]:
        """
        Oracle-only one-hot objective probes.
        Probes the next token scores for multiple preference vectors.
        Reuses past_key_values for speed (approximation).
        """
        scores = []
        original_alpha = list(self.current_alpha) if self.current_alpha else None
        
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(-1)
            
        for alpha in objective_vectors:
            self.set_preference(alpha)
            outputs = self.model(
                input_ids=token_id,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
            scores.append(outputs.logits[:, -1, :].clone())
            
        # Restore original alpha
        if original_alpha:
            self.set_preference(original_alpha)
            
        return scores

    @torch.no_grad()
    def probe_prefix(
        self,
        input_ids: torch.Tensor,
        objective_vectors: List[List[float]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return correct next-token logits for each alpha by replaying the prefix.

        PBLoRA changes the projections that produced the KV cache, so a cache
        created under one preference is not reused for another preference.
        This method is intentionally expensive and is for oracle probing only.
        """
        scores: List[torch.Tensor] = []
        original_alpha = list(self.current_alpha) if self.current_alpha else None
        for alpha in objective_vectors:
            self.set_preference(alpha)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            scores.append(outputs.logits[:, -1, :].clone())
        if original_alpha is not None:
            self.set_preference(original_alpha)
        return scores
