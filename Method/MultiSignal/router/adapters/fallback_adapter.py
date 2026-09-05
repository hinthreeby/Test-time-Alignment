from __future__ import annotations

from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class LocalFallbackAdapter:
    """Minimal adapter used to make MultiSignal runnable in this workspace.

    It implements the interface expected by MultiSignal/core/cache.py:
    - load_models()
    - encode_prompt(prompt)
    - encode_response(response)
    - get_signal_scores(prefix_ids, candidate_ids)

    The score is computed from the base language model logits, with a small
    method-specific scaling. This keeps the pipeline executable even when the
    full Router package is absent.
    """

    def __init__(self, config, project_root: Path, device: torch.device):
        self.config = config
        self.project_root = Path(project_root)
        self.device = torch.device(device)
        self.model_name = config.get("model_name", "gpt2-large")
        self.model_path = self.project_root / "models" / self.model_name
        self.max_length = int(config.get("max_length", 256))
        self.source = "fallback_lm_logits"

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = None

    def load_models(self):
        return self

    def ensure_lm(self):
        if self.lm is None:
            self.lm = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ).to(self.device).eval()
        return self.lm

    def encode_prompt(self, prompt):
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            prompt_text = str(prompt or "")
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        return inputs["input_ids"].to(self.device)

    def encode_response(self, response):
        if isinstance(response, str):
            response_text = response
        else:
            response_text = str(response or "")
        inputs = self.tokenizer(response_text, return_tensors="pt", truncation=True)
        return inputs["input_ids"].to(self.device)

    def _candidate_logits(self, prefix_ids):
        with torch.inference_mode():
            logits = self.ensure_lm()(input_ids=prefix_ids).logits[:, -1, :]
        return logits.float()

    def next_logits(self, prefix_ids):
        return self._candidate_logits(prefix_ids)

    def get_signal_scores(self, prefix_ids, candidate_ids):
        prefix_ids = prefix_ids.to(self.device)
        candidate_ids = candidate_ids.to(self.device)
        logits = self._candidate_logits(prefix_ids)
        gathered = logits.gather(dim=-1, index=candidate_ids.unsqueeze(0).expand(logits.size(0), -1))
        return gathered.squeeze(0).float()

    def candidate_texts(self, prefix_ids, candidate_ids):
        candidates = torch.cat(
            [prefix_ids.repeat(candidate_ids.numel(), 1), candidate_ids.view(-1, 1).to(prefix_ids.device)],
            dim=1,
        )
        return self.tokenizer.batch_decode(candidates, skip_special_tokens=True)

    def metadata(self):
        return {"name": self.config.get("name"), "source": self.source, "model": str(self.model_path)}


class RADAdapter(LocalFallbackAdapter):
    def __init__(self, config, project_root: Path, device):
        super().__init__(config, project_root, device)
        self.scale = float(config.get("beta", 30.0))
        self.rm = None
        self.rm_tokenizer = None

        rm_base = self.project_root / "models" / config.get("rm_base_model", "gpt2-small")
        rm_path = self.project_root / "models" / config.get("rm_checkpoint", "rad_rm_sentiment")
        checkpoint = rm_path / "pytorch_model.bin"
        if rm_base.exists() and checkpoint.exists():
            from Method.RAD.reward_modeling.reward_model import GPT2RewardModel

            self.rm_tokenizer = AutoTokenizer.from_pretrained(str(rm_path), local_files_only=True)
            if self.rm_tokenizer.pad_token is None:
                self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
            self.rm_tokenizer.padding_side = "right"

            self.rm = GPT2RewardModel(str(rm_base), out_features=1, loss_fn="cumulative_mse")
            state_dict = torch.load(str(checkpoint), map_location="cpu")
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.endswith(".attn.bias") and not key.endswith(".attn.masked_bias")
            }
            self.rm.load_state_dict(state_dict, strict=True)
            self.rm.to(self.device).eval()
            self.source = "rad_reward_model"

    def get_signal_scores(self, prefix_ids, candidate_ids):
        if self.rm is not None:
            texts = self.candidate_texts(prefix_ids.to(self.device), candidate_ids.to(self.device))
            inputs = self.rm_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            _, scores = self.rm(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            return scores.view(-1).float() * self.scale / 30.0

        base = super().get_signal_scores(prefix_ids, candidate_ids)
        return base * self.scale / 100.0


class GenARMAdapter(LocalFallbackAdapter):
    def __init__(self, config, project_root: Path, device):
        super().__init__(config, project_root, device)
        self.scale = float(config.get("alpha", 1.0))
        self.arm = None

        arm_path = self.project_root / "models" / config.get("arm_model", "genarm-gpt2-medium-hh")
        if arm_path.exists():
            self.arm = AutoModelForCausalLM.from_pretrained(
                str(arm_path),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ).to(self.device).eval()
            self.source = "genarm_autoregressive_rm"

    def get_signal_scores(self, prefix_ids, candidate_ids):
        if self.arm is not None:
            logits = self.arm(input_ids=prefix_ids.to(self.device)).logits[:, -1, :].float()
            ids = candidate_ids.to(self.device).unsqueeze(0).expand(logits.size(0), -1)
            return logits.gather(dim=-1, index=ids).squeeze(0).float() * self.scale

        base = super().get_signal_scores(prefix_ids, candidate_ids)
        return base * self.scale


class CDQAdapter(LocalFallbackAdapter):
    def __init__(self, config, project_root: Path, device):
        super().__init__(config, project_root, device)
        self.scale = float(config.get("alpha", 0.5))
        self.scorer = None
        self.scorer_tokenizer = None

        cd_model_dir = self.project_root / "Method" / "CD" / "models"
        if str(cd_model_dir) not in sys.path:
            sys.path.insert(0, str(cd_model_dir))

        backbone = self.project_root / "models" / config.get("scorer_backbone", "gpt2-small")
        checkpoint = self.project_root / config.get("scorer_checkpoint", "Method/CD/checkpoints/cd_q.pt")
        if backbone.exists() and checkpoint.exists():
            from prefix_scorer import PrefixScorer

            self.scorer_tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
            if self.scorer_tokenizer.pad_token is None:
                self.scorer_tokenizer.pad_token = self.scorer_tokenizer.eos_token
            self.scorer_tokenizer.padding_side = "right"
            self.scorer = PrefixScorer(str(backbone)).to(self.device)
            self.scorer.load_state_dict(torch.load(str(checkpoint), map_location=self.device))
            self.scorer.eval()
            self.source = "cd_prefix_value_scorer"

    def get_signal_scores(self, prefix_ids, candidate_ids):
        if self.scorer is not None:
            texts = self.candidate_texts(prefix_ids.to(self.device), candidate_ids.to(self.device))
            inputs = self.scorer_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            return self.scorer(inputs["input_ids"], inputs["attention_mask"]).float() * self.scale

        base = super().get_signal_scores(prefix_ids, candidate_ids)
        return (base - base.mean()) * self.scale


class ARGSAdapter(LocalFallbackAdapter):
    def __init__(self, config, project_root: Path, device):
        super().__init__(config, project_root, device)
        self.scale = float(config.get("alpha", 0.5))
        self.rm = None
        self.rm_tokenizer = None

        rm_path = self.project_root / "models" / config.get("sequence_rm", "sentiment-roberta-large-english")
        if rm_path.exists():
            self.rm_tokenizer = AutoTokenizer.from_pretrained(str(rm_path), local_files_only=True)
            self.rm = AutoModelForSequenceClassification.from_pretrained(
                str(rm_path),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ).to(self.device).eval()
            self.source = "args_outcome_sequence_rm"

    def get_signal_scores(self, prefix_ids, candidate_ids):
        if self.rm is not None:
            texts = self.candidate_texts(prefix_ids.to(self.device), candidate_ids.to(self.device))
            inputs = self.rm_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.rm(**inputs).logits.float()
            scores = logits[:, -1] if logits.size(-1) > 1 else logits.squeeze(-1)
            return scores.view(-1) * self.scale

        base = super().get_signal_scores(prefix_ids, candidate_ids)
        return base * self.scale
