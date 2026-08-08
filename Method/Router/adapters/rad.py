import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Method.Router.adapters.base import RouterAdapter
from Method.RAD.reward_modeling.reward_model import GPT2RewardModel


class RADAdapter(RouterAdapter):
    def load_models(self):
        lm_path = self.project_root / self.config["lm_path"]
        rm_base_path = self.project_root / self.config["rm_base_path"]
        rm_path = self.project_root / self.config["rm_path"]

        self.lm_tokenizer = AutoTokenizer.from_pretrained(str(lm_path), local_files_only=True)
        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(str(lm_path), local_files_only=True, torch_dtype=dtype).to(self.device).eval()

        self.rm_tokenizer = AutoTokenizer.from_pretrained(str(rm_path), local_files_only=True)
        if self.rm_tokenizer.pad_token_id is None:
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token

        self.rm = GPT2RewardModel(reward_model_name=str(rm_base_path), out_features=1, loss_fn="cumulative_mse")
        self.rm.load_state_dict(torch.load(rm_path / "pytorch_model.bin", map_location="cpu"), strict=True)
        self.rm = self.rm.to(self.device).eval()

        for model in (self.lm, self.rm):
            for p in model.parameters():
                p.requires_grad = False

    def encode_prompt(self, prompt):
        return self.lm_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    def encode_response(self, response):
        return self.lm_tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

    @torch.inference_mode()
    def get_candidates(self, prefix_ids):
        logits = self.lm(input_ids=prefix_ids).logits[0, -1]
        top_logits, candidate_ids = torch.topk(logits, k=self.config["top_k"])
        return logits, top_logits, candidate_ids

    @torch.inference_mode()
    def get_signal_scores(self, prefix_ids, candidate_ids):
        repeated = prefix_ids.repeat(candidate_ids.shape[0], 1)
        sequences = torch.cat([repeated, candidate_ids.unsqueeze(1)], dim=1)
        texts = self.lm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        inputs = self.rm_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        rewards = self.rm(**inputs, labels=None, use_cache=False)[1][:, 0].float()
        return torch.clamp(rewards, 0.0, 1.0)

    def guided_scores(self, base_logits, signal_scores, router_value):
        return base_logits.float() + router_value.unsqueeze(-1) * signal_scores.float()

    def append_token(self, prefix_ids, token_id):
        return torch.cat([prefix_ids, token_id.view(1, 1)], dim=1)

    def decode_tokens(self, token_ids):
        return self.lm_tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    @property
    def eos_token_id(self):
        return self.lm_tokenizer.eos_token_id
