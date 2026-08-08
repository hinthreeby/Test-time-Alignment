import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Method.Router.adapters.base import RouterAdapter


class GenARMAdapter(RouterAdapter):
    def load_models(self):
        lm_path = self.project_root / self.config["lm_path"]
        arm_path = self.project_root / self.config["arm_path"]
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.lm_tokenizer = AutoTokenizer.from_pretrained(str(lm_path), local_files_only=True)
        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        self.arm_tokenizer = AutoTokenizer.from_pretrained(str(arm_path), local_files_only=True)
        if self.arm_tokenizer.pad_token_id is None:
            self.arm_tokenizer.pad_token = self.arm_tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(str(lm_path), local_files_only=True, torch_dtype=dtype).to(self.device).eval()
        self.arm = AutoModelForCausalLM.from_pretrained(str(arm_path), local_files_only=True, torch_dtype=dtype).to(self.device).eval()

        if self.lm.config.vocab_size != self.arm.config.vocab_size:
            raise ValueError("Base LM và ARM có vocab_size khác nhau.")

        for model in (self.lm, self.arm):
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
        arm_logits = self.arm(input_ids=prefix_ids).logits[0, -1].float()
        return arm_logits[candidate_ids]

    def guided_scores(self, base_logits, signal_scores, router_value):
        return base_logits.float() + router_value.unsqueeze(-1) * signal_scores.float()

    def append_token(self, prefix_ids, token_id):
        return torch.cat([prefix_ids, token_id.view(1, 1)], dim=1)

    def decode_tokens(self, token_ids):
        return self.lm_tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    @property
    def eos_token_id(self):
        return self.lm_tokenizer.eos_token_id
