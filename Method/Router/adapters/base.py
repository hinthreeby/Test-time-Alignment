from abc import ABC, abstractmethod
import torch


class RouterAdapter(ABC):
    def __init__(self, config, project_root, device):
        self.config = config
        self.project_root = project_root
        self.device = device

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def encode_prompt(self, prompt):
        pass

    @abstractmethod
    def encode_response(self, response):
        pass

    @abstractmethod
    def get_candidates(self, prefix_ids):
        pass

    @abstractmethod
    def get_signal_scores(self, prefix_ids, candidate_ids):
        pass

    @abstractmethod
    def guided_scores(self, base_logits, signal_scores, router_value):
        pass

    @abstractmethod
    def append_token(self, prefix_ids, token_id):
        pass

    @abstractmethod
    def decode_tokens(self, token_ids):
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        pass

    def build_features(self, base_logits, signal_scores):
        base = self.normalize(base_logits)
        signal = self.normalize(signal_scores)
        return torch.cat([base, signal], dim=-1)

    @staticmethod
    def normalize(x, eps=1e-6):
        x = x.float()
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True, unbiased=False) + eps)
