import torch


class SafetySignal:
    def __init__(self, safety_model=None):
        self.safety_model = safety_model

    def score(self, prefix_ids, candidate_ids):
        if self.safety_model is None:
            return torch.zeros(candidate_ids.shape[0], device=candidate_ids.device)

        return self.safety_model.score(prefix_ids, candidate_ids)
