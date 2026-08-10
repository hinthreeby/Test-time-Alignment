import torch


class UncertaintySignal:
    def __init__(self, mode="disagreement"):
        self.mode = mode

    def score(self, prefix_ids, candidate_ids, reward_scores):
        if self.mode == "disagreement":
            mean = reward_scores.mean(dim=-1, keepdim=True)
            uncertainty = torch.abs(reward_scores - mean)
            return uncertainty

        raise ValueError(f"Unknown uncertainty mode: {self.mode}")
