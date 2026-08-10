import torch


REWARD_SIGNAL_ORDER = ["token", "prefix", "future", "global"]


class SignalContainer:
    def __init__(self, token_provider, prefix_provider, future_provider, global_provider, uncertainty_provider, safety_provider):
        self.reward_providers = [
            token_provider,
            prefix_provider,
            future_provider,
            global_provider
        ]
        self.uncertainty_provider = uncertainty_provider
        self.safety_provider = safety_provider

    def score(self, prefix_ids, candidate_ids):
        reward_scores = [provider.score(prefix_ids, candidate_ids) for provider in self.reward_providers]
        reward_scores = torch.stack(reward_scores, dim=-1)

        uncertainties = self.uncertainty_provider.score(prefix_ids, candidate_ids, reward_scores)
        safety_scores = self.safety_provider.score(prefix_ids, candidate_ids)

        return {
            "reward_scores": reward_scores,
            "uncertainties": uncertainties,
            "safety_scores": safety_scores
        }
