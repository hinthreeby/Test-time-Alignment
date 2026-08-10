import torch


def reward_disagreement(signal_scores):
    normalized = signal_scores.float()
    mean = normalized.mean(dim=1, keepdim=True)
    std = normalized.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (normalized - mean) / std
    return normalized.std(dim=-1, unbiased=False).mean(dim=-1)


def uncertainty_level(uncertainties):
    return uncertainties.float().mean(dim=(1, 2))


def difficulty_score(signal_scores, uncertainties, alpha=0.5):
    disagreement = reward_disagreement(signal_scores)
    uncertainty = uncertainty_level(uncertainties)
    return disagreement + alpha * uncertainty
