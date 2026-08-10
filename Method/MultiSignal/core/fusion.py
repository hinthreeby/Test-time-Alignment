import torch


def normalize_signals(signal_scores, eps=1e-6):
    mean = signal_scores.mean(dim=1, keepdim=True)
    std = signal_scores.std(dim=1, keepdim=True, unbiased=False)
    return (signal_scores.float() - mean) / (std + eps)


def fuse_scores(lm_logits, signal_scores, controller_output):
    normalized = normalize_signals(signal_scores)
    weights = controller_output["weights"].unsqueeze(1)
    strength = controller_output["strength"].unsqueeze(-1)

    fused_reward = (weights * normalized).sum(dim=-1)
    final_scores = lm_logits.float() + strength * fused_reward

    return final_scores, fused_reward
