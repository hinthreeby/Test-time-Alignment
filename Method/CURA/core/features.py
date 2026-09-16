from __future__ import annotations

import torch


def base_routing_features(base_logits, position=None, prefix_length=None, full_logits=None):
    logits = base_logits.float()
    distribution_logits = full_logits.float() if full_logits is not None else logits
    probs = torch.softmax(distribution_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1, keepdim=True)
    top2 = torch.topk(distribution_logits, min(2, distribution_logits.size(-1)), dim=-1).values
    margin = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
    top_mass = torch.softmax(distribution_logits, -1).topk(logits.size(-1), dim=-1).values.sum(-1, keepdim=True)
    batch = logits.size(0)
    position = torch.zeros(batch, 1, device=logits.device) if position is None else position.float().reshape(batch, 1)
    prefix_length = torch.ones(batch, 1, device=logits.device) if prefix_length is None else prefix_length.float().reshape(batch, 1)
    return torch.cat([
        entropy, margin, top_mass, logits.mean(-1, keepdim=True), logits.std(-1, keepdim=True, unbiased=False),
        logits.max(-1, keepdim=True).values, position / prefix_length.clamp_min(1.0), prefix_length.log1p(),
    ], dim=-1)


def weighted_disagreement(mu: torch.Tensor, weights: torch.Tensor | None = None):
    if weights is None:
        weights = torch.full(
            (mu.size(0), mu.size(-1)), 1.0 / mu.size(-1), device=mu.device, dtype=mu.dtype
        )
    w = weights.unsqueeze(1)
    center = (w * mu).sum(dim=-1, keepdim=True)
    candidate = (w * (mu - center).square()).sum(dim=-1)
    return candidate, candidate.mean(dim=-1)


def controller_features(base_logits, mu, log_var, signal_mask, position=None, prefix_length=None, base_features=None):
    logits = base_logits.float()
    base_features = base_features if base_features is not None else base_routing_features(logits, position, prefix_length)
    _, disagreement = weighted_disagreement(mu)
    uncertainty = log_var.exp().mean(dim=1)
    return torch.cat([
        base_features,
        mu.mean(dim=1), mu.std(dim=1, unbiased=False), uncertainty,
        disagreement.unsqueeze(-1), signal_mask.float(),
    ], dim=-1)
