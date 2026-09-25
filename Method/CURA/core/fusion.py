from __future__ import annotations

import torch

from Method.CURA.core.features import weighted_disagreement


def categorical_kl(p, q):
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(-1)


def _policies(base_logits, risk_reward, strength, gate):
    p_base = torch.softmax(base_logits.float(), dim=-1)
    p_guided = torch.softmax(base_logits.float() + strength.unsqueeze(-1) * risk_reward, dim=-1)
    p_final = (1.0 - gate.unsqueeze(-1)) * p_base + gate.unsqueeze(-1) * p_guided
    return p_base, p_guided, p_final


def fuse_policies(base_logits, mu, log_var, output, kappa=0.5, disagreement_penalty=0.1,
                  epsilon_kl=0.15, projection_steps=16, disagreement_clip=None):
    weights = output["weights"]
    disagreement, step_disagreement = weighted_disagreement(mu, weights)
    if disagreement_clip is not None:
        disagreement = disagreement.clamp_max(float(disagreement_clip))
        step_disagreement = disagreement.mean(dim=-1)
    adjusted = mu - float(kappa) * torch.sqrt(log_var.exp().clamp_min(1e-8))
    risk_reward = (weights.unsqueeze(1) * adjusted).sum(-1) - float(disagreement_penalty) * disagreement
    strength = output["strength"]
    requested_strength = strength
    gate = output["gate"]
    p_base, p_guided, p_final = _policies(base_logits, risk_reward, strength, gate)
    kl = categorical_kl(p_final, p_base)
    pre_projection_kl = kl
    kl_limit_hit = (
        torch.zeros_like(kl, dtype=torch.bool)
        if epsilon_kl is None else kl > float(epsilon_kl)
    )

    # Project by shrinking reward strength; differentiable unprojected path is used when feasible.
    if epsilon_kl is not None and (kl > epsilon_kl).any():
        low, high = torch.zeros_like(strength), strength.detach().clone()
        for _ in range(projection_steps):
            middle = (low + high) / 2.0
            _, _, candidate = _policies(base_logits, risk_reward, middle, gate)
            feasible = categorical_kl(candidate, p_base) <= epsilon_kl
            low = torch.where(feasible, middle, low)
            high = torch.where(feasible, high, middle)
        strength = torch.where(kl > epsilon_kl, low, strength)
        p_base, p_guided, p_final = _policies(base_logits, risk_reward, strength, gate)
        kl = categorical_kl(p_final, p_base)
    return {
        "probabilities": p_final,
        "base_probabilities": p_base,
        "guided_probabilities": p_guided,
        "risk_reward": risk_reward,
        "disagreement": disagreement,
        "step_disagreement": step_disagreement,
        "kl": kl,
        "pre_projection_kl": pre_projection_kl,
        "kl_limit_hit": kl_limit_hit,
        "requested_strength": requested_strength,
        "projected_strength": strength,
    }
