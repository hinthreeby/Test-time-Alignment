import torch
import torch.nn.functional as F
from typing import Union

def fuse_scores(
    base_scores: torch.Tensor,
    parm_scores: torch.Tensor,
    weight: Union[float, torch.Tensor],
    mode: str = "convex"
) -> torch.Tensor:
    """
    Fuse base model scores and PARM scores.
    
    Args:
        base_scores: Tensor of shape (batch_size, vocab_size) representing base model logits/scores.
        parm_scores: Tensor of shape (batch_size, vocab_size) representing PARM logits/scores.
        weight: Scalar or Tensor of shape (batch_size, 1) representing the intervention weight w_t.
        mode: The fusion mode. 'convex' or 'parm_product'.
            - convex: (1 - weight) * base_scores + weight * parm_scores
            - parm_product: base_log_probs + weight * parm_log_probs (log space of pi_base * pi_parm^w)
            
    Returns:
        Fused scores of shape (batch_size, vocab_size).
    """
    if base_scores.shape != parm_scores.shape:
        raise ValueError(f"Shape mismatch: base_scores {base_scores.shape} != parm_scores {parm_scores.shape}")
        
    if isinstance(weight, torch.Tensor):
        if not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
            raise ValueError("weight tensor must be finite and in [0, 1]")
        if weight.dim() == 1:
            weight = weight.unsqueeze(-1)
        if weight.shape[0] != base_scores.shape[0] and weight.shape[0] != 1:
            raise ValueError(f"Batch size mismatch for weight: {weight.shape} vs {base_scores.shape}")

    elif not 0.0 <= float(weight) <= 1.0:
        raise ValueError("weight must be in [0, 1]")

    if mode == "convex":
        # Direct interpolation of scores (logits or probabilities depending on calibration step before this)
        return (1.0 - weight) * base_scores + weight * parm_scores
        
    elif mode == "parm_product":
        # parm_product: base_log_probs + weight * parm_log_probs
        # Convert logits to log probabilities first
        base_log_probs = F.log_softmax(base_scores, dim=-1)
        parm_log_probs = F.log_softmax(parm_scores, dim=-1)
        return base_log_probs + weight * parm_log_probs
        
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
