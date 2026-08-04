"""
Adapted from dpo_utils.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import DataCollatorForSeq2Seq

torch.backends.cuda.matmul.allow_tf32 = True


def arm_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    beta: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the arm loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model
            for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model
            for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something
            in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the arm loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards
            for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # ref_logratios = reference_chosen_logps - reference_rejected_logps

    # if reference_free:
    #     ref_logratios = 0

    # logits = pi_logratios - ref_logratios
    logits = pi_logratios

    losses = -F.logsigmoid(beta * logits)
    # chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    # rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    chosen_rewards = beta * (policy_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards