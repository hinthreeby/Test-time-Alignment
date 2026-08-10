import torch


BUDGET_LEVELS = [1, 4, 16]


def select_budget(controller_output):
    indexes = torch.argmax(controller_output["budget_logits"], dim=-1)
    return torch.tensor([BUDGET_LEVELS[i] for i in indexes.tolist()], device=indexes.device)
