import torch
import torch.nn as nn
import torch.nn.functional as F


def alpha_zero_loss(policy_logits, value_preds, targets_pi, targets_v, policy_mask):
    v_loss = nn.MSELoss()(value_preds, targets_v)

    masked_logits = policy_logits.masked_fill(~policy_mask, -1e4)

    log_preds = F.log_softmax(masked_logits, dim=1)
    p_loss = -torch.sum(targets_pi * log_preds, dim=1).mean()

    total_loss = v_loss + p_loss
    return total_loss, p_loss, v_loss