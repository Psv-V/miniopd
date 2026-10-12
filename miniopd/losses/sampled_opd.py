"""Sampled-token k1 advantage and policy-gradient surrogate."""

from __future__ import annotations

import torch

from miniopd.losses.masking import masked_mean


def sampled_token_advantage(
    teacher_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    log_prob_min_clamp: float = -10.0,
    loss_max_clamp: float = 10.0,
) -> torch.Tensor:
    """Compute stop-gradient ``teacher_logp - old_student_logp`` on sampled tokens."""
    if teacher_log_probs.shape != old_student_log_probs.shape:
        raise ValueError("teacher and old-student log-prob shapes must match")
    if teacher_log_probs.shape != response_mask.shape:
        raise ValueError("log-probs and response_mask shapes must match")
    if log_prob_min_clamp > 0:
        raise ValueError("log_prob_min_clamp must be non-positive")
    if loss_max_clamp <= 0:
        raise ValueError("loss_max_clamp must be positive")

    teacher = teacher_log_probs.detach().float().clamp_min(log_prob_min_clamp)
    old_student = old_student_log_probs.detach().float().clamp_min(log_prob_min_clamp)
    advantage = (teacher - old_student).clamp(min=-loss_max_clamp, max=loss_max_clamp)
    advantage = advantage.masked_fill(~response_mask.bool(), 0.0)
    if not torch.isfinite(advantage).all():
        raise FloatingPointError("sampled-token advantage contains NaN or Inf")
    return advantage.detach()


def sampled_token_policy_loss(
    current_student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Policy-gradient surrogate ``mean(-stopgrad(A) * current_logp)``."""
    if current_student_log_probs.shape != advantage.shape:
        raise ValueError("current log-probs and advantage shapes must match")
    if current_student_log_probs.shape != response_mask.shape:
        raise ValueError("current log-probs and response_mask shapes must match")
    loss = masked_mean(
        -current_student_log_probs.float() * advantage.detach(), response_mask
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("sampled-token policy loss is NaN or Inf")
    return loss
