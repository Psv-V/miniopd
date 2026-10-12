"""Categorical entropy and teacher/student entropy-gap diagnostics."""

from __future__ import annotations

import torch

from miniopd.losses.masking import masked_mean


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must include a vocabulary dimension")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1)
    if not torch.isfinite(entropy).all():
        raise FloatingPointError("entropy contains NaN or Inf")
    return entropy


def entropy_gap(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits shapes must match")
    student = token_entropy(student_logits)
    teacher = token_entropy(teacher_logits)
    if response_mask.shape != student.shape:
        raise ValueError("response_mask must match token entropy shape")
    student_mean = masked_mean(student, response_mask)
    teacher_mean = masked_mean(teacher, response_mask)
    return {
        "student_entropy": student_mean,
        "teacher_entropy": teacher_mean,
        "entropy_gap": student_mean - teacher_mean,
    }
