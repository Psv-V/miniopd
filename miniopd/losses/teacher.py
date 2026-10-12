"""Frozen-teacher trajectory scoring helpers."""

from __future__ import annotations

from typing import Any

import torch

from miniopd.losses.token_log_probs import gather_shifted_token_log_probs


def freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def score_teacher_trajectory(
    teacher: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    log_prob_min_clamp: float = -10.0,
) -> torch.Tensor:
    """Score the student's exact trajectory without creating a teacher graph."""
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("teacher must be frozen before trajectory scoring")
    teacher.eval()
    with torch.inference_mode():
        output: Any = teacher(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output
        log_probs = gather_shifted_token_log_probs(
            logits,
            input_ids,
            log_prob_min_clamp=log_prob_min_clamp,
        )
    return log_probs.detach()
