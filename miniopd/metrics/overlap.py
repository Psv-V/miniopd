"""Top-k vocabulary-overlap diagnostics."""

from __future__ import annotations

import torch

from miniopd.losses.masking import masked_mean


def topk_overlap(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token |student top-k ∩ teacher top-k|/k and masked mean."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("student and teacher logits must share [batch, token, vocab]")
    if response_mask.shape != student_logits.shape[:2]:
        raise ValueError("response_mask must match logits batch/token dimensions")
    vocabulary_size = student_logits.shape[-1]
    if not 1 <= k <= vocabulary_size:
        raise ValueError("k must be between 1 and the vocabulary size")

    student_ids = student_logits.topk(k, dim=-1).indices
    teacher_ids = teacher_logits.topk(k, dim=-1).indices
    matches = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2))
    intersection_size = matches.any(dim=-1).sum(dim=-1)
    per_token = intersection_size.to(torch.float32) / k
    per_token = per_token.masked_fill(~response_mask.bool(), 0.0)
    return per_token, masked_mean(per_token, response_mask)
