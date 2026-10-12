"""Next-token alignment and sampled-token log-probability helpers."""

from __future__ import annotations

import torch


def shift_logits_and_labels(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align position ``t`` logits with the token stored at position ``t + 1``."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/sequence dimensions must match")
    if input_ids.shape[1] < 2:
        raise ValueError("at least two sequence positions are required")
    return logits[:, :-1, :], input_ids[:, 1:]


def gather_shifted_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    log_prob_min_clamp: float | None = None,
) -> torch.Tensor:
    """Return log p(token[t+1] | tokens[:t+1]) with shape [batch, sequence-1]."""
    shifted_logits, target_ids = shift_logits_and_labels(logits, input_ids)
    if target_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long token IDs")
    vocabulary_size = shifted_logits.shape[-1]
    if torch.any(target_ids < 0) or torch.any(target_ids >= vocabulary_size):
        raise ValueError("input_ids contain a token outside the logits vocabulary")

    # OPD only needs token log-probs. Upcasting avoids BF16 softmax instability while
    # preserving the gradient path to the original logits.
    all_log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    sampled = all_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if log_prob_min_clamp is not None:
        if log_prob_min_clamp > 0:
            raise ValueError("log_prob_min_clamp must be non-positive")
        sampled = sampled.clamp_min(log_prob_min_clamp)
    return sampled
