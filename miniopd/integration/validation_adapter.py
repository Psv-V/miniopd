"""Token-level validation metadata shared by the verl runtime adapter."""

from __future__ import annotations

import torch


def response_truncated_flags(
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prompt_width: int,
    eos_token_id: int | None,
) -> list[bool]:
    """Return explicit length-truncation flags from generated token tensors."""
    if responses.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("responses and attention_mask must be rank-2 tensors")
    response_mask = attention_mask[:, prompt_width:].bool()
    if response_mask.shape != responses.shape:
        raise ValueError("response attention mask is not aligned with responses")

    flags = []
    for token_ids, token_mask in zip(responses, response_mask, strict=True):
        exhausted_budget = bool(token_mask.all().item())
        active_ids = token_ids[token_mask]
        ended_with_eos = bool(
            eos_token_id is not None
            and active_ids.numel()
            and active_ids[-1].item() == eos_token_id
        )
        flags.append(exhausted_budget and not ended_with_eos)
    return flags
