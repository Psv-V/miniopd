"""Response-token masking with explicit EOS semantics."""

from __future__ import annotations

import torch


def build_response_mask(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    eos_token_id: int,
) -> torch.Tensor:
    """Build a next-token-aligned mask, including the first EOS and nothing after it.

    Returned position ``t`` describes the target token ``input_ids[:, t + 1]``.
    ``prompt_lengths`` counts valid prompt tokens, so both left-padded and unpadded
    prompts are supported. Prompt targets and padding are excluded. A first response
    EOS is included so the policy can learn to stop; later positions are excluded.
    """
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be rank-2")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask shapes must match")
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != input_ids.shape[0]:
        raise ValueError("prompt_lengths must have one value per batch item")
    if prompt_lengths.device != input_ids.device:
        raise ValueError("prompt_lengths and input_ids must share a device")
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask and input_ids must share a device")
    if torch.any(prompt_lengths < 1):
        raise ValueError("every prompt must contain at least one token")
    valid_counts = attention_mask.bool().sum(dim=1)
    if torch.any(valid_counts == 0) or torch.any(prompt_lengths > valid_counts):
        raise ValueError("prompt length exceeds the number of valid sequence tokens")

    target_ids = input_ids[:, 1:]
    target_is_valid = attention_mask[:, 1:].bool()
    first_valid = attention_mask.bool().to(torch.int64).argmax(dim=1)
    response_start = first_valid + prompt_lengths
    target_positions = torch.arange(
        1, input_ids.shape[1], device=input_ids.device
    ).unsqueeze(0)
    is_response = target_positions >= response_start.unsqueeze(1)

    is_eos = target_ids.eq(eos_token_id) & target_is_valid & is_response
    eos_before = is_eos.cumsum(dim=1) - is_eos.to(torch.int64)
    through_first_eos = eos_before.eq(0)
    return is_response & target_is_valid & through_first_eos


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over true mask positions, rejecting an empty mask instead of hiding it."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    selected = values.masked_select(mask.bool())
    if selected.numel() == 0:
        raise ValueError("masked mean is undefined for an empty mask")
    return selected.mean()
