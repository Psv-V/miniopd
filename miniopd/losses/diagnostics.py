"""Serializable tensor contracts used while integrating the algorithm with verl."""

from __future__ import annotations

from typing import Mapping

import torch


def tensor_record(name: str, tensor: torch.Tensor, semantics: str) -> dict[str, object]:
    if not name.strip() or not semantics.strip():
        raise ValueError("tensor name and semantics must be non-empty")
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "semantics": semantics,
    }


def opd_tensor_records(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    semantics = {
        "input_ids": "full prompt+response token IDs before next-token shift",
        "shifted_logits": "position t predicts target token at original position t+1",
        "target_ids": "next-token targets input_ids[:, 1:]",
        "response_mask": "response and first EOS; excludes prompt, padding, post-EOS",
        "old_student_log_probs": "frozen rollout-policy log-prob on sampled tokens",
        "teacher_log_probs": "frozen teacher log-prob on the same sampled tokens",
        "advantage": "detached teacher_logp-old_student_logp, zero outside mask",
        "current_student_log_probs": "trainable actor log-prob on sampled tokens",
    }
    missing = set(semantics).difference(tensors)
    unexpected = set(tensors).difference(semantics)
    if missing or unexpected:
        raise ValueError(
            f"tensor diagnostics mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        name: tensor_record(name, tensors[name], meaning)
        for name, meaning in semantics.items()
    }
