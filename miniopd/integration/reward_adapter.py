"""Runtime verl reward-manager adapter with explicit truncation evidence."""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import NaiveRewardManager, register

from miniopd.integration.validation_adapter import response_truncated_flags


def compute_score(*args: Any, **kwargs: Any) -> Any:
    """Preserve upstream verl's default dataset-specific reward function."""
    return default_compute_score(*args, **kwargs)


@register("miniopd")
class MiniOPDRewardManager(NaiveRewardManager):
    """Add token-derived truncation flags without changing task rewards."""

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        result = super().__call__(data, return_dict=return_dict)
        if not return_dict:
            return result
        flags = response_truncated_flags(
            data.batch["responses"],
            data.batch["attention_mask"],
            prompt_width=data.batch["prompts"].shape[-1],
            eos_token_id=self.tokenizer.eos_token_id,
        )
        extra_info = result["reward_extra_info"]
        if "truncated" in extra_info:
            raise ValueError("reward metadata already contains truncated")
        extra_info["truncated"] = flags
        return result
