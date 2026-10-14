"""Runtime verl reward-manager adapter with explicit truncation evidence."""

from __future__ import annotations

from typing import Any

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils.reward_score import default_compute_score

from miniopd.integration.validation_adapter import response_truncated_flags


def compute_score(*args: Any, **kwargs: Any) -> Any:
    """Preserve upstream verl's default dataset-specific reward function."""
    return default_compute_score(*args, **kwargs)


class MiniOPDRewardManager(NaiveRewardManager):
    """verl v0.8 reward-loop adapter with explicit truncation evidence."""

    async def run_single(self, data: Any) -> dict[str, Any]:
        result = await super().run_single(data)
        final_sequence = data[-1:]
        flags = response_truncated_flags(
            final_sequence.batch["responses"],
            final_sequence.batch["attention_mask"],
            prompt_width=final_sequence.batch["prompts"].shape[-1],
            eos_token_id=self.tokenizer.eos_token_id,
        )
        extra_info = result["reward_extra_info"]
        if "truncated" in extra_info:
            raise ValueError("reward metadata already contains truncated")
        extra_info["truncated"] = flags[0]
        return result
