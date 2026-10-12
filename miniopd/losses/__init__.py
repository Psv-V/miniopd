"""Small, framework-independent building blocks for sampled-token OPD."""

from miniopd.losses.masking import build_response_mask, masked_mean
from miniopd.losses.sampled_opd import (
    sampled_token_advantage,
    sampled_token_policy_loss,
)
from miniopd.losses.token_log_probs import (
    gather_shifted_token_log_probs,
    shift_logits_and_labels,
)

__all__ = [
    "build_response_mask",
    "gather_shifted_token_log_probs",
    "masked_mean",
    "sampled_token_advantage",
    "sampled_token_policy_loss",
    "shift_logits_and_labels",
]
