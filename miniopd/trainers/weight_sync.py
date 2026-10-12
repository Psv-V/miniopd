"""Reference in-process actor-to-rollout weight synchronization."""

from __future__ import annotations

import torch


def modules_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[name], right_state[name]) for name in left_state)


def sync_module_weights(
    actor: torch.nn.Module, rollout: torch.nn.Module
) -> dict[str, int]:
    """Copy actor parameters and buffers into an in-process rollout reference model."""
    actor_state = actor.state_dict()
    rollout_state = rollout.state_dict()
    if actor_state.keys() != rollout_state.keys():
        raise ValueError("actor and rollout state-dict keys do not match")
    for name in actor_state:
        if actor_state[name].shape != rollout_state[name].shape:
            raise ValueError(f"state tensor shape mismatch for {name}")
    rollout.load_state_dict(actor_state, strict=True)
    if not modules_equal(actor, rollout):
        raise RuntimeError("actor-to-rollout weight synchronization failed")
    return {
        "tensor_count": len(actor_state),
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
    }
