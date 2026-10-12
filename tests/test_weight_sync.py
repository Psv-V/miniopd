from copy import deepcopy

import torch
from miniopd.trainers.weight_sync import modules_equal, sync_module_weights


def test_actor_update_is_invisible_until_rollout_weight_sync():
    torch.manual_seed(42)
    actor = torch.nn.Linear(3, 2)
    rollout = deepcopy(actor)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    rollout_before = rollout(inputs).detach().clone()

    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    optimizer.zero_grad()
    actor(inputs).sum().backward()
    optimizer.step()

    assert not modules_equal(actor, rollout)
    torch.testing.assert_close(rollout(inputs), rollout_before)
    assert not torch.equal(actor(inputs), rollout(inputs))

    report = sync_module_weights(actor, rollout)

    assert modules_equal(actor, rollout)
    torch.testing.assert_close(actor(inputs), rollout(inputs), rtol=0, atol=0)
    assert report["parameter_count"] == sum(p.numel() for p in actor.parameters())
