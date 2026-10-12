import math

import torch
from miniopd.losses.diagnostics import opd_tensor_records
from miniopd.losses.sampled_opd import (
    sampled_token_advantage,
    sampled_token_policy_loss,
)


def test_k1_reward_matches_hand_calculation_and_has_correct_sign():
    teacher = torch.tensor([[math.log(0.5), math.log(0.1), -2.0]], requires_grad=True)
    old_student = torch.tensor(
        [[math.log(0.2), math.log(0.6), -3.0]], requires_grad=True
    )
    mask = torch.tensor([[True, True, False]])

    advantage = sampled_token_advantage(teacher, old_student, mask)

    expected = torch.tensor([[math.log(2.5), math.log(1 / 6), 0.0]])
    torch.testing.assert_close(advantage, expected)
    assert advantage[0, 0] > 0  # Teacher likes the sampled token more.
    assert advantage[0, 1] < 0  # Teacher likes the sampled token less.
    assert not advantage.requires_grad

    sampled_token_policy_loss(
        torch.nn.Parameter(torch.full_like(advantage, -1.0)), advantage, mask
    ).backward()
    assert teacher.grad is None
    assert old_student.grad is None


def test_policy_surrogate_uses_detached_advantage_to_control_update():
    current = torch.nn.Parameter(torch.tensor([[-1.0, -1.0]]))
    advantage = torch.tensor([[2.0, -4.0]])
    mask = torch.tensor([[True, True]])

    loss = sampled_token_policy_loss(current, advantage, mask)
    loss.backward()

    # Gradient descent increases the positive-advantage token's log-prob and
    # decreases the negative-advantage token's log-prob.
    torch.testing.assert_close(current.grad, torch.tensor([[-1.0, 2.0]]))


def test_extreme_log_probs_are_clamped_and_finite():
    advantage = sampled_token_advantage(
        torch.tensor([[-1.0e30, -1.0]]),
        torch.tensor([[-1.0, -1.0e30]]),
        torch.tensor([[True, True]]),
        log_prob_min_clamp=-10.0,
        loss_max_clamp=5.0,
    )
    assert torch.isfinite(advantage).all()
    assert advantage.tolist() == [[-5.0, 5.0]]


def test_critical_tensor_diagnostics_record_contract():
    tensors = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "shifted_logits": torch.ones(2, 3, 5, requires_grad=True),
        "target_ids": torch.ones(2, 3, dtype=torch.long),
        "response_mask": torch.ones(2, 3, dtype=torch.bool),
        "old_student_log_probs": torch.ones(2, 3),
        "teacher_log_probs": torch.ones(2, 3),
        "advantage": torch.ones(2, 3),
        "current_student_log_probs": torch.ones(2, 3, requires_grad=True),
    }
    records = opd_tensor_records(tensors)

    assert records["response_mask"]["shape"] == [2, 3]
    assert records["response_mask"]["dtype"] == "torch.bool"
    assert records["response_mask"]["device"] == "cpu"
    assert "excludes prompt" in records["response_mask"]["semantics"]
    assert records["current_student_log_probs"]["requires_grad"] is True
