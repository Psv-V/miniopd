from copy import deepcopy
from types import SimpleNamespace

import torch
from miniopd.losses.sampled_opd import (
    sampled_token_advantage,
    sampled_token_policy_loss,
)
from miniopd.losses.teacher import freeze_teacher, score_teacher_trajectory


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.output = torch.nn.Linear(4, 5, bias=False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_teacher_is_frozen_and_unchanged_after_student_backward():
    torch.manual_seed(42)
    teacher = freeze_teacher(TinyCausalLM())
    initial_state = deepcopy(teacher.state_dict())
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    teacher_log_probs = score_teacher_trajectory(teacher, input_ids, attention_mask)
    old_student = torch.full_like(teacher_log_probs, -2.0)
    response_mask = torch.tensor([[False, True, True]])
    advantage = sampled_token_advantage(teacher_log_probs, old_student, response_mask)
    current = torch.nn.Parameter(torch.full_like(teacher_log_probs, -1.0))
    sampled_token_policy_loss(current, advantage, response_mask).backward()

    assert not teacher.training
    assert not teacher_log_probs.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
