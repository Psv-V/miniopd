import math

import torch
from miniopd.metrics.entropy import entropy_gap, token_entropy


def test_uniform_distribution_entropy_is_log_vocabulary_size():
    entropy = token_entropy(torch.zeros(1, 1, 2))
    torch.testing.assert_close(entropy, torch.tensor([[math.log(2)]]))


def test_entropy_gap_matches_hand_calculation_and_mask():
    student = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    teacher = torch.tensor([[[100.0, -100.0], [0.0, 0.0]]])
    metrics = entropy_gap(student, teacher, torch.tensor([[True, False]]))

    torch.testing.assert_close(
        metrics["student_entropy"], torch.tensor(math.log(2)), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        metrics["teacher_entropy"], torch.tensor(0.0), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        metrics["entropy_gap"], torch.tensor(math.log(2)), rtol=1e-5, atol=1e-6
    )


def test_extreme_logits_have_finite_entropy():
    entropy = token_entropy(torch.tensor([[[-1.0e30, 0.0, -1.0e30]]]))
    assert torch.isfinite(entropy).all()
