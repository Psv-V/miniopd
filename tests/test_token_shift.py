import torch
from miniopd.losses.token_log_probs import (
    gather_shifted_token_log_probs,
    shift_logits_and_labels,
)


def test_position_t_logits_score_token_t_plus_one():
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits = torch.zeros(1, 4, 4, dtype=torch.float64)
    logits[0, 0, 1] = 2.0
    logits[0, 1, 2] = 3.0
    logits[0, 2, 3] = 4.0
    logits[0, 3, 0] = 100.0  # No next-token target exists for this position.

    shifted_logits, targets = shift_logits_and_labels(logits, input_ids)
    actual = gather_shifted_token_log_probs(logits, input_ids)
    expected = torch.log_softmax(shifted_logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    )

    assert shifted_logits.shape == (1, 3, 4)
    assert targets.tolist() == [[1, 2, 3]]
    torch.testing.assert_close(actual, expected.squeeze(-1))


def test_log_prob_gather_preserves_gradient_to_predicting_position():
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    gather_shifted_token_log_probs(logits, input_ids).sum().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :-1]).item() > 0
    assert torch.count_nonzero(logits.grad[:, -1]).item() == 0
