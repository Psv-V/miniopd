import torch
from miniopd.metrics.overlap import topk_overlap


def test_topk_overlap_matches_hand_calculation_and_mask():
    student = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0]]])
    mask = torch.tensor([[True, False]])

    per_token, mean = topk_overlap(student, teacher, mask, k=2)

    # First position: {0, 1} intersects {1, 2} at one of two tokens.
    torch.testing.assert_close(per_token, torch.tensor([[0.5, 0.0]]))
    torch.testing.assert_close(mean, torch.tensor(0.5))


def test_topk_overlap_is_one_for_identical_rankings():
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    _, mean = topk_overlap(logits, logits.clone(), torch.tensor([[True]]), k=3)
    torch.testing.assert_close(mean, torch.tensor(1.0))
