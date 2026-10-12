import pytest
import torch
from miniopd.losses.masking import build_response_mask, masked_mean


def test_response_mask_excludes_prompt_padding_and_post_eos():
    input_ids = torch.tensor(
        [
            [10, 11, 20, 21, 2, 2, 0],
            [10, 11, 12, 30, 31, 0, 0],
            [10, 11, 12, 0, 0, 0, 0],
            [0, 0, 10, 11, 40, 2, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    mask = build_response_mask(
        input_ids,
        torch.tensor([2, 3, 3, 2]),
        attention_mask,
        eos_token_id=2,
    )

    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [False, True, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, True, True, False],
    ]


def test_masked_mean_refuses_an_empty_response():
    with pytest.raises(ValueError, match="empty mask"):
        masked_mean(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_masked_mean_ignores_nonfinite_values_outside_mask():
    value = masked_mean(
        torch.tensor([[2.0, float("nan")]]), torch.tensor([[True, False]])
    )
    torch.testing.assert_close(value, torch.tensor(2.0))
