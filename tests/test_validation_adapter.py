import torch
from miniopd.integration.validation_adapter import response_truncated_flags


def test_response_truncation_uses_mask_and_terminal_eos() -> None:
    responses = torch.tensor(
        [
            [10, 2, 0, 0],
            [10, 11, 12, 13],
            [10, 11, 12, 2],
            [10, 11, 0, 0],
        ]
    )
    prompt_mask = torch.ones((4, 3), dtype=torch.long)
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]
    )
    attention_mask = torch.cat((prompt_mask, response_mask), dim=1)

    assert response_truncated_flags(
        responses,
        attention_mask,
        prompt_width=3,
        eos_token_id=2,
    ) == [False, True, False, False]


def test_response_truncation_rejects_misaligned_mask() -> None:
    responses = torch.ones((1, 4), dtype=torch.long)
    attention_mask = torch.ones((1, 6), dtype=torch.long)

    try:
        response_truncated_flags(
            responses,
            attention_mask,
            prompt_width=3,
            eos_token_id=2,
        )
    except ValueError as error:
        assert "not aligned" in str(error)
    else:
        raise AssertionError("misaligned response mask was accepted")
