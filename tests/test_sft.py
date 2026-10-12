import pytest
from miniopd.training.sft import (
    IGNORE_INDEX,
    SFTConfig,
    length_report,
    tokenize_conversation,
    validate_sft_splits,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if len(messages) == 2:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21, 22]


def _row():
    return {
        "sample_id": "gsm8k-train-00000",
        "messages": [
            {"role": "system", "content": "Reason."},
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "1+1=2. \\boxed{2}"},
        ],
    }


def test_sft_masks_prompt_and_supervises_only_assistant_tokens():
    encoded = tokenize_conversation(FakeTokenizer(), _row())
    assert encoded["labels"] == [IGNORE_INDEX] * 3 + [20, 21, 22]
    assert encoded["target_tokens"] == 3


def test_length_report_uses_real_observed_maximum():
    report = length_report(
        [{"total_tokens": value} for value in (65, 100, 129)], multiple=64
    )
    assert report["max"] == 129
    assert report["recommended_max_length"] == 192


def test_sft_config_protects_effective_batch_size():
    assert SFTConfig().effective_batch_size == 32
    with pytest.raises(ValueError, match="effective batch"):
        SFTConfig(gradient_accumulation_steps=8)


def test_sft_split_validation_rejects_leakage_and_test_rows():
    train = [
        {
            "sample_id": "gsm8k-train-00001",
            "split": "sft",
            "question": "Train question",
        }
    ]
    validation = [
        {
            "sample_id": "gsm8k-train-00002",
            "split": "validation",
            "question": "Validation question",
        }
    ]
    validate_sft_splits(train, validation)
    with pytest.raises(ValueError, match="validation split"):
        validate_sft_splits(train, [{**validation[0], "split": "test"}])
    with pytest.raises(ValueError, match="leak"):
        validate_sft_splits(train, [{**validation[0], "question": "Train question"}])
