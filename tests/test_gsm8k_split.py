from collections import Counter

import pytest
from miniopd.data.gsm8k import (
    DEFAULT_SPLIT_SIZES,
    GSM8K_TRAIN_SIZE,
    SYSTEM_PROMPT,
    DataValidationError,
    SplitSizes,
    build_evaluation_records,
    build_manifest,
    build_sft_records,
    build_teacher_rollout_inputs,
    build_verl_opd_records,
    manifest_jsonl_bytes,
)


def make_records(count: int, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "question": f"{prefix} question {index}?",
            "answer": f"reasoning {index}\n#### {index}",
        }
        for index in range(count)
    ]


def make_manifest_fixture():
    sizes = SplitSizes(sft=2, opd_train=2, validation=1)
    train = make_records(5, "train")
    test = make_records(2, "test")
    manifest = build_manifest(
        train,
        test,
        dataset_revision="dataset-commit",
        code_version="code-commit",
        seed=42,
        sizes=sizes,
    )
    return train, test, manifest, sizes


def test_default_split_sizes_cover_official_train() -> None:
    assert DEFAULT_SPLIT_SIZES.total == GSM8K_TRAIN_SIZE


def test_split_is_deterministic_and_has_exact_counts() -> None:
    train, test, first, sizes = make_manifest_fixture()
    second = build_manifest(
        train,
        test,
        dataset_revision="dataset-commit",
        code_version="code-commit",
        seed=42,
        sizes=sizes,
    )

    assert first == second
    assert manifest_jsonl_bytes(first) == manifest_jsonl_bytes(second)
    assert Counter(row["split"] for row in first) == {
        "sft": 2,
        "opd_train": 2,
        "validation": 1,
        "test": 2,
    }
    assert len({row["question_hash"] for row in first}) == len(first)


def test_manifest_rejects_train_test_question_leakage() -> None:
    train = make_records(5, "train")
    test = make_records(2, "test")
    test[0] = dict(train[0])

    with pytest.raises(DataValidationError, match="question leakage"):
        build_manifest(
            train,
            test,
            dataset_revision="dataset-commit",
            code_version="code-commit",
            sizes=SplitSizes(sft=2, opd_train=2, validation=1),
        )


def test_output_formats_share_prompt_and_identity() -> None:
    train, test, manifest, _ = make_manifest_fixture()
    validation = build_evaluation_records(train, test, manifest, split="validation")
    teacher_inputs = build_teacher_rollout_inputs(train, test, manifest)
    opd = build_verl_opd_records(train, test, manifest)

    assert validation[0]["messages"][0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    assert len(teacher_inputs) == 2
    assert len(opd) == 2
    assert set(opd[0]) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert opd[0]["extra_info"]["split"] == "opd_train"


def test_sft_records_require_one_response_per_teacher_input() -> None:
    train, test, manifest, _ = make_manifest_fixture()
    teacher_inputs = build_teacher_rollout_inputs(train, test, manifest)
    responses = {
        str(item["sample_id"]): "reasoning\n\\boxed{1}" for item in teacher_inputs
    }
    sft_records = build_sft_records(teacher_inputs, responses)

    assert len(sft_records) == 2
    assert sft_records[0]["messages"][-1]["role"] == "assistant"

    responses.pop(next(iter(responses)))
    with pytest.raises(DataValidationError, match="do not match"):
        build_sft_records(teacher_inputs, responses)
