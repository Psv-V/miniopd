import hashlib
import json

import pytest
from miniopd.data.gsm8k import (
    DataValidationError,
    extract_ground_truth,
    normalize_question,
    question_hash,
    validate_raw_split,
    verify_raw_snapshot,
)


def test_normalization_and_hash_ignore_formatting_differences() -> None:
    first = "  How many   apples?\n"
    second = "how many apples?"

    assert normalize_question(first) == normalize_question(second)
    assert question_hash(first) == question_hash(second)


def test_extract_ground_truth() -> None:
    assert extract_ground_truth("work\n#### 1,234") == "1234"


def test_raw_validation_rejects_normalized_duplicates() -> None:
    records = [
        {"question": "How many apples?", "answer": "#### 2"},
        {"question": " how  many APPLES? ", "answer": "#### 2"},
    ]

    with pytest.raises(DataValidationError, match="normalized duplicate"):
        validate_raw_split(records, "train")


def test_raw_validation_rejects_missing_final_answer() -> None:
    records = [{"question": "1 + 1?", "answer": "The answer is two."}]

    with pytest.raises(DataValidationError, match="####"):
        validate_raw_split(records, "train")


def test_snapshot_verification_detects_file_changes(tmp_path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    train_path = snapshot / "train.parquet"
    test_path = snapshot / "test.parquet"
    train_path.write_bytes(b"train")
    test_path.write_bytes(b"test")

    metadata = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "revision": revision,
        "split_counts": {"train": 7473, "test": 1319},
        "sha256": {
            "train.parquet": hashlib.sha256(b"train").hexdigest(),
            "test.parquet": hashlib.sha256(b"test").hexdigest(),
        },
    }
    (snapshot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    verify_raw_snapshot(snapshot, revision)
    train_path.write_bytes(b"changed")

    with pytest.raises(DataValidationError, match="checksum mismatch"):
        verify_raw_snapshot(snapshot, revision)
