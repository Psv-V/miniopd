import pytest
from miniopd.evaluation.answer_parser import (
    AnswerSource,
    ErrorType,
    detect_repetition,
    evaluate_answer,
    normalize_numeric_value,
    parse_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000", "1000"),
        ("+42", "42"),
        ("−7", "-7"),
        ("0.50", "1/2"),
        ("1/2", "1/2"),
        (r"\frac{3}{4}", "3/4"),
        ("50%", "1/2"),
        (r"12.5\%", "1/8"),
        ("The result is 24 dollars.", "24"),
    ],
)
def test_numeric_normalization(raw: str, expected: str) -> None:
    assert normalize_numeric_value(raw) == expected


def test_last_boxed_answer_has_highest_priority() -> None:
    parsed = parse_answer(r"first \boxed{1}, final \boxed{2}. #### 3")

    assert parsed.source == AnswerSource.BOXED
    assert parsed.raw_value == "2"
    assert parsed.normalized_value == "2"


def test_nested_latex_fraction_inside_box() -> None:
    parsed = parse_answer(r"Therefore, \boxed{\frac{1}{2}}.")

    assert parsed.source == AnswerSource.BOXED
    assert parsed.normalized_value == "1/2"


def test_gsm8k_hash_precedes_last_number_fallback() -> None:
    parsed = parse_answer("work 12\n#### -1,080 dollars\nafterward 99")

    assert parsed.source == AnswerSource.GSM8K_HASH
    assert parsed.raw_value == "-1,080 dollars"
    assert parsed.normalized_value == "-1080"


def test_fallback_uses_last_legal_number() -> None:
    parsed = parse_answer("We tried 10, but the final result is 2.5 kg.")

    assert parsed.source == AnswerSource.LAST_NUMBER
    assert parsed.normalized_value == "5/2"


def test_empty_and_unparseable_outputs_are_not_silently_ignored() -> None:
    assert parse_answer("  ").error_type == ErrorType.EMPTY_OUTPUT
    assert parse_answer("I cannot determine the result.").error_type == (
        ErrorType.FORMAT_ERROR
    )
    assert parse_answer("invalid 1,00").error_type == ErrorType.FORMAT_ERROR


def test_equivalent_number_forms_compare_exactly() -> None:
    result = evaluate_answer(r"The answer is \boxed{50\%}.", "#### 0.5")

    assert result.correct
    assert result.error_type is None
    assert result.reference_value == "1/2"


def test_wrong_answer_classification() -> None:
    result = evaluate_answer(r"\boxed{41}", "#### 42")

    assert not result.correct
    assert result.error_type == ErrorType.WRONG_ANSWER
    assert result.as_dict()["error_type"] == "wrong_answer"


def test_truncation_and_runtime_take_explicit_precedence() -> None:
    truncated = evaluate_answer("partial 41", "42", truncated=True)
    runtime = evaluate_answer(None, "42", runtime_error="CUDA OOM")

    assert truncated.error_type == ErrorType.TRUNCATED
    assert runtime.error_type == ErrorType.RUNTIME_ERROR
    assert runtime.runtime_message == "CUDA OOM"


def test_correct_answer_remains_correct_when_length_finished() -> None:
    result = evaluate_answer(r"\boxed{42}", "42", truncated=True)

    assert result.correct
    assert result.error_type is None


def test_repetition_detection_and_classification() -> None:
    output = "still calculating 1\nstill calculating 1\nstill calculating 1"

    assert detect_repetition(output)
    assert evaluate_answer(output, "2").error_type == ErrorType.REPETITION


def test_unparseable_reference_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="reference answer"):
        evaluate_answer("42", "not a number")
