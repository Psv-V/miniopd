"""Deterministic numeric answer parsing and error classification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction


class AnswerSource(str, Enum):
    BOXED = "boxed"
    GSM8K_HASH = "gsm8k_hash"
    LAST_NUMBER = "last_number"


class ErrorType(str, Enum):
    WRONG_ANSWER = "wrong_answer"
    FORMAT_ERROR = "format_error"
    TRUNCATED = "truncated"
    EMPTY_OUTPUT = "empty_output"
    REPETITION = "repetition"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class ParsedAnswer:
    raw_value: str | None
    normalized_value: str | None
    source: AnswerSource | None
    value: Fraction | None
    error_type: ErrorType | None = None

    @property
    def parsed(self) -> bool:
        return self.value is not None and self.error_type is None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "source": self.source.value if self.source is not None else None,
            "error_type": (
                self.error_type.value if self.error_type is not None else None
            ),
        }


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    prediction: ParsedAnswer
    reference_value: str
    error_type: ErrorType | None
    runtime_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "correct": self.correct,
            "prediction": self.prediction.as_dict(),
            "reference_value": self.reference_value,
            "error_type": self.error_type.value
            if self.error_type is not None
            else None,
            "runtime_message": self.runtime_message,
        }


@dataclass(frozen=True)
class _NumericCandidate:
    start: int
    end: int
    raw: str
    value: Fraction


_MINUS_TRANSLATION = str.maketrans({"−": "-", "–": "-", "—": "-"})
_UNSIGNED_DECIMAL = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_SIGNED_DECIMAL = rf"[+-]?{_UNSIGNED_DECIMAL}"
_LATEX_FRACTION_RE = re.compile(
    rf"\\(?:dfrac|tfrac|frac)\s*"
    rf"\{{\s*(?P<numerator>{_SIGNED_DECIMAL})\s*\}}\s*"
    rf"\{{\s*(?P<denominator>{_SIGNED_DECIMAL})\s*\}}"
    rf"(?P<percent>\s*\\?%)?"
)
_SIMPLE_NUMBER_RE = re.compile(
    rf"(?<![\w.])"
    rf"(?P<numerator>{_SIGNED_DECIMAL})"
    rf"(?:\s*/\s*(?P<denominator>{_SIGNED_DECIMAL}))?"
    rf"(?P<percent>\s*\\?%)?"
    rf"(?![\d,.])"
)


def _parse_decimal(value: str) -> Fraction | None:
    value = value.strip().translate(_MINUS_TRANSLATION)
    unsigned = value[1:] if value.startswith(("+", "-")) else value
    integer_part = unsigned.split(".", 1)[0]
    if "," in integer_part:
        groups = integer_part.split(",")
        if not (
            1 <= len(groups[0]) <= 3
            and groups[0].isdigit()
            and all(len(group) == 3 and group.isdigit() for group in groups[1:])
        ):
            return None
    try:
        return Fraction(Decimal(value.replace(",", "")))
    except (InvalidOperation, ValueError):
        return None


def _candidate_value(
    numerator: str,
    denominator: str | None,
    percent: str | None,
) -> Fraction | None:
    value = _parse_decimal(numerator)
    if value is None:
        return None
    if denominator is not None:
        denominator_value = _parse_decimal(denominator)
        if denominator_value in {None, Fraction(0)}:
            return None
        value /= denominator_value
    if percent is not None:
        value /= 100
    return value


def _numeric_candidates(text: str) -> list[_NumericCandidate]:
    normalized = unicodedata.normalize("NFKC", text).translate(_MINUS_TRANSLATION)
    candidates: list[_NumericCandidate] = []
    latex_spans: list[tuple[int, int]] = []

    for match in _LATEX_FRACTION_RE.finditer(normalized):
        latex_spans.append(match.span())
        value = _candidate_value(
            match.group("numerator"),
            match.group("denominator"),
            match.group("percent"),
        )
        if value is not None:
            candidates.append(
                _NumericCandidate(match.start(), match.end(), match.group(0), value)
            )

    for match in _SIMPLE_NUMBER_RE.finditer(normalized):
        if any(
            start <= match.start() and match.end() <= end for start, end in latex_spans
        ):
            continue
        value = _candidate_value(
            match.group("numerator"),
            match.group("denominator"),
            match.group("percent"),
        )
        if value is not None:
            candidates.append(
                _NumericCandidate(match.start(), match.end(), match.group(0), value)
            )

    return sorted(candidates, key=lambda candidate: (candidate.start, candidate.end))


def _canonical_value(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def normalize_numeric_value(raw_value: str) -> str | None:
    """Normalize the last valid numeric value in a candidate string."""
    candidates = _numeric_candidates(raw_value)
    return _canonical_value(candidates[-1].value) if candidates else None


def _boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    marker = "\\boxed"
    search_from = 0
    while True:
        marker_index = text.find(marker, search_from)
        if marker_index < 0:
            break
        brace_index = marker_index + len(marker)
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1
        if brace_index >= len(text) or text[brace_index] != "{":
            search_from = marker_index + len(marker)
            continue

        depth = 0
        position = brace_index
        while position < len(text):
            character = text[position]
            if character == "{" and (position == 0 or text[position - 1] != "\\"):
                depth += 1
            elif character == "}" and (position == 0 or text[position - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    contents.append(text[brace_index + 1 : position])
                    search_from = position + 1
                    break
            position += 1
        else:
            search_from = brace_index + 1
    return contents


def _last_hash_value(text: str) -> str | None:
    marker_index = text.rfind("####")
    if marker_index < 0:
        return None
    remainder = text[marker_index + 4 :]
    return remainder.splitlines()[0].strip() if remainder else ""


def _parsed_candidate(raw_value: str, source: AnswerSource) -> ParsedAnswer | None:
    candidates = _numeric_candidates(raw_value)
    if not candidates:
        return None
    candidate = candidates[-1]
    return ParsedAnswer(
        raw_value=raw_value.strip(),
        normalized_value=_canonical_value(candidate.value),
        source=source,
        value=candidate.value,
    )


def parse_answer(output: str | None) -> ParsedAnswer:
    """Parse an answer using boxed, GSM8K hash, then last-number priority."""
    if output is None or not output.strip():
        return ParsedAnswer(
            raw_value=None,
            normalized_value=None,
            source=None,
            value=None,
            error_type=ErrorType.EMPTY_OUTPUT,
        )

    boxed = _boxed_contents(output)
    if boxed:
        parsed = _parsed_candidate(boxed[-1], AnswerSource.BOXED)
        if parsed is not None:
            return parsed

    hash_value = _last_hash_value(output)
    if hash_value is not None:
        parsed = _parsed_candidate(hash_value, AnswerSource.GSM8K_HASH)
        if parsed is not None:
            return parsed

    candidates = _numeric_candidates(output)
    if candidates:
        candidate = candidates[-1]
        return ParsedAnswer(
            raw_value=candidate.raw.strip(),
            normalized_value=_canonical_value(candidate.value),
            source=AnswerSource.LAST_NUMBER,
            value=candidate.value,
        )

    return ParsedAnswer(
        raw_value=None,
        normalized_value=None,
        source=None,
        value=None,
        error_type=ErrorType.FORMAT_ERROR,
    )


def detect_repetition(output: str, min_repeats: int = 3) -> bool:
    """Detect obvious repeated lines or repeated token blocks at the output tail."""
    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2")
    lines = [" ".join(line.split()).casefold() for line in output.splitlines()]
    lines = [line for line in lines if line]
    for index in range(len(lines) - min_repeats + 1):
        if len(set(lines[index : index + min_repeats])) == 1:
            return True

    tokens = output.casefold().split()
    max_width = min(20, len(tokens) // min_repeats)
    for width in range(3, max_width + 1):
        block = tokens[-width:]
        if all(
            tokens[-repeat * width : -(repeat - 1) * width or None] == block
            for repeat in range(2, min_repeats + 1)
        ):
            return True
    return False


def evaluate_answer(
    output: str | None,
    reference: str,
    *,
    truncated: bool = False,
    repetition: bool | None = None,
    runtime_error: str | None = None,
) -> EvaluationResult:
    """Compare an output exactly and assign one exclusive diagnostic category."""
    prediction = parse_answer(output)
    reference_answer = parse_answer(reference)
    if not reference_answer.parsed:
        raise ValueError(f"reference answer is not parseable: {reference!r}")
    reference_value = str(reference_answer.normalized_value)

    if runtime_error is not None:
        return EvaluationResult(
            correct=False,
            prediction=prediction,
            reference_value=reference_value,
            error_type=ErrorType.RUNTIME_ERROR,
            runtime_message=runtime_error,
        )
    if prediction.error_type == ErrorType.EMPTY_OUTPUT:
        return EvaluationResult(
            False, prediction, reference_value, ErrorType.EMPTY_OUTPUT
        )
    if prediction.parsed and prediction.value == reference_answer.value:
        return EvaluationResult(True, prediction, reference_value, None)
    if truncated:
        return EvaluationResult(False, prediction, reference_value, ErrorType.TRUNCATED)
    if repetition is None:
        repetition = detect_repetition(output or "")
    if repetition:
        return EvaluationResult(
            False, prediction, reference_value, ErrorType.REPETITION
        )
    if prediction.error_type == ErrorType.FORMAT_ERROR:
        return EvaluationResult(
            False, prediction, reference_value, ErrorType.FORMAT_ERROR
        )
    return EvaluationResult(False, prediction, reference_value, ErrorType.WRONG_ANSWER)
