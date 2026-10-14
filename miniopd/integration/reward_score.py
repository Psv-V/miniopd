"""Task-score adapter that uses MiniOPD's shared answer semantics."""

from __future__ import annotations

from typing import Any

from miniopd.evaluation.answer_parser import evaluate_answer

GSM8K_DATA_SOURCE = "openai/gsm8k"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Return GSM8K exact match under the shared boxed-answer parser."""
    if data_source != GSM8K_DATA_SOURCE:
        raise ValueError(f"unsupported MiniOPD reward data source: {data_source}")
    del extra_info
    return float(evaluate_answer(solution_str, ground_truth).correct)
