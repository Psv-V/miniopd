"""Input and JSONL helpers for resumable evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from miniopd.evaluation.schema import EvaluationExample


def load_evaluation_examples(path: Path) -> list[EvaluationExample]:
    """Load the Stage 4 parquet format or an equivalent JSONL file."""
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix == ".parquet":
        try:
            import datasets
        except ImportError as error:
            raise RuntimeError(
                "install MiniOPD's data extra to read parquet evaluation data"
            ) from error
        rows = [dict(row) for row in datasets.Dataset.from_parquet(str(path))]
    else:
        raise ValueError("evaluation input must be .parquet or .jsonl")

    examples = [EvaluationExample.from_mapping(row) for row in rows]
    sample_ids = [example.sample_id for example in examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation input contains duplicate sample_id values")
    return examples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = _read_jsonl(path)
    identifiers: list[str] = []
    for row in rows:
        if "sample_id" not in row:
            raise ValueError(f"existing output row has no sample_id: {path}")
        identifiers.append(str(row["sample_id"]))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"existing output contains duplicate sample_id values: {path}")
    return set(identifiers)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Durably append complete canonical JSON lines for checkpoint-style resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
