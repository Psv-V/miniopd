"""Resumable generation, deterministic scoring, and aggregate metrics."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from miniopd.evaluation.answer_parser import ErrorType, evaluate_answer
from miniopd.evaluation.io import (
    append_jsonl,
    completed_sample_ids,
    read_jsonl,
    write_json,
)
from miniopd.evaluation.schema import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationExample,
    GenerationBackend,
    GenerationConfig,
    GenerationOutput,
)


def _batches(
    values: Sequence[EvaluationExample], size: int
) -> Iterable[Sequence[EvaluationExample]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _record(
    example: EvaluationExample,
    output: GenerationOutput,
    *,
    backend: GenerationBackend,
    config: GenerationConfig,
) -> dict[str, Any]:
    result = evaluate_answer(
        output.response,
        example.reference_answer,
        truncated=output.truncated,
        runtime_error=output.runtime_error,
    )
    format_status = (
        "ok"
        if result.prediction.parsed
        else (
            result.prediction.error_type.value
            if result.prediction.error_type is not None
            else "format_error"
        )
    )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "sample_id": example.sample_id,
        "split": example.split,
        "question": example.question,
        "ground_truth": example.reference_answer,
        "raw_response": output.response,
        "parsed_answer": result.prediction.as_dict(),
        "correct": result.correct,
        "format_status": format_status,
        "error_type": result.error_type.value if result.error_type else None,
        "truncated": output.truncated,
        "finish_reason": output.finish_reason,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "latency_seconds": output.latency_seconds,
        "tokens_per_second": output.tokens_per_second,
        "runtime_message": result.runtime_message,
        "model": backend.model_id,
        "model_revision": getattr(backend, "model_revision", None),
        "backend": backend.name,
        "generation_config": config.as_dict(),
    }


def run_evaluation(
    examples: Sequence[EvaluationExample],
    backend: GenerationBackend,
    output_path: Path,
    *,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate and score examples, skipping IDs already safely saved in JSONL."""
    config = config or GenerationConfig()
    completed = completed_sample_ids(output_path)
    existing = read_jsonl(output_path) if output_path.exists() else []
    for row in existing:
        if row.get("model") != backend.model_id or row.get("backend") != backend.name:
            raise ValueError("cannot resume with a different model or backend")
        if row.get("model_revision") != getattr(backend, "model_revision", None):
            raise ValueError("cannot resume with a different model revision")
        if row.get("generation_config") != config.as_dict():
            raise ValueError("cannot resume with a different generation configuration")
    known_ids = {example.sample_id for example in examples}
    unexpected = completed.difference(known_ids)
    if unexpected:
        raise ValueError(f"output contains IDs absent from input: {sorted(unexpected)}")
    pending = [example for example in examples if example.sample_id not in completed]

    started = time.perf_counter()
    for batch in _batches(pending, config.batch_size):
        batch_started = time.perf_counter()
        try:
            outputs = backend.generate_batch(
                [example.messages for example in batch], config
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"backend returned {len(outputs)} outputs for {len(batch)} inputs"
                )
        except Exception as error:
            elapsed = time.perf_counter() - batch_started
            outputs = [
                GenerationOutput(
                    latency_seconds=elapsed,
                    runtime_error=f"{type(error).__name__}: {error}",
                )
                for _ in batch
            ]
        append_jsonl(
            output_path,
            [
                _record(example, output, backend=backend, config=config)
                for example, output in zip(batch, outputs, strict=True)
            ],
        )

    summary = summarize_records(read_jsonl(output_path))
    summary["run_wall_seconds"] = time.perf_counter() - started
    summary["model"] = backend.model_id
    summary["model_revision"] = getattr(backend, "model_revision", None)
    summary["backend"] = backend.name
    summary["generation_config"] = config.as_dict()
    return summary


def rescore_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reapply the shared parser without running a model backend."""
    rescored: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        result = evaluate_answer(
            str(row.get("raw_response", "")),
            str(row["ground_truth"]),
            truncated=bool(row.get("truncated", False)),
            runtime_error=(
                str(row["runtime_message"])
                if row.get("runtime_message") is not None
                else None
            ),
        )
        row["parsed_answer"] = result.prediction.as_dict()
        row["correct"] = result.correct
        row["format_status"] = (
            "ok"
            if result.prediction.parsed
            else (
                result.prediction.error_type.value
                if result.prediction.error_type is not None
                else "format_error"
            )
        )
        row["error_type"] = result.error_type.value if result.error_type else None
        row["runtime_message"] = result.runtime_message
        rescored.append(row)
    return rescored


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record.get("correct")) for record in records)
    errors = Counter(
        str(record["error_type"])
        for record in records
        if record.get("error_type") is not None
    )
    truncated = sum(bool(record.get("truncated")) for record in records)
    output_tokens = sum(int(record.get("output_tokens", 0)) for record in records)
    latencies = [float(record.get("latency_seconds", 0.0)) for record in records]
    total_latency = sum(latencies)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "exact_match": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "format_error": errors[ErrorType.FORMAT_ERROR.value],
        "format_error_ratio": errors[ErrorType.FORMAT_ERROR.value] / total
        if total
        else 0.0,
        "truncated": truncated,
        "truncated_ratio": truncated / total if total else 0.0,
        "average_output_tokens": output_tokens / total if total else 0.0,
        "tokens_per_second": (
            output_tokens / total_latency if total_latency > 0 else None
        ),
        "total_latency_seconds": total_latency,
        "error_counts": dict(sorted(errors.items())),
    }


def score_jsonl(input_path: Path, output_path: Path, summary_path: Path) -> None:
    records = rescore_records(read_jsonl(input_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    append_jsonl(output_path, records)
    write_json(summary_path, summarize_records(records))
