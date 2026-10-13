"""Resumable teacher generation and deterministic cold-start SFT filtering."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from miniopd.evaluation.answer_parser import detect_repetition
from miniopd.evaluation.io import load_evaluation_examples, read_jsonl, write_json
from miniopd.evaluation.pipeline import run_evaluation
from miniopd.evaluation.prompt import build_sft_messages
from miniopd.evaluation.schema import (
    EvaluationExample,
    GenerationBackend,
    GenerationConfig,
)

EXPECTED_COLDSTART_INPUTS = 3_000
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_ARTIFACTS = {
    "raw_generations.jsonl",
    "sft_train.jsonl",
    "summary.json",
    "review_queue.json",
    "resolved_config.yaml",
}


@dataclass(frozen=True)
class ColdStartConfig:
    backend: str = "vllm"
    dtype: str = "bfloat16"
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    max_new_tokens: int = 512
    batch_size: int = 8
    expected_inputs: int = EXPECTED_COLDSTART_INPUTS
    seed: int = 42
    review_samples_per_group: int = 10

    def __post_init__(self) -> None:
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be transformers or vllm")
        if self.dtype != "bfloat16":
            raise ValueError("cold-start generation must use bfloat16")
        if not self.do_sample:
            raise ValueError("cold-start generation must sample")
        if self.temperature != 0.7 or self.top_p != 0.95:
            raise ValueError("cold-start generation fixes temperature=0.7, top_p=0.95")
        if self.max_new_tokens != 512:
            raise ValueError("cold-start generation fixes max_new_tokens=512")
        if self.expected_inputs != EXPECTED_COLDSTART_INPUTS:
            raise ValueError("cold-start generation requires exactly 3000 inputs")
        if self.batch_size < 1 or self.review_samples_per_group < 1:
            raise ValueError("batch and review sizes must be positive")
        self.generation_config()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ColdStartConfig:
        unexpected = set(value).difference(cls.__dataclass_fields__)
        if unexpected:
            raise ValueError(f"unknown cold-start config keys: {sorted(unexpected)}")
        return cls(**dict(value))

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            dtype=self.dtype,
            seed=self.seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_coldstart_inputs(
    examples: Sequence[EvaluationExample], expected_inputs: int
) -> None:
    if len(examples) != expected_inputs:
        raise ValueError(f"expected {expected_inputs} SFT prompts, got {len(examples)}")
    if any(example.split != "sft" for example in examples):
        raise ValueError("cold-start input must contain only the fixed SFT split")
    if any(not example.sample_id.startswith("gsm8k-train-") for example in examples):
        raise ValueError("cold-start input may only contain GSM8K train rows")
    ids = [example.sample_id for example in examples]
    questions = [example.question for example in examples]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("cold-start input contains duplicate IDs or questions")


def filter_reason(record: Mapping[str, Any]) -> str | None:
    """Return the first deterministic rejection reason, or None when retained."""
    if record.get("runtime_message") is not None:
        return "runtime_error"
    response = str(record.get("raw_response", ""))
    if not response.strip():
        return "empty_output"
    if bool(record.get("truncated")):
        return "truncated"
    if detect_repetition(response):
        return "repetition"
    if record.get("format_status") != "ok":
        return "format_error"
    if not bool(record.get("correct")):
        return "wrong_answer"
    return None


def build_sft_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if filter_reason(record) is not None:
            continue
        response = str(record["raw_response"])
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "split": "sft",
                "question": str(record["question"]),
                "reference_answer": str(record["ground_truth"]),
                "messages": build_sft_messages(str(record["question"]), response),
                "teacher_response": response,
                "parsed_answer": record["parsed_answer"],
                "teacher_model": record["model"],
                "teacher_model_revision": record.get("model_revision"),
            }
        )
    return rows


def summarize_filter(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        reason for record in records if (reason := filter_reason(record)) is not None
    )
    successful = sum(record.get("runtime_message") is None for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    output_tokens = sum(int(record.get("output_tokens", 0)) for record in records)
    latency = sum(float(record.get("latency_seconds", 0.0)) for record in records)
    return {
        "input_count": len(records),
        "successful_generation_count": successful,
        "correct_count": correct,
        "format_error_count": sum(
            record.get("format_status") != "ok" for record in records
        ),
        "truncated_count": sum(bool(record.get("truncated")) for record in records),
        "empty_output_count": sum(
            not str(record.get("raw_response", "")).strip() for record in records
        ),
        "repetition_count": sum(
            detect_repetition(str(record.get("raw_response", ""))) for record in records
        ),
        "runtime_error_count": sum(
            record.get("runtime_message") is not None for record in records
        ),
        "wrong_answer_count": reasons["wrong_answer"],
        "retained_count": len(records) - sum(reasons.values()),
        "total_output_tokens": output_tokens,
        "generation_latency_seconds": latency,
        "tokens_per_second": output_tokens / latency if latency > 0 else None,
        "gpu_hours": latency / 3600,
        "rejection_counts": dict(sorted(reasons.items())),
    }


def build_review_queue(
    records: Sequence[Mapping[str, Any]], *, seed: int, samples_per_group: int
) -> dict[str, Any]:
    kept = [dict(row) for row in records if filter_reason(row) is None]
    dropped = [dict(row) for row in records if filter_reason(row) is not None]
    generator = random.Random(seed)

    def sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = sorted(rows, key=lambda row: str(row["sample_id"]))
        return generator.sample(rows, min(samples_per_group, len(rows)))

    dropped_sample = sample(dropped)
    return {
        "seed": seed,
        "samples_per_group": samples_per_group,
        "random_retained": sample(kept),
        "random_rejected": [
            {**row, "filter_reason": filter_reason(row)} for row in dropped_sample
        ],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(value), sort_keys=True), encoding="utf-8")


def _create_backend(
    model_id: str,
    model_revision: str,
    model_path: Path | None,
    config: ColdStartConfig,
    trust_remote_code: bool,
) -> GenerationBackend:
    load_source = str(model_path) if model_path is not None else model_id
    load_revision = None if model_path is not None else model_revision
    if config.backend == "transformers":
        from miniopd.evaluation.backends import TransformersBackend

        backend = TransformersBackend(
            load_source,
            revision=load_revision,
            dtype=config.dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        from miniopd.evaluation.backends import VLLMBackend

        backend = VLLMBackend(
            load_source,
            revision=load_revision,
            dtype=config.dtype,
            trust_remote_code=trust_remote_code,
        )
    backend.model_id = model_id
    backend.model_revision = model_revision
    return backend


def run_coldstart_generation(
    *,
    input_path: Path,
    output_dir: Path,
    model_id: str,
    model_revision: str,
    model_path: Path | None = None,
    config: ColdStartConfig,
    trust_remote_code: bool = False,
    backend_factory: Callable[
        [str, str, Path | None, ColdStartConfig, bool], GenerationBackend
    ] = _create_backend,
) -> dict[str, Any]:
    if not _REVISION_RE.fullmatch(model_revision):
        raise ValueError("teacher revision must be a full 40-character commit SHA")
    if model_path is not None:
        model_path = model_path.resolve()
        if not model_path.is_dir():
            raise ValueError(
                f"local teacher model path is not a directory: {model_path}"
            )
    examples = load_evaluation_examples(input_path)
    validate_coldstart_inputs(examples, config.expected_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_dir.iterdir()} - _ARTIFACTS
    if unexpected:
        raise RuntimeError(
            f"output directory has unexpected files: {sorted(unexpected)}"
        )

    resolved = {
        **config.as_dict(),
        "input": str(input_path.resolve()),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_path": str(model_path) if model_path is not None else None,
        "scope": "gsm8k_train_fixed_coldstart_3000",
    }
    _write_yaml(output_dir / "resolved_config.yaml", resolved)
    backend = backend_factory(
        model_id, model_revision, model_path, config, trust_remote_code
    )
    raw_path = output_dir / "raw_generations.jsonl"
    run_evaluation(examples, backend, raw_path, config=config.generation_config())
    records = read_jsonl(raw_path)
    if len(records) != config.expected_inputs:
        raise RuntimeError("raw generation did not produce exactly one row per input")

    sft_rows = build_sft_rows(records)
    _write_jsonl(output_dir / "sft_train.jsonl", sft_rows)
    summary = summarize_filter(records)
    summary.update(
        {
            "scope": "gsm8k_train_fixed_coldstart_3000",
            "model": model_id,
            "model_revision": model_revision,
            "generation_config": config.generation_config().as_dict(),
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "review_queue.json",
        build_review_queue(
            records,
            seed=config.seed,
            samples_per_group=config.review_samples_per_group,
        ),
    )
    return summary
