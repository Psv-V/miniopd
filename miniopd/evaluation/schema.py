"""Stable, backend-independent schemas for generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic settings used by formal evaluation."""

    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 512
    batch_size: int = 8
    dtype: str = "auto"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.do_sample and self.temperature != 0:
            raise ValueError("greedy decoding requires temperature=0")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationExample:
    sample_id: str
    question: str
    reference_answer: str
    messages: tuple[dict[str, str], ...]
    split: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> EvaluationExample:
        required = ("sample_id", "question", "reference_answer", "messages")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"evaluation row missing fields: {missing}")
        messages = tuple(
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in row["messages"]
        )
        if not messages:
            raise ValueError("messages must not be empty")
        return cls(
            sample_id=str(row["sample_id"]),
            question=str(row["question"]),
            reference_answer=str(row["reference_answer"]),
            messages=messages,
            split=str(row["split"]) if row.get("split") is not None else None,
        )


@dataclass(frozen=True)
class GenerationOutput:
    """One backend response before answer scoring."""

    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    latency_seconds: float = 0.0
    runtime_error: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def tokens_per_second(self) -> float | None:
        if self.latency_seconds <= 0:
            return None
        return self.output_tokens / self.latency_seconds


class GenerationBackend:
    """Small protocol implemented by Transformers, vLLM, and test backends."""

    name: str
    model_id: str
    model_revision: str | None

    def generate_batch(
        self,
        messages: Sequence[Sequence[Mapping[str, str]]],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        raise NotImplementedError
