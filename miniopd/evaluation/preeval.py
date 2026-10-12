"""GPU pre-evaluation orchestration for the fixed GSM8K first-100 benchmark."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from miniopd.evaluation.io import (
    load_evaluation_examples,
    read_jsonl,
    write_json,
)
from miniopd.evaluation.pipeline import run_evaluation
from miniopd.evaluation.prompt import (
    render_generation_prompt,
    save_rendered_prompt_debug,
)
from miniopd.evaluation.schema import (
    EvaluationExample,
    GenerationBackend,
    GenerationConfig,
)


@dataclass(frozen=True)
class PreEvaluationConfig:
    backend: str = "vllm"
    dtype: str = "bfloat16"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 512
    batch_size: int = 8
    offset: int = 0
    limit: int = 100
    review_seed: int = 42
    review_samples_per_group: int = 5

    def __post_init__(self) -> None:
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be transformers or vllm")
        if self.dtype != "bfloat16":
            raise ValueError("formal pre-evaluation must use dtype=bfloat16")
        if self.offset != 0 or self.limit != 100:
            raise ValueError("formal pre-evaluation must use the first 100 examples")
        if self.review_samples_per_group < 1:
            raise ValueError("review_samples_per_group must be positive")
        self.generation_config()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PreEvaluationConfig:
        allowed = set(cls.__dataclass_fields__)
        unexpected = set(value).difference(allowed)
        if unexpected:
            raise ValueError(
                f"unknown pre-evaluation config keys: {sorted(unexpected)}"
            )
        return cls(**dict(value))

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=self.temperature,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def select_first_100(
    examples: Sequence[EvaluationExample], config: PreEvaluationConfig
) -> list[EvaluationExample]:
    selected = list(examples[config.offset : config.offset + config.limit])
    if len(selected) != config.limit:
        raise ValueError(
            f"expected {config.limit} pre-evaluation examples, got {len(selected)}"
        )
    if any(example.split != "test" for example in selected):
        raise ValueError("pre-evaluation input must come from the official test split")
    return selected


def build_review_queue(
    records: Sequence[Mapping[str, Any]], *, seed: int, samples_per_group: int
) -> dict[str, Any]:
    """Select deterministic random examples plus every format/truncation failure."""
    ordered = sorted(
        (dict(record) for record in records), key=lambda row: row["sample_id"]
    )
    correct = [row for row in ordered if bool(row.get("correct"))]
    wrong = [row for row in ordered if not bool(row.get("correct"))]
    format_errors = [
        row for row in ordered if row.get("format_status") == "format_error"
    ]
    truncated = [row for row in ordered if bool(row.get("truncated"))]
    generator = random.Random(seed)

    def sample(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        count = min(samples_per_group, len(pool))
        return generator.sample(pool, count) if count else []

    return {
        "seed": seed,
        "samples_per_group": samples_per_group,
        "random_correct": sample(correct),
        "random_wrong": sample(wrong),
        "all_format_errors": format_errors,
        "all_truncated": truncated,
    }


def _create_backend(
    model_id: str, config: PreEvaluationConfig, trust_remote_code: bool
) -> GenerationBackend:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GPU pre-evaluation requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pre-evaluation requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")
    if config.backend == "transformers":
        from miniopd.evaluation.backends import TransformersBackend

        return TransformersBackend(
            model_id,
            dtype=config.dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    from miniopd.evaluation.backends import VLLMBackend

    return VLLMBackend(
        model_id,
        dtype=config.dtype,
        trust_remote_code=trust_remote_code,
    )


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "verl",
        "ray",
        "datasets",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch
    except ImportError:
        return environment
    environment["cuda"] = {
        "available": torch.cuda.is_available(),
        "runtime_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        ),
    }
    return environment


def write_resolved_config(path: Path, value: Mapping[str, Any]) -> None:
    import yaml

    payload = yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("resolved_config.yaml differs from the existing run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def save_or_validate_prompts(
    examples: Sequence[EvaluationExample], backend: GenerationBackend, path: Path
) -> None:
    records = [
        render_generation_prompt(
            backend.tokenizer, example.question, sample_id=example.sample_id
        )
        for example in examples
    ]
    expected = [record.as_dict() for record in records]
    if path.exists():
        if read_jsonl(path) != expected:
            raise ValueError("prompt_debug.jsonl differs from the existing run")
        return
    save_rendered_prompt_debug(records, path)


def run_preevaluation(
    *,
    input_path: Path,
    output_dir: Path,
    model_id: str,
    config: PreEvaluationConfig,
    trust_remote_code: bool = False,
    backend_factory: Callable[
        [str, PreEvaluationConfig, bool], GenerationBackend
    ] = _create_backend,
) -> dict[str, Any]:
    examples = select_first_100(load_evaluation_examples(input_path), config)
    resolved = {
        **config.as_dict(),
        "model": model_id,
        "input": str(input_path.resolve()),
        "scope": "gsm8k_test_first_100_preevaluation_not_official_result",
    }
    write_resolved_config(output_dir / "resolved_config.yaml", resolved)

    write_json(output_dir / "environment.json", collect_environment())
    backend = backend_factory(model_id, config, trust_remote_code)
    save_or_validate_prompts(examples, backend, output_dir / "prompt_debug.jsonl")
    summary = run_evaluation(
        examples,
        backend,
        output_dir / "predictions.jsonl",
        config=config.generation_config(),
    )
    summary.update(
        {
            "scope": "gsm8k_test_first_100_preevaluation_not_official_result",
            "subset": {"split": "test", "offset": 0, "limit": 100},
        }
    )
    write_json(output_dir / "summary.json", summary)
    review_queue = build_review_queue(
        read_jsonl(output_dir / "predictions.jsonl"),
        seed=config.review_seed,
        samples_per_group=config.review_samples_per_group,
    )
    write_json(output_dir / "review_queue.json", review_queue)
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True))
