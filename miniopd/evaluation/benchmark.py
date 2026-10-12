"""Full GSM8K BF16 benchmark with reproducibility and preflight checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from miniopd.evaluation.io import (
    load_evaluation_examples,
    read_jsonl,
    write_json,
)
from miniopd.evaluation.pipeline import run_evaluation
from miniopd.evaluation.preeval import (
    build_review_queue,
    collect_environment,
    save_or_validate_prompts,
    write_resolved_config,
)
from miniopd.evaluation.schema import (
    EvaluationExample,
    GenerationBackend,
    GenerationConfig,
)

GSM8K_TEST_SIZE = 1319
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_BENCHMARK_ARTIFACTS = {
    "environment.json",
    "predictions.jsonl",
    "preflight.json",
    "prompt_debug.jsonl",
    "resolved_config.yaml",
    "review_queue.json",
    "summary.json",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    role: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not _REVISION_RE.fullmatch(self.revision):
            raise ValueError("model revision must be a full 40-character commit SHA")
        if self.role not in {"base_student", "teacher"}:
            raise ValueError("role must be base_student or teacher")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelSpec:
        allowed = {"model_id", "revision", "role"}
        unexpected = set(value).difference(allowed)
        if unexpected:
            raise ValueError(f"unknown model config keys: {sorted(unexpected)}")
        return cls(**dict(value))

    def as_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "role": self.role,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str = "vllm"
    dtype: str = "bfloat16"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 512
    batch_size: int = 8
    expected_test_size: int = GSM8K_TEST_SIZE
    minimum_free_disk_gb: float = 10.0
    prompt_debug_samples: int = 5
    review_seed: int = 42
    review_samples_per_group: int = 10

    def __post_init__(self) -> None:
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be transformers or vllm")
        if self.dtype != "bfloat16":
            raise ValueError("formal benchmark must use dtype=bfloat16")
        if self.expected_test_size != GSM8K_TEST_SIZE:
            raise ValueError("formal benchmark must use all 1319 GSM8K test rows")
        if self.minimum_free_disk_gb <= 0:
            raise ValueError("minimum_free_disk_gb must be positive")
        if self.prompt_debug_samples < 1:
            raise ValueError("prompt_debug_samples must be positive")
        if self.review_samples_per_group < 1:
            raise ValueError("review_samples_per_group must be positive")
        self.generation_config()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkConfig:
        allowed = set(cls.__dataclass_fields__)
        unexpected = set(value).difference(allowed)
        if unexpected:
            raise ValueError(f"unknown benchmark config keys: {sorted(unexpected)}")
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


def _validate_dataset(
    input_path: Path, config: BenchmarkConfig
) -> list[EvaluationExample]:
    examples = load_evaluation_examples(input_path)
    if len(examples) != config.expected_test_size:
        raise ValueError(
            f"expected {config.expected_test_size} test rows, got {len(examples)}"
        )
    if any(example.split != "test" for example in examples):
        raise ValueError("formal benchmark input must contain only official test rows")
    expected_ids = [f"gsm8k-test-{index:05d}" for index in range(GSM8K_TEST_SIZE)]
    actual_ids = [example.sample_id for example in examples]
    if actual_ids != expected_ids:
        raise ValueError("formal test rows are missing, reordered, or duplicated")
    return examples


def _check_output_directory(output_dir: Path, minimum_free_disk_gb: float) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name
        for path in output_dir.iterdir()
        if path.name not in _BENCHMARK_ARTIFACTS
    }
    if unexpected:
        raise RuntimeError(
            "benchmark output directory contains unexpected entries: "
            f"{sorted(unexpected)}"
        )
    with tempfile.NamedTemporaryFile(dir=output_dir):
        pass
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gb < minimum_free_disk_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB free; require {minimum_free_disk_gb:.2f} GiB"
        )
    return free_gb


def _check_revision_exists(model: ModelSpec) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for revision preflight"
        ) from error
    info = HfApi().model_info(model.model_id, revision=model.revision)
    if info.sha != model.revision:
        raise RuntimeError(
            f"resolved revision {info.sha!r} does not match {model.revision!r}"
        )
    return str(info.sha)


def collect_gpu_topology() -> dict[str, Any]:
    result: dict[str, Any] = {"query": None, "topology": None}
    commands = {
        "query": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=15
            )
            result[key] = completed.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            result[key] = None
    return result


def collect_gpu_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("formal benchmark requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("formal benchmark requires an available CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA GPU does not support BF16")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no visible CUDA GPU")
    return {
        "cuda_available": True,
        "bf16_supported": True,
        "visible_gpu_count": torch.cuda.device_count(),
        "topology": collect_gpu_topology(),
    }


def _create_backend(
    model: ModelSpec, config: BenchmarkConfig, trust_remote_code: bool
) -> GenerationBackend:
    if config.backend == "transformers":
        from miniopd.evaluation.backends import TransformersBackend

        return TransformersBackend(
            model.model_id,
            revision=model.revision,
            dtype=config.dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    from miniopd.evaluation.backends import VLLMBackend

    return VLLMBackend(
        model.model_id,
        revision=model.revision,
        dtype=config.dtype,
        trust_remote_code=trust_remote_code,
    )


def _smoke_generate(
    backend: GenerationBackend,
    examples: Sequence[EvaluationExample],
    config: GenerationConfig,
) -> dict[str, Any]:
    checks = {}
    for name, count in (("single", 1), ("small_batch", 2)):
        outputs = backend.generate_batch(
            [example.messages for example in examples[:count]], config
        )
        if len(outputs) != count:
            raise RuntimeError(
                f"{name} returned {len(outputs)} outputs, expected {count}"
            )
        for output in outputs:
            if output.runtime_error is not None:
                raise RuntimeError(f"{name} failed: {output.runtime_error}")
            if not output.response.strip() or output.output_tokens < 1:
                raise RuntimeError(f"{name} produced an empty response")
        checks[name] = {
            "passed": True,
            "output_tokens": [output.output_tokens for output in outputs],
        }
    return checks


def check_tokenizer_compatibility(
    student: ModelSpec, teacher: ModelSpec, *, trust_remote_code: bool = False
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for tokenizer preflight"
        ) from error
    student_tokenizer = AutoTokenizer.from_pretrained(
        student.model_id,
        revision=student.revision,
        trust_remote_code=trust_remote_code,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher.model_id,
        revision=teacher.revision,
        trust_remote_code=trust_remote_code,
    )
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    same_vocab = student_vocab == teacher_vocab
    special_ids = {
        name: {
            "student": getattr(student_tokenizer, name),
            "teacher": getattr(teacher_tokenizer, name),
        }
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    compatible = same_vocab and all(
        values["student"] == values["teacher"] for values in special_ids.values()
    )
    report = {
        "compatible": compatible,
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "same_vocab_mapping": same_vocab,
        "special_token_ids": special_ids,
    }
    if not compatible:
        raise RuntimeError(f"student and teacher tokenizers are incompatible: {report}")
    return report


def run_benchmark(
    *,
    input_path: Path,
    output_dir: Path,
    model: ModelSpec,
    config: BenchmarkConfig,
    trust_remote_code: bool = False,
    revision_checker: Callable[[ModelSpec], str] = _check_revision_exists,
    gpu_checker: Callable[[], Mapping[str, Any]] = collect_gpu_preflight,
    backend_factory: Callable[
        [ModelSpec, BenchmarkConfig, bool], GenerationBackend
    ] = _create_backend,
) -> dict[str, Any]:
    examples = _validate_dataset(input_path, config)
    free_disk_gb = _check_output_directory(output_dir, config.minimum_free_disk_gb)
    resolved = {
        **config.as_dict(),
        "model": model.as_dict(),
        "input": str(input_path.resolve()),
        "scope": "official_gsm8k_test_full_1319_bf16",
    }
    write_resolved_config(output_dir / "resolved_config.yaml", resolved)

    revision = revision_checker(model)
    gpu = dict(gpu_checker())
    environment = collect_environment()
    environment["gpu_preflight"] = gpu
    environment["free_disk_gb"] = free_disk_gb
    write_json(output_dir / "environment.json", environment)

    backend = backend_factory(model, config, trust_remote_code)
    save_or_validate_prompts(
        examples[: config.prompt_debug_samples],
        backend,
        output_dir / "prompt_debug.jsonl",
    )
    smoke = _smoke_generate(backend, examples, config.generation_config())
    preflight = {
        "passed": True,
        "dataset_rows": len(examples),
        "model_revision": revision,
        "free_disk_gb": free_disk_gb,
        "gpu": gpu,
        "generation": smoke,
    }
    write_json(output_dir / "preflight.json", preflight)

    summary = run_evaluation(
        examples,
        backend,
        output_dir / "predictions.jsonl",
        config=config.generation_config(),
    )
    summary.update(
        {
            "scope": "official_gsm8k_test_full_1319_bf16",
            "model_role": model.role,
            "official_test_rows": GSM8K_TEST_SIZE,
            "external_published_score_used": False,
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


def print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True))
