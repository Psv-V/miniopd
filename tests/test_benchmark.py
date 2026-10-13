import json

import pytest
from miniopd.evaluation.benchmark import (
    BenchmarkConfig,
    ModelSpec,
    run_benchmark,
)
from miniopd.evaluation.io import read_jsonl
from miniopd.evaluation.schema import GenerationOutput

REVISION = "a" * 40


class FakeTokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        text = "|".join(message["content"] for message in conversation) + "|assistant"
        return [1, 2, 3] if tokenize else text


class FakeBackend:
    name = "vllm"

    def __init__(self, model):
        self.model_id = model.model_id
        self.model_revision = model.revision
        self.tokenizer = FakeTokenizer()

    def generate_batch(self, messages, config):
        return [
            GenerationOutput(
                response="Reasoning. \\boxed{42}",
                input_tokens=20,
                output_tokens=6,
                finish_reason="stop",
                latency_seconds=0.01,
            )
            for _ in messages
        ]


def _write_full_test(path, *, reverse=False):
    rows = [
        {
            "sample_id": f"gsm8k-test-{index:05d}",
            "split": "test",
            "question": f"Question {index}",
            "reference_answer": "42",
            "messages": [
                {"role": "system", "content": "Reason step by step."},
                {"role": "user", "content": f"Question {index}"},
            ],
        }
        for index in range(1319)
    ]
    if reverse:
        rows.reverse()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_model_spec_requires_locked_revision():
    with pytest.raises(ValueError, match="40-character"):
        ModelSpec("Qwen/model", "main", "teacher")


def test_formal_benchmark_contract_is_fixed():
    config = BenchmarkConfig()
    assert config.dtype == "bfloat16"
    assert config.expected_test_size == 1319
    with pytest.raises(ValueError, match="1319"):
        BenchmarkConfig(expected_test_size=100)


def test_full_benchmark_creates_reproducible_artifacts(tmp_path):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "result"
    _write_full_test(input_path)
    model = ModelSpec("Qwen/fake", REVISION, "base_student")
    config = BenchmarkConfig(minimum_free_disk_gb=0.000001)

    summary = run_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        config=config,
        revision_checker=lambda value: value.revision,
        gpu_checker=lambda: {
            "cuda_available": True,
            "bf16_supported": True,
            "visible_gpu_count": 1,
        },
        backend_factory=lambda value, config, trust: FakeBackend(value),
    )

    assert summary["total"] == 1319
    assert summary["correct"] == 1319
    assert summary["model_revision"] == REVISION
    assert summary["scope"] == "official_gsm8k_test_full_1319_bf16"
    assert summary["external_published_score_used"] is False
    assert len(read_jsonl(output_dir / "predictions.jsonl")) == 1319
    assert len(read_jsonl(output_dir / "prompt_debug.jsonl")) == 5
    for name in (
        "environment.json",
        "preflight.json",
        "resolved_config.yaml",
        "review_queue.json",
        "summary.json",
    ):
        assert (output_dir / name).is_file()


def test_full_benchmark_rejects_reordered_test_rows(tmp_path):
    input_path = tmp_path / "test.jsonl"
    _write_full_test(input_path, reverse=True)
    with pytest.raises(ValueError, match="reordered"):
        run_benchmark(
            input_path=input_path,
            output_dir=tmp_path / "result",
            model=ModelSpec("Qwen/fake", REVISION, "teacher"),
            config=BenchmarkConfig(minimum_free_disk_gb=0.000001),
            revision_checker=lambda value: value.revision,
            gpu_checker=dict,
            backend_factory=lambda value, config, trust: FakeBackend(value),
        )


def test_full_benchmark_rejects_unrelated_output_files(tmp_path):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    (output_dir / "another-experiment.txt").write_text("do not overwrite")
    _write_full_test(input_path)
    with pytest.raises(RuntimeError, match="unexpected entries"):
        run_benchmark(
            input_path=input_path,
            output_dir=output_dir,
            model=ModelSpec("Qwen/fake", REVISION, "teacher"),
            config=BenchmarkConfig(minimum_free_disk_gb=0.000001),
            revision_checker=lambda value: value.revision,
            gpu_checker=dict,
            backend_factory=lambda value, config, trust: FakeBackend(value),
        )
