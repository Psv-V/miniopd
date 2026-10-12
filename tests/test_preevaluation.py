import json

import pytest
from miniopd.evaluation.io import read_jsonl
from miniopd.evaluation.preeval import (
    PreEvaluationConfig,
    build_review_queue,
    run_preevaluation,
)
from miniopd.evaluation.schema import GenerationOutput


class FakeTokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        text = "|".join(message["content"] for message in conversation) + "|assistant"
        return list(range(len(text.split()))) if tokenize else text


class FakeBackend:
    name = "vllm"

    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = FakeTokenizer()
        self.calls = 0

    def generate_batch(self, messages, config):
        self.calls += 1
        return [
            GenerationOutput(
                response="The answer is \\boxed{42}.",
                input_tokens=20,
                output_tokens=8,
                finish_reason="stop",
                latency_seconds=0.1,
            )
            for _ in messages
        ]


def _write_test_input(path):
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
        for index in range(100)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_formal_preevaluation_contract_is_fixed():
    assert PreEvaluationConfig().generation_config().dtype == "bfloat16"
    with pytest.raises(ValueError, match="first 100"):
        PreEvaluationConfig(limit=99)
    with pytest.raises(ValueError, match="bfloat16"):
        PreEvaluationConfig(dtype="float16")


def test_review_queue_is_deterministic_and_keeps_all_diagnostics():
    records = [
        {
            "sample_id": "a",
            "correct": True,
            "format_status": "ok",
            "truncated": False,
        },
        {
            "sample_id": "b",
            "correct": False,
            "format_status": "format_error",
            "truncated": False,
        },
        {
            "sample_id": "c",
            "correct": False,
            "format_status": "ok",
            "truncated": True,
        },
    ]
    first = build_review_queue(records, seed=42, samples_per_group=1)
    second = build_review_queue(records, seed=42, samples_per_group=1)
    assert first == second
    assert [row["sample_id"] for row in first["all_format_errors"]] == ["b"]
    assert [row["sample_id"] for row in first["all_truncated"]] == ["c"]


def test_preevaluation_creates_reviewable_artifacts_and_resumes(tmp_path):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "run"
    _write_test_input(input_path)
    backend = FakeBackend("fake/qwen")

    def backend_factory(model_id, config, trust_remote_code):
        assert model_id == "fake/qwen"
        assert config.dtype == "bfloat16"
        assert trust_remote_code is False
        return backend

    summary = run_preevaluation(
        input_path=input_path,
        output_dir=output_dir,
        model_id="fake/qwen",
        config=PreEvaluationConfig(),
        backend_factory=backend_factory,
    )
    assert summary["total"] == 100
    assert summary["exact_match"] == 1.0
    assert summary["scope"].endswith("not_official_result")
    assert len(read_jsonl(output_dir / "predictions.jsonl")) == 100
    assert len(read_jsonl(output_dir / "prompt_debug.jsonl")) == 100
    for name in (
        "resolved_config.yaml",
        "environment.json",
        "summary.json",
        "review_queue.json",
    ):
        assert (output_dir / name).is_file()

    calls = backend.calls
    run_preevaluation(
        input_path=input_path,
        output_dir=output_dir,
        model_id="fake/qwen",
        config=PreEvaluationConfig(),
        backend_factory=backend_factory,
    )
    assert backend.calls == calls
