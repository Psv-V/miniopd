import json

import pytest
from miniopd.evaluation.io import load_evaluation_examples, read_jsonl
from miniopd.evaluation.pipeline import (
    rescore_records,
    run_evaluation,
    summarize_records,
)
from miniopd.evaluation.schema import GenerationConfig, GenerationOutput


class FakeBackend:
    name = "fake"
    model_id = "fake/math-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate_batch(self, messages, config):
        self.calls += 1
        answers = ["\\boxed{42}", "No numeric answer"]
        return [
            GenerationOutput(
                response=answers[int(item[-1]["content"][-1]) - 1],
                input_tokens=10,
                output_tokens=3,
                finish_reason="stop",
                latency_seconds=0.5,
            )
            for item in messages
        ]


@pytest.fixture
def examples(tmp_path):
    path = tmp_path / "input.jsonl"
    rows = [
        {
            "sample_id": f"sample-{number}",
            "split": "test",
            "question": f"Question {number}",
            "reference_answer": "42",
            "messages": [
                {"role": "system", "content": "Reason."},
                {"role": "user", "content": f"Question {number}"},
            ],
        }
        for number in (1, 2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return load_evaluation_examples(path)


def test_generation_config_rejects_non_greedy_mismatch():
    with pytest.raises(ValueError, match="greedy"):
        GenerationConfig(temperature=0.5, do_sample=False)


def test_pipeline_writes_stable_schema_and_resumes(examples, tmp_path):
    output = tmp_path / "predictions.jsonl"
    backend = FakeBackend()
    config = GenerationConfig(batch_size=2)
    summary = run_evaluation(
        examples,
        backend,
        output,
        config=config,
    )

    rows = read_jsonl(output)
    assert backend.calls == 1
    assert len(rows) == 2
    assert rows[0]["correct"] is True
    assert rows[0]["parsed_answer"]["normalized_value"] == "42"
    assert rows[1]["format_status"] == "format_error"
    assert rows[1]["error_type"] == "format_error"
    assert summary["exact_match"] == 0.5
    assert summary["format_error"] == 1
    assert summary["average_output_tokens"] == 3

    run_evaluation(examples, backend, output, config=config)
    assert backend.calls == 1
    assert read_jsonl(output) == rows


def test_rescoring_is_deterministic_and_backend_independent(examples, tmp_path):
    output = tmp_path / "predictions.jsonl"
    run_evaluation(examples, FakeBackend(), output)
    original = read_jsonl(output)
    once = rescore_records(original)
    twice = rescore_records(once)
    assert once == twice
    assert summarize_records(once) == summarize_records(twice)


def test_backend_failure_is_recorded_for_manual_review(examples, tmp_path):
    class BrokenBackend(FakeBackend):
        def generate_batch(self, messages, config):
            raise RuntimeError("out of memory")

    output = tmp_path / "failed.jsonl"
    summary = run_evaluation(examples, BrokenBackend(), output)
    rows = read_jsonl(output)
    assert summary["error_counts"] == {"runtime_error": 2}
    assert all("out of memory" in row["runtime_message"] for row in rows)
    assert all(row["question"] for row in rows)
