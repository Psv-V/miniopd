import miniopd.evaluation.backends as evaluation_backends
from miniopd.generation.coldstart import (
    ColdStartConfig,
    _create_backend,
    build_sft_rows,
    filter_reason,
    summarize_filter,
)


def _record(**updates):
    row = {
        "sample_id": "gsm8k-train-00001",
        "question": "What is 40 + 2?",
        "ground_truth": "42",
        "raw_response": "40 + 2 = 42. \\boxed{42}",
        "parsed_answer": {"normalized_value": "42"},
        "correct": True,
        "format_status": "ok",
        "truncated": False,
        "runtime_message": None,
        "output_tokens": 8,
        "latency_seconds": 0.5,
        "model": "Qwen/teacher",
        "model_revision": "a" * 40,
    }
    row.update(updates)
    return row


def test_coldstart_config_is_fixed_and_seeded():
    generation = ColdStartConfig().generation_config()
    assert generation.temperature == 0.7
    assert generation.top_p == 0.95
    assert generation.seed == 42


def test_filter_rejects_correct_but_truncated_or_repetitive_rows():
    assert filter_reason(_record(truncated=True)) == "truncated"
    repeated = "same three token block " * 3
    assert filter_reason(_record(raw_response=repeated)) == "repetition"


def test_filtered_sft_keeps_teacher_text_and_three_message_contract():
    kept = _record()
    rejected = _record(sample_id="gsm8k-train-00002", correct=False)
    rows = build_sft_rows([kept, rejected])
    assert len(rows) == 1
    assert [message["role"] for message in rows[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert rows[0]["teacher_response"] == kept["raw_response"]


def test_filter_summary_preserves_independent_diagnostics():
    records = [_record(), _record(truncated=True, format_status="format_error")]
    summary = summarize_filter(records)
    assert summary["input_count"] == 2
    assert summary["retained_count"] == 1
    assert summary["truncated_count"] == 1
    assert summary["format_error_count"] == 1


def test_local_model_path_preserves_canonical_provenance(tmp_path, monkeypatch):
    model_path = tmp_path / "teacher"
    model_path.mkdir()
    captured = {}

    class FakeBackend:
        def __init__(self, model_id, *, revision, **kwargs):
            captured.update(model_id=model_id, revision=revision, kwargs=kwargs)
            self.model_id = model_id
            self.model_revision = revision

    monkeypatch.setattr(evaluation_backends, "VLLMBackend", FakeBackend)
    backend = _create_backend(
        "Qwen/teacher",
        "a" * 40,
        model_path,
        ColdStartConfig(),
        False,
    )

    assert captured["model_id"] == str(model_path)
    assert captured["revision"] is None
    assert backend.model_id == "Qwen/teacher"
    assert backend.model_revision == "a" * 40
