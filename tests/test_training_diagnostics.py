import json
import math
from pathlib import Path

import torch
from miniopd.diagnostics.opd import (
    DiagnosticsConfig,
    DiagnosticsRecorder,
    compute_periodic_diagnostics,
    position_segment_masks,
    select_fixed_sample_ids,
    summarize_step_metrics,
)


def test_position_segments_partition_only_valid_response_tokens() -> None:
    mask = torch.tensor(
        [
            [False, True, True, True, True, True, True, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    segments = position_segment_masks(mask)

    assert segments.shape == (3, 2, 8)
    assert torch.equal(segments.sum(dim=0).bool(), mask)
    assert not (segments.sum(dim=0) > 1).any()
    assert [int(item.sum()) for item in segments[:, 0]] == [2, 2, 2]
    assert [int(item.sum()) for item in segments[:, 1]] == [1, 1, 1]


def test_fixed_diagnostic_batch_is_deterministic() -> None:
    sample_ids = [f"sample-{index}" for index in range(20)]
    first = select_fixed_sample_ids(sample_ids, sample_size=4, seed=42)
    second = select_fixed_sample_ids(list(reversed(sample_ids)), sample_size=4, seed=42)

    assert first == second
    assert len(first) == 4


def test_step_metrics_use_only_response_mask_and_existing_values() -> None:
    metrics = summarize_step_metrics(
        step=7,
        token_reward=torch.tensor([[100.0, 1.0, 3.0], [2.0, 4.0, -100.0]]),
        response_mask=torch.tensor([[False, True, True], [True, True, False]]),
        policy_loss=torch.tensor(0.5, requires_grad=True),
        distillation_loss=0.25,
        gradient_norm=1.5,
        rollout_correct=[True, False],
        truncated=[False, True],
        format_error=[False, False],
        timing_seconds={"rollout": 2.0, "teacher_scoring": 1.0, "training": 4.0},
        token_counts={"rollout": 20, "teacher_scoring": 30, "training": 40},
        gpu_memory_bytes=[{"device": 0, "allocated": 123}],
    )

    assert metrics["token_reward_mean"] == 2.5
    assert metrics["token_reward_min"] == 1.0
    assert metrics["token_reward_max"] == 4.0
    assert metrics["response_length_mean"] == 2.0
    assert metrics["rollout_accuracy"] == 0.5
    assert metrics["truncated_ratio"] == 0.5
    assert metrics["format_error_ratio"] == 0.0
    assert metrics["rollout_tokens_per_second"] == 10.0
    assert metrics["teacher_scoring_tokens_per_second"] == 30.0
    assert metrics["training_tokens_per_second"] == 10.0


def test_periodic_topk_entropy_and_segments_match_hand_values() -> None:
    config = DiagnosticsConfig(top_k=2, sample_size=1)
    student = torch.tensor(
        [
            [
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [
                [1.0, 4.0, 3.0, 2.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 4.0, 3.0, 2.0],
            ]
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True]])
    result = compute_periodic_diagnostics(
        step=50,
        student_logits=student,
        teacher_logits=teacher,
        token_reward=torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
        response_mask=mask,
        config=config,
        diagnostic_seconds=0.5,
        reference_step_seconds=10.0,
    )

    assert result["entropy_kind"] == "exact_full_vocabulary"
    assert math.isclose(result["top2_overlap_ratio"], 2 / 3, rel_tol=1e-6)
    assert result["segments"]["front"]["top2_overlap_ratio"] == 0.5
    assert result["segments"]["middle"]["top2_overlap_ratio"] == 1.0
    assert result["segments"]["back"]["top2_overlap_ratio"] == 0.5
    assert result["segments"]["front"]["token_reward_mean"] == 1.0
    assert result["segments"]["middle"]["token_reward_mean"] == 2.0
    assert result["segments"]["back"]["token_reward_mean"] == 3.0
    assert result["diagnostic_overhead_ratio"] == 0.05
    assert result["diagnostic_overhead_within_budget"] is True

    # The diagnostic API returns detached Python values and never builds a graph.
    assert not any(isinstance(value, torch.Tensor) for value in result.values())
    assert student.grad is None
    assert teacher.grad is None


def test_overlap_advantage_and_probability_masses_match_hand_values() -> None:
    student = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 4.0, 3.0, 2.0]]])
    result = compute_periodic_diagnostics(
        step=0,
        student_logits=student,
        teacher_logits=teacher,
        token_reward=torch.tensor([[0.0]]),
        response_mask=torch.tensor([[True]]),
        config=DiagnosticsConfig(top_k=2, sample_size=1),
    )
    student_logp = torch.log_softmax(student, dim=-1)[0, 0, 1]
    teacher_logp = torch.log_softmax(teacher, dim=-1)[0, 0, 1]

    assert math.isclose(result["top2_overlap_ratio"], 0.5)
    assert math.isclose(
        result["top2_overlap_token_advantage"],
        float(teacher_logp - student_logp),
        rel_tol=1e-6,
    )
    assert math.isclose(
        result["top2_student_overlap_probability_mass"],
        float(student_logp.exp()),
        rel_tol=1e-6,
    )
    assert math.isclose(
        result["top2_teacher_overlap_probability_mass"],
        float(teacher_logp.exp()),
        rel_tol=1e-6,
    )


def test_diagnostic_schedule_and_append_only_metrics(tmp_path: Path) -> None:
    config = DiagnosticsConfig(interval_steps=50)
    assert config.should_run(0)
    assert config.should_run(50)
    assert not config.should_run(49)

    recorder = DiagnosticsRecorder(tmp_path, DiagnosticsConfig())
    recorder.record_step({"step": 0, "value": 1})
    recorder.record_step({"step": 50, "value": 2})
    rows = [json.loads(line) for line in recorder.metrics_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 50]


def test_recorder_locks_fixed_batch_and_enforces_schedule(tmp_path: Path) -> None:
    config = DiagnosticsConfig(interval_steps=50, sample_size=2)
    recorder = DiagnosticsRecorder(tmp_path, config)
    candidates = ["a", "b", "c", "d"]
    selected = recorder.initialize_fixed_batch(candidates)

    assert recorder.initialize_fixed_batch(list(reversed(candidates))) == selected
    recorder.record_step({"step": 1, "token_reward_mean": 0.1})
    recorder.record_periodic({"step": 50, "top16_overlap_ratio": 0.5})
    assert recorder.metrics_path.is_file()
    assert recorder.diagnostics_path.is_file()

    try:
        recorder.record_periodic({"step": 51})
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("off-schedule diagnostic was accepted")
