from pathlib import Path

import pytest
from miniopd.diagnostics.finalize_formal import (
    GIB,
    _parser,
    normalize_formal_step_metrics,
    validate_step_coverage,
)


def _row(step: int) -> dict[str, float | int | str]:
    return {
        "step": step,
        "source": "run.log",
        "response_length/mean": 100.5,
        "timing_s/gen": 2.0,
        "timing_s/update_actor": 4.0,
        "perf/total_num_tokens": 8000,
        "actor/perf/max_memory_allocated_gb": 9.0,
        "actor/perf/max_memory_reserved_gb": 10.0,
    }


def test_normalize_formal_metrics_preserves_source_and_computes_rates() -> None:
    result = normalize_formal_step_metrics([_row(1)], rollout_samples_per_step=46)

    assert len(result) == 1
    assert result[0]["source"] == "run.log"
    assert result[0]["rollout_response_tokens"] == 4623
    assert result[0]["rollout_tokens_per_second"] == 2311.5
    assert result[0]["training_tokens_per_second"] == 2000
    assert result[0]["gpu_memory_bytes"] == [
        {
            "device": 0,
            "scope": "actor_training_worker",
            "max_allocated": 9 * GIB,
            "max_reserved": 10 * GIB,
        }
    ]
    assert "teacher_scoring_tokens_per_second" not in result[0]


def test_normalize_formal_metrics_rejects_missing_evidence() -> None:
    row = _row(1)
    del row["timing_s/gen"]

    with pytest.raises(ValueError, match="timing_s/gen"):
        normalize_formal_step_metrics([row], rollout_samples_per_step=46)


def test_step_coverage_requires_checkpoint_for_allowed_missing_step(
    tmp_path: Path,
) -> None:
    rows = [_row(1), _row(3)]

    with pytest.raises(ValueError, match="no checkpoint evidence"):
        validate_step_coverage(
            rows,
            final_step=3,
            allowed_missing_steps=[2],
            checkpoint_root=tmp_path,
        )

    (tmp_path / "global_step_2").mkdir()
    assert validate_step_coverage(
        rows,
        final_step=3,
        allowed_missing_steps=[2],
        checkpoint_root=tmp_path,
    ) == [2]


def test_step_coverage_rejects_unapproved_gap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicitly allowed"):
        validate_step_coverage(
            [_row(1), _row(3)],
            final_step=3,
            allowed_missing_steps=[],
            checkpoint_root=tmp_path,
        )


def test_formal_finalizer_cli_maps_repeatable_arguments() -> None:
    args = _parser().parse_args(
        [
            "--run-dir",
            "run",
            "--log",
            "run.log",
            "resume.log",
            "--student-model",
            "student",
            "--teacher-model",
            "teacher",
            "--config",
            "opd.yaml",
            "--rollout-step",
            "120",
            "--final-step",
            "151",
            "--allow-missing-step",
            "40",
            "--rollout-samples-per-step",
            "46",
        ]
    )

    assert args.log_paths == [Path("run.log"), Path("resume.log")]
    assert args.allowed_missing_steps == [40]
