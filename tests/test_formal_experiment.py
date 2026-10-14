import json
import math
from pathlib import Path

import pytest
from miniopd.evaluation.pipeline import summarize_records
from miniopd.experiments.formal import (
    ExperimentError,
    build_evaluation_plan,
    build_final_report,
    build_validation_metrics,
    load_formal_config,
    select_validation_checkpoint,
)
from miniopd.integration.verl_opd import build_verl_command, load_yaml

ROOT = Path(__file__).parents[1]
FORMAL_CONFIG = ROOT / "configs/experiments/formal_sampled_opd.yaml"
OPD_CONFIG = ROOT / "configs/opd/qwen05b_sampled_k1.yaml"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _checkpoint(root: Path, step: int) -> Path:
    path = root / f"global_step_{step}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(f"model-{step}".encode())
    (path / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
    return path


def _validation_row(checkpoint: Path, step: int, accuracy: float) -> dict:
    return {
        "step": step,
        "split": "validation",
        "checkpoint": str(checkpoint),
        "exact_match": accuracy,
        "format_error_ratio": 0.01,
        "truncated_ratio": 0.02,
        "total": 1000,
    }


def _raw_validation_rows(step: int, correct: int) -> list[dict]:
    return [
        {
            "input": "question",
            "output": "The answer is \\boxed{42}."
            if index < correct
            else "The answer is \\boxed{41}.",
            "gts": "#### 42",
            "score": 1.0 if index < correct else 0.0,
            "step": step,
            "truncated": index == 999,
        }
        for index in range(1000)
    ]


def test_validation_dumps_are_rescored_and_bound_to_checkpoints(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _checkpoint(checkpoint_root, 20)
    validation_dir = tmp_path / "validation"
    dump = validation_dir / "20.jsonl"
    _write_jsonl(dump, _raw_validation_rows(20, 600))

    rows = build_validation_metrics(
        validation_dir=validation_dir,
        checkpoint_root=checkpoint_root,
        output_path=tmp_path / "validation_metrics.jsonl",
        formal_config=load_formal_config(FORMAL_CONFIG),
    )

    assert rows[0]["correct"] == 600
    assert rows[0]["exact_match"] == 0.6
    assert rows[0]["truncated"] == 1
    assert rows[0]["source_sha256"]
    assert rows[0]["scorer"].endswith("evaluate_answer")


def test_validation_metrics_require_explicit_truncation_evidence(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _checkpoint(checkpoint_root, 20)
    rows = _raw_validation_rows(20, 600)
    rows[0].pop("truncated")
    _write_jsonl(tmp_path / "validation/20.jsonl", rows)

    with pytest.raises(ExperimentError, match="explicit truncated flag"):
        build_validation_metrics(
            validation_dir=tmp_path / "validation",
            checkpoint_root=checkpoint_root,
            output_path=tmp_path / "validation_metrics.jsonl",
            formal_config=load_formal_config(FORMAL_CONFIG),
        )


def test_formal_command_uses_full_epoch_validation_checkpoint_and_resume(
    tmp_path: Path,
) -> None:
    command = build_verl_command(
        load_yaml(OPD_CONFIG),
        mode="full",
        student_checkpoint=tmp_path / "student",
        teacher_model=tmp_path / "teacher",
        train_data=tmp_path / "opd_train.parquet",
        validation_data=tmp_path / "opd_validation.parquet",
        run_dir=tmp_path / "run",
    )
    overrides = dict(item.split("=", 1) for item in command[3:])

    assert overrides["trainer.total_epochs"] == "1"
    assert "trainer.total_training_steps" not in overrides
    assert overrides["trainer.save_freq"] == "40"
    assert overrides["trainer.test_freq"] == "40"
    assert overrides["trainer.max_actor_ckpt_to_keep"] == "4"
    assert overrides["trainer.resume_mode"] == "auto"
    assert overrides["trainer.log_val_generations"] == "8"
    assert overrides["data.train_batch_size"] == "23"
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == "23"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "769"
    assert overrides["actor_rollout_ref.rollout.agent.num_workers"] == "2"


def test_checkpoint_selection_uses_validation_and_deterministic_tie_breaks(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    step20 = _checkpoint(checkpoints, 20)
    step40 = _checkpoint(checkpoints, 40)
    metrics = tmp_path / "validation_metrics.jsonl"
    first = _validation_row(step20, 20, 0.50)
    second = _validation_row(step40, 40, 0.50)
    second["format_error_ratio"] = 0.02
    _write_jsonl(metrics, [first, second])

    report = select_validation_checkpoint(
        metrics_path=metrics,
        checkpoint_root=checkpoints,
        formal_config=load_formal_config(FORMAL_CONFIG),
        output_path=tmp_path / "selection.json",
    )

    assert report["selected"]["step"] == 20
    assert report["selection_data_split"] == "validation"
    assert report["test_metrics_read"] is False
    assert report["selected_checkpoint"]["total_size_bytes"] > 0
    assert len(report["selected_checkpoint"]["manifest_sha256"]) == 64


def test_checkpoint_selection_refuses_existing_test_predictions(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoint = _checkpoint(checkpoints, 20)
    metrics = tmp_path / "validation_metrics.jsonl"
    _write_jsonl(metrics, [_validation_row(checkpoint, 20, 0.5)])
    test_root = tmp_path / "test-results"
    (test_root / "base").mkdir(parents=True)
    (test_root / "base/predictions.jsonl").write_text("{}\n")

    with pytest.raises(ExperimentError, match="test predictions already exist"):
        select_validation_checkpoint(
            metrics_path=metrics,
            checkpoint_root=checkpoints,
            formal_config=load_formal_config(FORMAL_CONFIG),
            output_path=tmp_path / "selection.json",
            forbidden_test_roots=[test_root],
        )


def test_evaluation_plan_has_all_four_models_and_one_shared_protocol(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    _write_json(selection, {"selection_data_split": "validation"})
    plan = build_evaluation_plan(
        formal_config=load_formal_config(FORMAL_CONFIG),
        test_data=tmp_path / "test.parquet",
        evaluation_config=ROOT / "configs/evaluation/gsm8k_greedy.yaml",
        output_root=tmp_path / "evaluations",
        selection_report=selection,
        models={
            "base": ("Qwen/base", "a" * 40),
            "sft": ("/models/sft", None),
            "opd": ("/models/opd", None),
            "teacher": ("Qwen/teacher", "b" * 40),
        },
    )

    assert tuple(plan["commands"]) == ("base", "sft", "opd", "teacher")
    for command in plan["commands"].values():
        assert "--dtype" not in command
        assert "--config" in command
    assert plan["protocol"]["dtype"] == "bfloat16"


def _write_evaluation(directory: Path, role: str, correct: int) -> None:
    generation = {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "max_new_tokens": 512,
        "batch_size": 8,
        "dtype": "bfloat16",
        "seed": 42,
    }
    rows = [
        {
            "sample_id": f"gsm8k-test-{index:05d}",
            "split": "test",
            "model": f"model-{role}",
            "generation_config": generation,
            "correct": index < correct,
            "error_type": None,
            "truncated": False,
            "output_tokens": 10,
            "latency_seconds": 0.1,
            "runtime_message": None,
        }
        for index in range(1319)
    ]
    _write_jsonl(directory / "predictions.jsonl", rows)
    _write_json(directory / "summary.json", summarize_records(rows))


def test_final_report_computes_gains_and_efficiency_from_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "training/checkpoints"
    checkpoint = _checkpoint(checkpoint_root, 20)
    validation_metrics = tmp_path / "training/validation_metrics.jsonl"
    _write_jsonl(validation_metrics, [_validation_row(checkpoint, 20, 0.5)])
    selection_path = tmp_path / "selection.json"
    select_validation_checkpoint(
        metrics_path=validation_metrics,
        checkpoint_root=checkpoint_root,
        formal_config=load_formal_config(FORMAL_CONFIG),
        output_path=selection_path,
    )
    training = tmp_path / "training"
    _write_json(
        training / "run_status.json",
        {
            "status": "complete",
            "exit_code": 0,
            "wall_seconds": 3600,
            "visible_gpu_count": 2,
        },
    )
    _write_jsonl(
        training / "run_history.jsonl",
        [
            {"status": "failed", "error": "OOM"},
            {"status": "complete", "exit_code": 0},
        ],
    )
    _write_jsonl(
        training / "metrics.jsonl",
        [
            {
                "rollout_tokens_per_second": 10.0,
                "teacher_scoring_tokens_per_second": 20.0,
                "training_tokens_per_second": 30.0,
                "gpu_memory_bytes": [{"max_allocated": 100, "max_reserved": 120}],
            }
        ],
    )
    counts = {"base": 100, "sft": 200, "opd": 300, "teacher": 500}
    evaluation_dirs = {}
    for role, correct in counts.items():
        directory = tmp_path / "evaluations" / role
        _write_evaluation(directory, role, correct)
        evaluation_dirs[role] = directory
    plan_path = tmp_path / "evaluations/evaluation_plan.json"
    plan = build_evaluation_plan(
        formal_config=load_formal_config(FORMAL_CONFIG),
        test_data=tmp_path / "test.parquet",
        evaluation_config=ROOT / "configs/evaluation/gsm8k_greedy.yaml",
        output_root=tmp_path / "evaluations",
        selection_report=selection_path,
        models={
            role: (f"model-{role}", None) for role in ("base", "sft", "opd", "teacher")
        },
    )
    _write_json(plan_path, plan)

    report = build_final_report(
        formal_config=load_formal_config(FORMAL_CONFIG),
        selection_report=selection_path,
        evaluation_plan=plan_path,
        training_run_dir=training,
        evaluation_dirs=evaluation_dirs,
        output_path=tmp_path / "final_report.json",
    )

    assert math.isclose(report["gains"]["sft_gain"], 100 / 1319)
    assert math.isclose(report["gains"]["opd_additional_gain"], 100 / 1319)
    assert math.isclose(report["gains"]["gap_recovery"], 0.5)
    assert report["efficiency"]["gpu_hours"] == 2.0
    assert report["efficiency"]["peak_gpu_memory_bytes"] == 120
    assert len(report["efficiency"]["failed_or_interrupted_attempts"]) == 1
    assert report["all_values_from_saved_artifacts"] is True
