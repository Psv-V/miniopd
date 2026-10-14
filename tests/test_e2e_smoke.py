import json
from pathlib import Path

import torch
from miniopd.integration.e2e_smoke import (
    build_e2e_commands,
    check_e2e_artifacts,
    record_tensor_audit,
    select_smoke_rows,
    validate_smoke_config,
    validate_tensor_audit,
)
from miniopd.integration.verl_opd import load_yaml

ROOT = Path(__file__).parents[1]
OPD_CONFIG = ROOT / "configs/opd/qwen05b_sampled_k1.yaml"
SMOKE_CONFIG = ROOT / "configs/opd/e2e_smoke.yaml"


def _row(index: int, split: str) -> dict:
    return {
        "data_source": "openai/gsm8k",
        "prompt": [],
        "ability": "math",
        "reward_model": {"ground_truth": str(index), "style": "rule"},
        "extra_info": {
            "sample_id": f"{split}-{index}",
            "split": split,
            "question": f"question {index}",
            "question_hash": f"hash-{split}-{index}",
        },
    }


def _overrides(command: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in command[3:])


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_smoke_selection_is_fixed_and_independent_of_source_order() -> None:
    rows = [_row(index, "opd_train") for index in range(40)]
    first = select_smoke_rows(rows, split="opd_train", count=32, seed=42)
    second = select_smoke_rows(
        list(reversed(rows)), split="opd_train", count=32, seed=42
    )

    assert [row["extra_info"]["sample_id"] for row in first] == [
        row["extra_info"]["sample_id"] for row in second
    ]
    assert len(first) == 32


def test_e2e_commands_train_two_steps_then_explicitly_resume_to_five(
    tmp_path: Path,
) -> None:
    opd = load_yaml(OPD_CONFIG)
    smoke = load_yaml(SMOKE_CONFIG)
    validate_smoke_config(smoke)
    phase1, phase2 = build_e2e_commands(
        opd_config=opd,
        smoke_config=smoke,
        student_checkpoint=tmp_path / "student",
        teacher_model=tmp_path / "teacher",
        train_data=tmp_path / "train.parquet",
        validation_data=tmp_path / "validation.parquet",
        run_dir=tmp_path / "run",
    )
    first = _overrides(phase1)
    second = _overrides(phase2)

    assert first["data.train_batch_size"] == "4"
    assert first["data.max_response_length"] == "128"
    assert first["actor_rollout_ref.rollout.n"] == "1"
    assert first["actor_rollout_ref.rollout.max_model_len"] == "385"
    assert first["actor_rollout_ref.rollout.agent.num_workers"] == "4"
    assert first["trainer.total_training_steps"] == "2"
    assert first["trainer.save_freq"] == "5"
    assert first["trainer.resume_mode"] == "disable"
    assert first["trainer.test_freq"] == "-1"
    assert second["trainer.total_training_steps"] == "5"
    assert second["trainer.save_freq"] == "5"
    assert second["trainer.resume_mode"] == "resume_path"
    assert "global_step_2" in second["trainer.resume_from_path"]
    assert second["trainer.test_freq"] == "5"
    assert second["trainer.log_val_generations"] == "8"
    assert len(first) == len(set(first))
    assert len(second) == len(set(second))


def _tensor_audit(step: int) -> dict:
    return {
        "step": step,
        "input_ids_shape": [4, 20],
        "shifted_logits_shape": [4, 19, 100],
        "target_ids_shape": [4, 19],
        "response_mask_shape": [4, 19],
        "response_token_count": 32,
        "teacher_requires_grad": False,
        "advantage_requires_grad": False,
        "all_finite": True,
        "sampled_token_reward_min": -1.0,
        "sampled_token_reward_max": 1.0,
    }


def test_tensor_audit_checks_shift_mask_stop_gradient_and_finiteness() -> None:
    assert validate_tensor_audit([_tensor_audit(1)])
    invalid = _tensor_audit(1)
    invalid["target_ids_shape"] = [4, 20]
    assert not validate_tensor_audit([invalid])


def test_tensor_audit_record_is_derived_from_real_tensors(tmp_path: Path) -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True]])
    teacher = torch.tensor([[0.0, -1.0, -2.0]])
    old = torch.tensor([[-0.5, -0.5, -0.5]])
    advantage = (teacher - old).detach().masked_fill(~mask, 0.0)
    record = record_tensor_audit(
        tmp_path / "tensor_audit.jsonl",
        step=1,
        input_ids=input_ids,
        shifted_logits=torch.zeros(1, 3, 10, requires_grad=True),
        target_ids=input_ids[:, 1:],
        response_mask=mask,
        teacher_log_probs=teacher,
        old_student_log_probs=old,
        advantage=advantage,
        current_student_log_probs=torch.zeros(1, 3, requires_grad=True),
    )

    assert validate_tensor_audit([record])
    assert record["response_token_count"] == 2
    assert record["sampled_token_reward_min"] == -1.5
    assert record["sampled_token_reward_max"] == -0.5
    invalid = _tensor_audit(1)
    invalid["teacher_requires_grad"] = True
    assert not validate_tensor_audit([invalid])
    invalid = _tensor_audit(1)
    invalid["all_finite"] = False
    assert not validate_tensor_audit([invalid])


def test_complete_e2e_artifacts_pass_auditor(tmp_path: Path) -> None:
    smoke = load_yaml(SMOKE_CONFIG)
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "inputs/subset_manifest.json").write_text("{}\n")
    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    phase1.mkdir()
    phase2.mkdir()
    (phase1 / "run.log").write_text(
        "rollout\nactor/distillation/loss=0.1\nactor/grad_norm=1.0\n"
    )
    (phase2 / "run.log").write_text(
        "Resuming from /run/global_step_2, setting global step to 2\n"
        "rollout\nactor/distillation/loss=0.1\nactor/grad_norm=1.0\n"
        "val/accuracy=0.25\n"
    )

    def sync(version: int) -> dict:
        return {
            "actor_version": version,
            "rollout_version_after": version,
            "policy_lag_after": 0,
        }

    _append_jsonl(phase1 / "sync_versions.jsonl", [sync(i) for i in range(3)])
    _append_jsonl(phase2 / "sync_versions.jsonl", [sync(i) for i in range(2, 6)])
    for step in (2, 5):
        checkpoint = tmp_path / "checkpoints" / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.bin").write_bytes(b"state")
    for directory in (tmp_path / "rollouts", tmp_path / "validation"):
        directory.mkdir()
        (directory / "samples.jsonl").write_text("{}\n")
    _append_jsonl(tmp_path / "metrics.jsonl", [{"step": 1, "loss": 0.1}])
    _append_jsonl(tmp_path / "diagnostics.jsonl", [{"step": 1, "entropy": 1.0}])
    _append_jsonl(tmp_path / "tensor_audit.jsonl", [_tensor_audit(1)])

    report = check_e2e_artifacts(tmp_path, smoke)
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_non_finite_log_fails_e2e_audit(tmp_path: Path) -> None:
    smoke = load_yaml(SMOKE_CONFIG)
    (tmp_path / "phase1").mkdir(parents=True)
    (tmp_path / "phase2").mkdir()
    (tmp_path / "phase1/run.log").write_text("loss=nan\n")
    (tmp_path / "phase2/run.log").write_text("loss=0.1\n")

    report = check_e2e_artifacts(tmp_path, smoke)
    assert report["passed"] is False
    assert report["checks"]["finite_logs"] is False
