import json
from pathlib import Path

import pytest
from miniopd.data.gsm8k import build_manifest, build_verl_prompt_records
from miniopd.integration.verl_opd import (
    IntegrationError,
    SyncAuditRecorder,
    build_verl_command,
    check_artifacts,
    load_yaml,
    validate_recipe,
    validate_verl_data,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/opd/qwen05b_sampled_k1.yaml"


def _overrides(command: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in command[3:])


def test_recipe_and_smoke_command_match_stage_12_contract(tmp_path: Path) -> None:
    config = load_yaml(CONFIG)
    validate_recipe(config)
    command = build_verl_command(
        config,
        mode="smoke",
        student_checkpoint=tmp_path / "student",
        teacher_model=tmp_path / "teacher",
        train_data=tmp_path / "train.parquet",
        validation_data=tmp_path / "validation.parquet",
        run_dir=tmp_path / "run",
    )
    overrides = _overrides(command)

    assert command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert overrides["actor_rollout_ref.model.use_remove_padding"] == "False"
    assert (
        overrides["+actor_rollout_ref.model.override_config.attn_implementation"]
        == "sdpa"
    )
    assert overrides["actor_rollout_ref.rollout.n"] == "2"
    assert overrides["actor_rollout_ref.rollout.temperature"] == "1.0"
    assert overrides["actor_rollout_ref.rollout.top_p"] == "1.0"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "769"
    assert overrides["actor_rollout_ref.rollout.agent.num_workers"] == "4"
    log_prob_micro_batch = overrides[
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
    ]
    assert log_prob_micro_batch == "1"
    assert overrides["distillation.distillation_loss.loss_mode"] == "k1"
    assert overrides["distillation.distillation_loss.use_policy_gradient"] == "True"
    assert overrides["distillation.distillation_loss.use_task_rewards"] == "False"
    assert overrides["actor_rollout_ref.actor.optim.lr"] == "1e-06"
    assert overrides["actor_rollout_ref.actor.grad_clip"] == "1.0"
    assert overrides["data.train_batch_size"] == "2"
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == "2"
    assert overrides["trainer.total_training_steps"] == "1"
    assert overrides["trainer.n_gpus_per_node"] == "1"
    assert overrides["distillation.n_gpus_per_node"] == "1"


def test_recipe_rejects_uneven_agent_loop_chunks(tmp_path: Path) -> None:
    config = load_yaml(CONFIG)
    config["smoke"]["agent_loop_num_workers"] = 3
    with pytest.raises(IntegrationError, match="trajectory batch"):
        build_verl_command(
            config,
            mode="smoke",
            student_checkpoint=tmp_path / "student",
            teacher_model=tmp_path / "teacher",
            train_data=tmp_path / "train.parquet",
            validation_data=tmp_path / "validation.parquet",
            run_dir=tmp_path / "run",
        )


def test_recipe_rejects_top1_substitution() -> None:
    config = load_yaml(CONFIG)
    config["method"] = "top-1"
    with pytest.raises(IntegrationError, match="method"):
        validate_recipe(config)


def _raw(count: int, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "question": f"{prefix} question {index}?",
            "answer": f"work\n#### {index}",
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_verl_data_is_exact_manifest_match_without_leakage(tmp_path: Path) -> None:
    from miniopd.data.gsm8k import SplitSizes

    train = _raw(5, "train")
    test = _raw(1, "test")
    manifest = build_manifest(
        train,
        test,
        dataset_revision="revision",
        code_version="code",
        sizes=SplitSizes(sft=1, opd_train=2, validation=2),
    )
    opd = build_verl_prompt_records(train, test, manifest, split="opd_train")
    validation = build_verl_prompt_records(train, test, manifest, split="validation")
    manifest_path = tmp_path / "manifest.jsonl"
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(train_path, opd)
    _write_jsonl(validation_path, validation)

    report = validate_verl_data(
        train_path,
        validation_path,
        manifest_path,
        train_count=2,
        validation_count=2,
    )
    assert report["opd_train"]["rows"] == 2
    assert report["validation"]["rows"] == 2

    validation[0]["extra_info"]["sample_id"] = opd[0]["extra_info"]["sample_id"]
    _write_jsonl(validation_path, validation)
    with pytest.raises(IntegrationError):
        validate_verl_data(
            train_path,
            validation_path,
            manifest_path,
            train_count=2,
            validation_count=2,
        )


def test_sync_audit_uses_confirmed_upstream_events(tmp_path: Path) -> None:
    path = tmp_path / "sync_versions.jsonl"
    recorder = SyncAuditRecorder(path)
    assert not recorder.observe("starting actor update")
    assert recorder.observe("vllm rollout update_weights done")
    assert recorder.observe("vllm rollout update_weights done")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["actor_version"] for row in rows] == [0, 1]
    assert all(row["policy_lag_after"] == 0 for row in rows)


def test_smoke_artifact_acceptance_requires_complete_training_chain(
    tmp_path: Path,
) -> None:
    for name in ("preflight.json", "launch_command.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "resolved_config.yaml").write_text("method: sampled-token\n")
    (tmp_path / "run.log").write_text(
        "rollout generation\n"
        "distillation/reward=0.1\n"
        "actor/grad_norm=1.0\n"
        "timing_s/update_weights=0.5\n",
        encoding="utf-8",
    )
    recorder = SyncAuditRecorder(tmp_path / "sync_versions.jsonl")
    recorder.observe("update_weights done")
    recorder.observe("update_weights done")
    (tmp_path / "checkpoints").mkdir()
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    (rollout_dir / "batch.jsonl").write_text("{}\n", encoding="utf-8")

    assert check_artifacts(tmp_path, "smoke")["passed"] is True
