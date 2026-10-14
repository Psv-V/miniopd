"""Two-process end-to-end OPD smoke-test orchestration and acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from miniopd.evaluation.io import append_jsonl, write_json
from miniopd.integration.verl_opd import (
    IntegrationError,
    build_verl_command,
    launch,
    load_yaml,
    read_rows,
    replace_hydra_overrides,
    validate_checkpoint,
    validate_recipe,
    validate_runtime,
    validate_verl_checkout,
    validate_verl_data,
    write_resolved_yaml,
)


def validate_smoke_config(config: Mapping[str, Any]) -> None:
    expected = {
        ("data", "train_count"): 32,
        ("data", "validation_count"): 8,
        ("rollout", "n"): 1,
        ("rollout", "max_response_length"): 128,
        ("training", "phase1_steps"): 2,
        ("training", "final_steps"): 5,
    }
    for keys, wanted in expected.items():
        value: Any = config
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                raise IntegrationError(f"missing smoke config key: {'.'.join(keys)}")
            value = value[key]
        if value != wanted:
            raise IntegrationError(
                f"smoke {'.'.join(keys)}={value!r}; expected {wanted!r}"
            )
    phase1 = int(config["training"]["phase1_steps"])
    final = int(config["training"]["final_steps"])
    if not 2 <= phase1 < final <= 5:
        raise IntegrationError(
            "smoke optimizer steps must satisfy 2 <= phase1 < final <= 5"
        )
    if int(config["training"]["train_batch_size"]) <= 0:
        raise IntegrationError("smoke train batch size must be positive")
    if int(config["diagnostics"]["interval_steps"]) != 1:
        raise IntegrationError("smoke diagnostics must run every step")


def _selection_key(seed: int, split: str, sample_id: str) -> bytes:
    value = f"miniopd-e2e-smoke:{seed}:{split}:{sample_id}".encode()
    return hashlib.sha256(value).digest()


def select_smoke_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically select smoke rows without depending on file order."""
    if count <= 0 or count > len(rows):
        raise IntegrationError(f"invalid {split} smoke count {count}")
    indexed = []
    seen: set[str] = set()
    for row in rows:
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping) or extra.get("split") != split:
            raise IntegrationError(f"invalid {split} row in smoke source")
        sample_id = str(extra.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise IntegrationError(f"missing or duplicate sample ID {sample_id!r}")
        seen.add(sample_id)
        indexed.append((_selection_key(seed, split, sample_id), dict(row)))
    return [row for _, row in sorted(indexed, key=lambda item: item[0])[:count]]


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import datasets

    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list([dict(row) for row in rows]).to_parquet(str(path))


def prepare_smoke_data(
    *,
    train_source: Path,
    validation_source: Path,
    output_dir: Path,
    smoke_config: Mapping[str, Any],
) -> dict[str, Path]:
    """Materialize an immutable 32/8 subset and its auditable manifest."""
    validate_smoke_config(smoke_config)
    paths = {
        "train": output_dir / "train_32.parquet",
        "validation": output_dir / "validation_8.parquet",
        "manifest": output_dir / "subset_manifest.json",
    }
    seed = int(smoke_config["seed"])
    train = select_smoke_rows(
        read_rows(train_source),
        split="opd_train",
        count=int(smoke_config["data"]["train_count"]),
        seed=seed,
    )
    validation = select_smoke_rows(
        read_rows(validation_source),
        split="validation",
        count=int(smoke_config["data"]["validation_count"]),
        seed=seed,
    )
    payload = {
        "seed": seed,
        "train_source": str(train_source.resolve()),
        "validation_source": str(validation_source.resolve()),
        "train_ids": [row["extra_info"]["sample_id"] for row in train],
        "validation_ids": [row["extra_info"]["sample_id"] for row in validation],
        "train_question_hashes": [row["extra_info"]["question_hash"] for row in train],
        "validation_question_hashes": [
            row["extra_info"]["question_hash"] for row in validation
        ],
    }
    if set(payload["train_ids"]) & set(payload["validation_ids"]):
        raise IntegrationError("smoke train/validation sample IDs overlap")
    if set(payload["train_question_hashes"]) & set(
        payload["validation_question_hashes"]
    ):
        raise IntegrationError("smoke train/validation questions overlap")
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        if len(existing) != len(paths):
            raise IntegrationError("partial smoke data already exists")
        saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if saved != payload:
            raise IntegrationError("existing smoke subset manifest does not match")
        actual_train = read_rows(paths["train"])
        actual_validation = read_rows(paths["validation"])
        if [row["extra_info"]["sample_id"] for row in actual_train] != payload[
            "train_ids"
        ]:
            raise IntegrationError("existing smoke train parquet does not match")
        if [row["extra_info"]["sample_id"] for row in actual_validation] != payload[
            "validation_ids"
        ]:
            raise IntegrationError("existing smoke validation parquet does not match")
        return paths
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_parquet(temporary_dir / paths["train"].name, train)
        _write_parquet(temporary_dir / paths["validation"].name, validation)
        write_json(temporary_dir / paths["manifest"].name, payload)
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return paths


def build_e2e_commands(
    *,
    opd_config: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
    student_checkpoint: Path,
    teacher_model: Path,
    train_data: Path,
    validation_data: Path,
    run_dir: Path,
) -> tuple[list[str], list[str]]:
    """Build phase-1 and explicit-resume phase-2 verl commands."""
    validate_recipe(opd_config)
    validate_smoke_config(smoke_config)
    base = build_verl_command(
        opd_config,
        mode="smoke",
        student_checkpoint=student_checkpoint,
        teacher_model=teacher_model,
        train_data=train_data,
        validation_data=validation_data,
        run_dir=run_dir,
    )
    training = smoke_config["training"]
    common = {
        "actor_rollout_ref.rollout.n": smoke_config["rollout"]["n"],
        "data.max_response_length": smoke_config["rollout"]["max_response_length"],
        "actor_rollout_ref.rollout.max_model_len": (
            int(opd_config["rollout"]["max_prompt_length"])
            + int(smoke_config["rollout"]["max_response_length"])
            + 1
        ),
        "data.train_batch_size": training["train_batch_size"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": training["train_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": training[
            "ppo_micro_batch_size_per_gpu"
        ],
        "distillation.teacher_models.teacher_model.inference.max_model_len": (
            int(opd_config["rollout"]["max_prompt_length"])
            + int(smoke_config["rollout"]["max_response_length"])
            + 1
        ),
        "trainer.default_local_dir": run_dir / "checkpoints",
        "trainer.rollout_data_dir": run_dir / "rollouts",
        "trainer.validation_data_dir": run_dir / "validation",
        "trainer.log_val_generations": smoke_config["data"]["validation_count"],
        "trainer.save_freq": training["save_freq"],
        "trainer.project_name": "miniopd",
        "trainer.experiment_name": "opd-e2e-smoke",
    }
    phase1 = replace_hydra_overrides(
        base,
        {
            **common,
            "trainer.total_training_steps": training["phase1_steps"],
            "trainer.resume_mode": "disable",
            "trainer.val_before_train": False,
            "trainer.test_freq": -1,
        },
    )
    checkpoint = run_dir / "checkpoints" / f"global_step_{training['phase1_steps']}"
    phase2 = replace_hydra_overrides(
        base,
        {
            **common,
            "trainer.total_training_steps": training["final_steps"],
            "trainer.resume_mode": "resume_path",
            "trainer.resume_from_path": checkpoint,
            "trainer.val_before_train": False,
            "trainer.test_freq": training["validation_frequency"],
        },
    )
    return phase1, phase2


_NON_FINITE = re.compile(r"(?<![\w.])(?:nan|[+-]?inf)(?![\w.])", re.IGNORECASE)


def _json_is_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_json_is_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_is_finite(item) for item in value)
    return True


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    return read_rows(path) if path.is_file() else []


def record_tensor_audit(
    path: Path,
    *,
    step: int,
    input_ids: torch.Tensor,
    shifted_logits: torch.Tensor,
    target_ids: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    current_student_log_probs: torch.Tensor,
) -> dict[str, Any]:
    """Persist auditable shape, mask, stop-gradient, reward, and finite evidence."""
    expected = input_ids.shape[1] - 1
    if shifted_logits.shape[:2] != (input_ids.shape[0], expected):
        raise IntegrationError("shifted logits are not next-token aligned")
    token_shape = (input_ids.shape[0], expected)
    tensors = {
        "target_ids": target_ids,
        "response_mask": response_mask,
        "teacher_log_probs": teacher_log_probs,
        "old_student_log_probs": old_student_log_probs,
        "advantage": advantage,
        "current_student_log_probs": current_student_log_probs,
    }
    wrong = {
        name: list(value.shape)
        for name, value in tensors.items()
        if value.shape != token_shape
    }
    if wrong:
        raise IntegrationError(f"next-token tensor shapes are misaligned: {wrong}")
    selected = advantage.detach().float().masked_select(response_mask.bool())
    if selected.numel() == 0:
        raise IntegrationError("response mask selects no reward tokens")
    finite_tensors = (
        shifted_logits,
        teacher_log_probs,
        old_student_log_probs,
        advantage,
        current_student_log_probs,
    )
    record = {
        "step": step,
        "input_ids_shape": list(input_ids.shape),
        "shifted_logits_shape": list(shifted_logits.shape),
        "target_ids_shape": list(target_ids.shape),
        "response_mask_shape": list(response_mask.shape),
        "response_token_count": int(response_mask.bool().sum().item()),
        "sampled_token_reward_min": float(selected.min().item()),
        "sampled_token_reward_max": float(selected.max().item()),
        "teacher_requires_grad": teacher_log_probs.requires_grad,
        "advantage_requires_grad": advantage.requires_grad,
        "all_finite": all(
            torch.isfinite(value).all().item() for value in finite_tensors
        ),
    }
    append_jsonl(path, [record])
    return record


def validate_tensor_audit(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Validate saved shift/mask/reward/no-grad evidence from real smoke batches."""
    if not rows:
        return False
    required = {
        "step",
        "input_ids_shape",
        "shifted_logits_shape",
        "target_ids_shape",
        "response_mask_shape",
        "response_token_count",
        "teacher_requires_grad",
        "advantage_requires_grad",
        "all_finite",
        "sampled_token_reward_min",
        "sampled_token_reward_max",
    }
    for row in rows:
        if required.difference(row):
            return False
        input_shape = row["input_ids_shape"]
        logits_shape = row["shifted_logits_shape"]
        target_shape = row["target_ids_shape"]
        mask_shape = row["response_mask_shape"]
        if not all(
            isinstance(shape, list)
            for shape in (input_shape, logits_shape, target_shape, mask_shape)
        ):
            return False
        if len(input_shape) != 2 or len(logits_shape) != 3:
            return False
        expected_tokens = [input_shape[0], input_shape[1] - 1]
        if logits_shape[:2] != expected_tokens:
            return False
        if target_shape != expected_tokens or mask_shape != expected_tokens:
            return False
        if int(row["response_token_count"]) <= 0:
            return False
        if row["teacher_requires_grad"] or row["advantage_requires_grad"]:
            return False
        if row["all_finite"] is not True:
            return False
        if not math.isfinite(float(row["sampled_token_reward_min"])):
            return False
        if not math.isfinite(float(row["sampled_token_reward_max"])):
            return False
    return True


def check_e2e_artifacts(
    run_dir: Path,
    smoke_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the complete train-save-restart-resume-validate chain."""
    validate_smoke_config(smoke_config)
    training = smoke_config["training"]
    phase1_log = run_dir / "phase1" / "run.log"
    phase2_log = run_dir / "phase2" / "run.log"
    text1 = phase1_log.read_text(encoding="utf-8") if phase1_log.is_file() else ""
    text2 = phase2_log.read_text(encoding="utf-8") if phase2_log.is_file() else ""
    combined = text1 + "\n" + text2
    checkpoint2 = run_dir / "checkpoints" / f"global_step_{training['phase1_steps']}"
    checkpoint5 = run_dir / "checkpoints" / f"global_step_{training['final_steps']}"
    metrics = _read_jsonl_if_present(run_dir / "metrics.jsonl")
    diagnostics = _read_jsonl_if_present(run_dir / "diagnostics.jsonl")
    tensor_audit = _read_jsonl_if_present(run_dir / "tensor_audit.jsonl")
    sync1 = _read_jsonl_if_present(run_dir / "phase1" / "sync_versions.jsonl")
    sync2 = _read_jsonl_if_present(run_dir / "phase2" / "sync_versions.jsonl")
    resume_pattern = re.compile(
        rf"Resuming from .*global_step_{training['phase1_steps']}.*"
        rf"setting global step to {training['phase1_steps']}",
        re.IGNORECASE | re.DOTALL,
    )
    checks = {
        "subset_manifest": (run_dir / "inputs" / "subset_manifest.json").is_file(),
        "phase1_log": phase1_log.is_file(),
        "phase2_log": phase2_log.is_file(),
        "phase1_checkpoint": checkpoint2.is_dir() and any(checkpoint2.rglob("*")),
        "final_checkpoint": checkpoint5.is_dir() and any(checkpoint5.rglob("*")),
        "explicit_resume": bool(resume_pattern.search(text2)),
        "rollout": "rollout" in combined.lower(),
        "teacher_scoring": "distillation/" in combined.lower(),
        "backward_optimizer": "grad_norm" in combined.lower(),
        "weight_sync": len(sync1) >= 3 and len(sync2) >= 4,
        "zero_recorded_policy_lag": bool(sync1 + sync2)
        and all(row.get("policy_lag_after") == 0 for row in sync1 + sync2),
        "rollout_samples": (run_dir / "rollouts").is_dir()
        and any((run_dir / "rollouts").rglob("*")),
        "validation_samples": (run_dir / "validation").is_dir()
        and any((run_dir / "validation").rglob("*")),
        "validation_metrics": "val/" in text2.lower() or "validation" in text2.lower(),
        "step_metrics": bool(metrics),
        "periodic_diagnostics": bool(diagnostics),
        "mask_shift_reward_audit": validate_tensor_audit(tensor_audit),
        "finite_logs": not bool(_NON_FINITE.search(combined)),
        "finite_metrics": _json_is_finite(metrics) and _json_is_finite(diagnostics),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "phase1_sync_events": len(sync1),
        "phase2_sync_events": len(sync2),
        "metric_records": len(metrics),
        "diagnostic_records": len(diagnostics),
        "tensor_audit_records": len(tensor_audit),
        "phase1_steps": training["phase1_steps"],
        "final_steps": training["final_steps"],
    }
    write_json(run_dir / "e2e_artifact_report.json", report)
    return report


def run_e2e_smoke(
    *,
    opd_config_path: Path,
    smoke_config_path: Path,
    verl_lock_path: Path,
    verl_root: Path,
    student_checkpoint: Path,
    teacher_model: Path,
    train_source: Path,
    validation_source: Path,
    manifest: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Run both fresh and resumed processes, then return the audit report."""
    opd_config = load_yaml(opd_config_path)
    smoke_config = load_yaml(smoke_config_path)
    validate_recipe(opd_config)
    validate_smoke_config(smoke_config)
    lock = load_yaml(verl_lock_path)
    if (run_dir / "phase1" / "run.log").exists() or (
        run_dir / "phase2" / "run.log"
    ).exists():
        raise IntegrationError(
            "smoke run logs already exist; choose a new run directory"
        )
    preflight = {
        "verl_commit": validate_verl_checkout(verl_root, lock, "smoke"),
        "student_checkpoint": validate_checkpoint(
            student_checkpoint, role="student checkpoint"
        ),
        "teacher_model": validate_checkpoint(teacher_model, role="teacher model"),
        "full_data": validate_verl_data(
            train_source,
            validation_source,
            manifest,
            train_count=int(opd_config["data"]["train_count"]),
            validation_count=int(opd_config["data"]["validation_count"]),
        ),
        "runtime": validate_runtime(
            student_checkpoint,
            teacher_model,
            int(opd_config["resources"]["expected_total_gpus"]),
        ),
    }
    paths = prepare_smoke_data(
        train_source=train_source,
        validation_source=validation_source,
        output_dir=run_dir / "inputs",
        smoke_config=smoke_config,
    )
    phase1, phase2 = build_e2e_commands(
        opd_config=opd_config,
        smoke_config=smoke_config,
        student_checkpoint=student_checkpoint,
        teacher_model=teacher_model,
        train_data=paths["train"],
        validation_data=paths["validation"],
        run_dir=run_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "preflight.json", preflight)
    write_resolved_yaml(run_dir / "resolved_opd_config.yaml", opd_config)
    write_resolved_yaml(run_dir / "resolved_smoke_config.yaml", smoke_config)
    write_json(run_dir / "commands.json", {"phase1": phase1, "phase2": phase2})

    if launch(phase1, run_dir / "phase1", cwd=verl_root) != 0:
        raise IntegrationError("phase 1 OPD smoke process failed")
    checkpoint = (
        run_dir
        / "checkpoints"
        / (f"global_step_{smoke_config['training']['phase1_steps']}")
    )
    if not checkpoint.is_dir() or not any(checkpoint.rglob("*")):
        raise IntegrationError(f"phase 1 checkpoint is missing: {checkpoint}")
    if (
        launch(
            phase2,
            run_dir / "phase2",
            cwd=verl_root,
            sync_version_offset=int(smoke_config["training"]["phase1_steps"]),
        )
        != 0
    ):
        raise IntegrationError("phase 2 resumed OPD smoke process failed")
    report = check_e2e_artifacts(run_dir, smoke_config)
    if not report["passed"]:
        raise IntegrationError("end-to-end smoke process finished but audit failed")
    return report
