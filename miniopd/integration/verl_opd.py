"""Strict, thin integration layer around upstream verl OPD."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniopd.data.gsm8k import SYSTEM_PROMPT, question_hash
from miniopd.diagnostics.opd import DiagnosticsConfig
from miniopd.evaluation.io import append_jsonl, write_json


class IntegrationError(ValueError):
    """Raised when a Stage 12 contract is not satisfied."""


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise IntegrationError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise IntegrationError(f"missing config key: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_recipe(config: Mapping[str, Any]) -> None:
    """Reject accidental deviations from the agreed sampled-token experiment."""
    expected = {
        ("method",): "sampled-token",
        ("rollout", "n"): 2,
        ("rollout", "temperature"): 1.0,
        ("rollout", "top_p"): 1.0,
        ("rollout", "max_prompt_length"): 256,
        ("rollout", "max_response_length"): 512,
        ("teacher", "temperature"): 1.0,
        ("distillation", "loss_mode"): "k1",
        ("distillation", "use_policy_gradient"): True,
        ("distillation", "use_task_rewards"): False,
        ("distillation", "policy_loss_mode"): "vanilla",
        ("actor", "learning_rate"): 1.0e-6,
        ("actor", "grad_clip"): 1.0,
        ("actor", "epochs"): 1,
        ("resources", "expected_total_gpus"): 2,
    }
    mismatches = []
    for keys, wanted in expected.items():
        actual = _nested(config, *keys)
        if actual != wanted:
            mismatches.append(f"{'.'.join(keys)}={actual!r}, expected {wanted!r}")
    if mismatches:
        raise IntegrationError("invalid OPD recipe: " + "; ".join(mismatches))
    diagnostics = _nested(config, "diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise IntegrationError("diagnostics config must be a mapping")
    DiagnosticsConfig.from_mapping(diagnostics)
    formal = _nested(config, "formal")
    if not isinstance(formal, Mapping):
        raise IntegrationError("formal config must be a mapping")
    if formal.get("resume_mode") != "auto":
        raise IntegrationError("formal training must use resume_mode=auto")
    if int(formal.get("checkpoint_interval_steps", 0)) <= 0:
        raise IntegrationError("formal checkpoint interval must be positive")
    if formal.get("validation_interval_steps") != formal.get(
        "checkpoint_interval_steps"
    ):
        raise IntegrationError(
            "formal validation and checkpoint intervals must be identical"
        )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise IntegrationError(f"non-object at {path}:{line_number}")
                rows.append(value)
        return rows
    if path.suffix == ".parquet":
        try:
            import datasets
        except ImportError as error:
            raise RuntimeError(
                "install MiniOPD's data extra to read parquet"
            ) from error
        return [dict(row) for row in datasets.Dataset.from_parquet(str(path))]
    raise IntegrationError(f"unsupported data file: {path}")


def _manifest_by_split(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_rows(path):
        split = str(row["split"])
        grouped.setdefault(split, {})[str(row["sample_id"])] = row
    return grouped


def validate_verl_data(
    train_path: Path,
    validation_path: Path,
    manifest_path: Path,
    *,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    """Validate schema, prompt identity, counts, and split non-leakage."""
    manifest = _manifest_by_split(manifest_path)
    reports: dict[str, Any] = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    for split, path, expected_count in (
        ("opd_train", train_path, train_count),
        ("validation", validation_path, validation_count),
    ):
        rows = read_rows(path)
        if len(rows) != expected_count:
            raise IntegrationError(
                f"{split} has {len(rows)} rows; expected {expected_count}"
            )
        expected = manifest.get(split, {})
        observed_ids: set[str] = set()
        for index, row in enumerate(rows):
            missing = {
                "data_source",
                "prompt",
                "ability",
                "reward_model",
                "extra_info",
            }.difference(row)
            if missing:
                raise IntegrationError(f"{split}[{index}] missing {sorted(missing)}")
            extra = row["extra_info"]
            if not isinstance(extra, Mapping) or extra.get("split") != split:
                raise IntegrationError(f"{split}[{index}] has wrong extra_info.split")
            sample_id = str(extra.get("sample_id", ""))
            digest = str(extra.get("question_hash", ""))
            question = str(extra.get("question", ""))
            if sample_id not in expected:
                raise IntegrationError(
                    f"{sample_id!r} is not in manifest split {split}"
                )
            if expected[sample_id]["question_hash"] != digest:
                raise IntegrationError(f"question hash mismatch for {sample_id}")
            if question_hash(question) != digest:
                raise IntegrationError(f"question text mismatch for {sample_id}")
            prompt = row["prompt"]
            if not isinstance(prompt, list) or len(prompt) != 2:
                raise IntegrationError(f"invalid prompt for {sample_id}")
            if prompt[0] != {"role": "system", "content": SYSTEM_PROMPT}:
                raise IntegrationError(f"system prompt mismatch for {sample_id}")
            if prompt[1] != {"role": "user", "content": question}:
                raise IntegrationError(f"user prompt mismatch for {sample_id}")
            if sample_id in observed_ids or sample_id in all_ids:
                raise IntegrationError(f"duplicate or leaked sample_id {sample_id}")
            if digest in all_hashes:
                raise IntegrationError(f"question leakage for {sample_id}")
            observed_ids.add(sample_id)
            all_ids.add(sample_id)
            all_hashes.add(digest)
        if observed_ids != set(expected):
            raise IntegrationError(f"{split} is not an exact manifest match")
        reports[split] = {"rows": len(rows), "unique_questions": len(observed_ids)}
    forbidden = set(manifest.get("sft", {})) | set(manifest.get("test", {}))
    if all_ids & forbidden:
        raise IntegrationError("SFT/test sample leaked into OPD inputs")
    return reports


def validate_checkpoint(path: Path, *, role: str = "model") -> dict[str, Any]:
    if not path.is_dir():
        raise IntegrationError(f"{role} directory does not exist: {path}")
    required = [path / "config.json", path / "tokenizer_config.json"]
    missing = [str(item) for item in required if not item.is_file()]
    tokenizer = list(path.glob("tokenizer.json")) + list(path.glob("tokenizer.model"))
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if missing or not tokenizer or not weights:
        raise IntegrationError(
            f"incomplete {role}: missing={missing}, "
            f"tokenizer={bool(tokenizer)}, weights={bool(weights)}"
        )
    return {"path": str(path.resolve()), "weight_files": len(weights)}


def validate_verl_checkout(root: Path, lock: Mapping[str, Any], mode: str) -> str:
    candidate = str(lock.get("candidate_commit", ""))
    if mode == "full":
        if lock.get("status") != "verified" or not lock.get("verified_commit"):
            raise IntegrationError(
                "full training requires a GPU-smoke-verified verl commit"
            )
        required = str(lock["verified_commit"])
    else:
        required = candidate
    command = ["git", "-C", str(root), "rev-parse", "HEAD"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    actual = result.stdout.strip()
    if actual != required:
        raise IntegrationError(f"verl HEAD is {actual}; required {required}")
    remote = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if "verl-project/verl" not in remote:
        raise IntegrationError(f"unexpected verl origin: {remote}")
    return actual


def validate_runtime(
    student: Path, teacher: Path, expected_gpus: int
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise IntegrationError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise IntegrationError(
            f"visible GPU count is {count}; expected {expected_gpus}"
        )
    if not torch.cuda.is_bf16_supported():
        raise IntegrationError("visible GPU does not support BF16")
    student_tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise IntegrationError("student and teacher vocabularies differ")
    token_ids = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    for field in token_ids:
        if getattr(student_tokenizer, field) != getattr(teacher_tokenizer, field):
            raise IntegrationError(f"student and teacher {field} differ")
    return {
        "cuda_device_count": count,
        "bf16_supported": True,
        "gpus": [torch.cuda.get_device_name(index) for index in range(count)],
        "tokenizer_vocab_size": len(student_tokenizer),
    }


def _hydra(key: str, value: Any) -> str:
    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = str(value)
    elif isinstance(value, Path):
        rendered = str(value.resolve())
    elif isinstance(value, str) and ("/" in value or " " in value):
        rendered = json.dumps(value)
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def build_verl_command(
    config: Mapping[str, Any],
    *,
    mode: str,
    student_checkpoint: Path,
    teacher_model: Path,
    train_data: Path,
    validation_data: Path,
    run_dir: Path,
) -> list[str]:
    """Translate MiniOPD's reviewed recipe to official verl Hydra overrides."""
    if mode not in {"smoke", "full"}:
        raise IntegrationError("mode must be smoke or full")
    validate_recipe(config)
    rollout = _nested(config, "rollout")
    distill = _nested(config, "distillation")
    actor = _nested(config, "actor")
    resources = _nested(config, "resources")
    smoke = _nested(config, "smoke")
    formal = _nested(config, "formal")
    batch_size = (
        smoke["train_batch_size"] if mode == "smoke" else actor["full_train_batch_size"]
    )
    mini_batch = int(batch_size) * int(rollout["n"])
    values: list[tuple[str, Any]] = [
        ("algorithm.adv_estimator", "grpo"),
        ("algorithm.use_kl_in_reward", False),
        ("data.train_files", train_data),
        ("data.val_files", validation_data),
        ("data.train_batch_size", batch_size),
        ("data.max_prompt_length", rollout["max_prompt_length"]),
        ("data.max_response_length", rollout["max_response_length"]),
        ("data.filter_overlong_prompts", True),
        ("data.truncation", "error"),
        ("actor_rollout_ref.model.path", student_checkpoint),
        ("actor_rollout_ref.model.use_remove_padding", True),
        ("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        ("actor_rollout_ref.actor.optim.lr", actor["learning_rate"]),
        ("actor_rollout_ref.actor.grad_clip", actor["grad_clip"]),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", mini_batch),
        (
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            actor["ppo_micro_batch_size_per_gpu"],
        ),
        ("actor_rollout_ref.actor.strategy", actor["strategy"]),
        ("actor_rollout_ref.actor.use_kl_loss", False),
        ("actor_rollout_ref.rollout.name", "vllm"),
        ("actor_rollout_ref.rollout.n", rollout["n"]),
        ("actor_rollout_ref.rollout.temperature", rollout["temperature"]),
        ("actor_rollout_ref.rollout.top_p", rollout["top_p"]),
        (
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            resources["rollout_tensor_parallel_size"],
        ),
        (
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            rollout["gpu_memory_utilization"],
        ),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("trainer.n_gpus_per_node", resources["trainer_gpus_per_node"]),
        ("trainer.nnodes", resources["nnodes"]),
        ("trainer.total_epochs", actor["epochs"]),
        ("trainer.default_local_dir", run_dir / "checkpoints"),
        ("trainer.rollout_data_dir", run_dir / "rollouts"),
        ("trainer.logger", "[console]"),
        ("trainer.val_before_train", False),
        (
            "trainer.test_freq",
            -1 if mode == "smoke" else formal["validation_interval_steps"],
        ),
        (
            "trainer.save_freq",
            smoke["save_freq"]
            if mode == "smoke"
            else formal["checkpoint_interval_steps"],
        ),
        (
            "trainer.validation_data_dir",
            run_dir / "validation" if mode == "full" else None,
        ),
        (
            "trainer.log_val_generations",
            0 if mode == "smoke" else formal["validation_generations_to_log"],
        ),
        (
            "trainer.resume_mode",
            "disable" if mode == "smoke" else formal["resume_mode"],
        ),
        ("distillation.enabled", True),
        ("distillation.n_gpus_per_node", resources["teacher_gpus_per_node"]),
        ("distillation.nnodes", resources["nnodes"]),
        ("distillation.teacher_models.teacher_model.model_path", teacher_model),
        ("distillation.teacher_models.teacher_model.inference.name", "vllm"),
        (
            "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size",
            resources["teacher_tensor_parallel_size"],
        ),
        (
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization",
            resources["teacher_gpu_memory_utilization"],
        ),
        (
            "distillation.teacher_models.teacher_model.inference.temperature",
            _nested(config, "teacher", "temperature"),
        ),
        (
            "distillation.teacher_models.teacher_model.inference.max_model_len",
            rollout["max_prompt_length"] + rollout["max_response_length"] + 1,
        ),
        ("distillation.distillation_loss.loss_mode", distill["loss_mode"]),
        (
            "distillation.distillation_loss.use_task_rewards",
            distill["use_task_rewards"],
        ),
        (
            "distillation.distillation_loss.use_policy_gradient",
            distill["use_policy_gradient"],
        ),
        (
            "distillation.distillation_loss.policy_loss_mode",
            distill["policy_loss_mode"],
        ),
        ("distillation.distillation_loss.loss_max_clamp", distill["loss_max_clamp"]),
        (
            "distillation.distillation_loss.log_prob_min_clamp",
            distill["log_prob_min_clamp"],
        ),
    ]
    if mode == "smoke":
        values.append(("trainer.total_training_steps", smoke["total_training_steps"]))
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        *[_hydra(*item) for item in values],
    ]


def replace_hydra_overrides(
    command: Sequence[str], overrides: Mapping[str, Any]
) -> list[str]:
    """Replace selected Hydra values without allowing duplicate keys."""
    if len(command) < 3 or command[1:3] != ["-m", "verl.trainer.main_ppo"]:
        raise IntegrationError("not a verl main_ppo command")
    replacements = dict(overrides)
    output = list(command[:3])
    seen: set[str] = set()
    for argument in command[3:]:
        key = argument.split("=", 1)[0]
        if key in seen:
            raise IntegrationError(f"duplicate Hydra key in command: {key}")
        seen.add(key)
        if key in replacements:
            output.append(_hydra(key, replacements.pop(key)))
        else:
            output.append(argument)
    output.extend(_hydra(key, value) for key, value in replacements.items())
    return output


@dataclass
class SyncAuditRecorder:
    """Record only weight-sync events confirmed by upstream verl log lines."""

    path: Path
    next_version: int = 0

    _PATTERN = re.compile(r"update_weights done", re.IGNORECASE)

    def observe(self, line: str) -> bool:
        if not self._PATTERN.search(line):
            return False
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor_version": self.next_version,
            "rollout_version_before": max(self.next_version - 1, -1),
            "rollout_version_after": self.next_version,
            "policy_lag_after": 0,
            "evidence": line.strip()[-500:],
        }
        append_jsonl(self.path, [record])
        self.next_version += 1
        return True


def launch(
    command: Sequence[str],
    run_dir: Path,
    *,
    cwd: Path,
    sync_version_offset: int = 0,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    recorder = SyncAuditRecorder(
        run_dir / "sync_versions.jsonl", next_version=sync_version_offset
    )
    environment = dict(os.environ)
    environment["VERL_LOGGING_LEVEL"] = "INFO"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            recorder.observe(line)
        return process.wait()


def check_artifacts(run_dir: Path, mode: str) -> dict[str, Any]:
    required = [
        "preflight.json",
        "resolved_config.yaml",
        "launch_command.json",
        "run.log",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    log = (run_dir / "run.log").read_text(encoding="utf-8") if not missing else ""
    evidence = {
        "generation": "gen" in log.lower() or "rollout" in log.lower(),
        "teacher_scoring": "distillation/" in log.lower(),
        "backward_update": "grad_norm" in log.lower(),
        "weight_sync_timing": "timing_s/update_weights" in log,
    }
    sync_path = run_dir / "sync_versions.jsonl"
    sync = read_rows(sync_path) if sync_path.is_file() else []
    minimum_syncs = 2 if mode == "smoke" else 1
    sync_ok = len(sync) >= minimum_syncs and all(
        row.get("policy_lag_after") == 0 for row in sync
    )
    checkpoint_ok = (run_dir / "checkpoints").is_dir()
    rollout_ok = (run_dir / "rollouts").is_dir() and any(
        (run_dir / "rollouts").iterdir()
    )
    passed = (
        not missing
        and all(evidence.values())
        and sync_ok
        and checkpoint_ok
        and rollout_ok
    )
    report = {
        "passed": passed,
        "mode": mode,
        "missing_files": missing,
        "evidence": evidence,
        "sync_events": len(sync),
        "sync_ok": sync_ok,
        "checkpoint_ok": checkpoint_ok,
        "rollout_dump_ok": rollout_ok,
    }
    write_json(run_dir / "artifact_report.json", report)
    return report


def write_resolved_yaml(path: Path, config: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
