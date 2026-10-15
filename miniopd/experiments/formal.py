"""Artifact-only reporting for the formal sampled-token OPD experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniopd.evaluation.answer_parser import ErrorType, evaluate_answer
from miniopd.evaluation.io import append_jsonl, read_jsonl, write_json
from miniopd.evaluation.pipeline import summarize_records
from miniopd.integration.verl_opd import load_yaml

FORMAL_ROLES = ("base", "sft", "opd", "teacher")


class ExperimentError(ValueError):
    """Raised when formal experiment artifacts violate the protocol."""


def validate_formal_config(config: Mapping[str, Any]) -> None:
    if config.get("training", {}).get("expected_train_rows") != 3473:
        raise ExperimentError("formal OPD must use all 3473 training prompts")
    if config.get("training", {}).get("epochs") != 1:
        raise ExperimentError("formal OPD must run exactly one epoch")
    selection = config.get("checkpoint_selection", {})
    if selection.get("split") != "validation":
        raise ExperimentError("checkpoint selection must use validation only")
    if selection.get("expected_validation_rows") != 1000:
        raise ExperimentError("checkpoint selection requires 1000 validation rows")
    if selection.get("primary_metric") != "exact_match":
        raise ExperimentError("checkpoint selection metric must be exact_match")
    if selection.get("mode") != "max":
        raise ExperimentError("exact_match checkpoint selection must maximize")
    evaluation = config.get("final_evaluation", {})
    if evaluation.get("expected_rows") != 1319:
        raise ExperimentError("final evaluation must use all 1319 test rows")
    if tuple(evaluation.get("roles", ())) != FORMAL_ROLES:
        raise ExperimentError(f"final roles must be {FORMAL_ROLES}")
    if evaluation.get("temperature") != 0.0 or evaluation.get("do_sample") is not False:
        raise ExperimentError("final evaluation must use greedy decoding")
    if evaluation.get("max_new_tokens") != 512:
        raise ExperimentError("final evaluation max_new_tokens must be 512")
    if evaluation.get("dtype") != "bfloat16":
        raise ExperimentError("final evaluation must use BF16")
    if evaluation.get("seed") != 42:
        raise ExperimentError("final evaluation seed must be 42")


def _formal_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config["final_evaluation"]
    return {
        "backend": "vllm",
        "dtype": evaluation["dtype"],
        "temperature": evaluation["temperature"],
        "top_p": 1.0,
        "do_sample": evaluation["do_sample"],
        "max_new_tokens": evaluation["max_new_tokens"],
        "batch_size": 8,
        "seed": evaluation["seed"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_directory(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ExperimentError(f"checkpoint does not exist: {path}")
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    if not files:
        raise ExperimentError(f"checkpoint contains no files: {path}")
    entries = [
        {
            "path": str(item.relative_to(path)),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in files
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path.resolve()),
        "files": entries,
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_validation_metrics(
    *,
    validation_dir: Path,
    checkpoint_root: Path,
    output_path: Path,
    formal_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rescore verl validation dumps and bind every metric row to a checkpoint."""
    validate_formal_config(formal_config)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not validation_dir.is_dir():
        raise ExperimentError(f"validation dump directory is missing: {validation_dir}")
    expected_rows = int(
        formal_config["checkpoint_selection"]["expected_validation_rows"]
    )
    dumps: list[tuple[int, Path]] = []
    for path in validation_dir.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError as error:
            raise ExperimentError(
                f"validation dump filename must be a numeric step: {path.name}"
            ) from error
        dumps.append((step, path))
    if not dumps:
        raise ExperimentError("no verl validation JSONL dumps were found")

    metrics: list[dict[str, Any]] = []
    for step, path in sorted(dumps):
        rows = read_jsonl(path)
        if len(rows) != expected_rows:
            raise ExperimentError(
                f"{path.name} has {len(rows)} rows; expected {expected_rows}"
            )
        required = {"output", "gts", "score", "step", "truncated"}
        for index, row in enumerate(rows):
            missing = required.difference(row)
            if missing:
                raise ExperimentError(
                    f"{path.name}:{index + 1} missing {sorted(missing)}; "
                    "the validation adapter must persist an explicit truncated flag"
                )
            if int(row["step"]) != step:
                raise ExperimentError(f"{path.name} contains a mismatched step value")
            if not isinstance(row["truncated"], bool):
                raise ExperimentError(f"{path.name} has a non-boolean truncated value")

        rescored = [
            evaluate_answer(
                str(row["output"]),
                str(row["gts"]),
                truncated=row["truncated"],
            )
            for row in rows
        ]
        rewards = [float(row["score"]) for row in rows]
        if any(not math.isfinite(value) for value in rewards):
            raise ExperimentError(f"{path.name} contains a non-finite validation score")
        checkpoint = checkpoint_root / f"global_step_{step}"
        if not checkpoint.is_dir():
            raise ExperimentError(
                f"validation step {step} has no matching checkpoint: {checkpoint}"
            )
        correct = sum(item.correct for item in rescored)
        format_errors = sum(
            item.error_type == ErrorType.FORMAT_ERROR for item in rescored
        )
        truncated = sum(bool(row["truncated"]) for row in rows)
        metrics.append(
            {
                "step": step,
                "split": "validation",
                "checkpoint": str(checkpoint.resolve()),
                "correct": correct,
                "total": expected_rows,
                "exact_match": correct / expected_rows,
                "format_error": format_errors,
                "format_error_ratio": format_errors / expected_rows,
                "truncated": truncated,
                "truncated_ratio": truncated / expected_rows,
                "mean_framework_score": sum(rewards) / expected_rows,
                "source_dump": str(path.resolve()),
                "source_sha256": _sha256(path),
                "scorer": "miniopd.evaluation.answer_parser.evaluate_answer",
            }
        )
    append_jsonl(output_path, metrics)
    return metrics


def _validate_validation_row(
    row: Mapping[str, Any], checkpoint_root: Path, expected_rows: int
) -> dict[str, Any]:
    required = {
        "step",
        "split",
        "checkpoint",
        "exact_match",
        "format_error_ratio",
        "truncated_ratio",
        "total",
    }
    missing = required.difference(row)
    if missing:
        raise ExperimentError(f"validation record missing {sorted(missing)}")
    if row["split"] != "validation":
        raise ExperimentError("test or training metrics cannot select a checkpoint")
    if int(row["total"]) != expected_rows:
        raise ExperimentError(f"validation record must contain {expected_rows} rows")
    checkpoint = Path(str(row["checkpoint"])).resolve()
    root = checkpoint_root.resolve()
    if not checkpoint.is_relative_to(root):
        raise ExperimentError("candidate checkpoint is outside checkpoint root")
    values = {
        "step": int(row["step"]),
        "checkpoint": str(checkpoint),
        "exact_match": float(row["exact_match"]),
        "format_error_ratio": float(row["format_error_ratio"]),
        "truncated_ratio": float(row["truncated_ratio"]),
        "total": int(row["total"]),
        "split": "validation",
    }
    if values["step"] < 1 or not all(
        math.isfinite(values[key])
        for key in ("exact_match", "format_error_ratio", "truncated_ratio")
    ):
        raise ExperimentError("invalid validation metric value")
    if not all(
        0 <= values[key] <= 1
        for key in ("exact_match", "format_error_ratio", "truncated_ratio")
    ):
        raise ExperimentError("validation rates must be in [0, 1]")
    if not checkpoint.is_dir():
        raise ExperimentError(f"candidate checkpoint is missing: {checkpoint}")
    return values


def select_validation_checkpoint(
    *,
    metrics_path: Path,
    checkpoint_root: Path,
    formal_config: Mapping[str, Any],
    output_path: Path,
    forbidden_test_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Select without reading test scores and save the immutable decision."""
    validate_formal_config(formal_config)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    leaked = [
        str(path)
        for root in forbidden_test_roots
        if root.exists()
        for name in ("predictions.jsonl", "summary.json")
        for path in root.rglob(name)
    ]
    if leaked:
        raise ExperimentError(
            "test predictions already exist before checkpoint selection: "
            + ", ".join(leaked)
        )
    rows = read_jsonl(metrics_path)
    if not rows:
        raise ExperimentError("validation metrics are empty")
    expected_rows = int(
        formal_config["checkpoint_selection"]["expected_validation_rows"]
    )
    candidates = [
        _validate_validation_row(row, checkpoint_root, expected_rows) for row in rows
    ]
    steps = [item["step"] for item in candidates]
    if len(steps) != len(set(steps)):
        raise ExperimentError("validation metrics contain duplicate steps")
    selected = min(
        candidates,
        key=lambda item: (
            -item["exact_match"],
            item["format_error_ratio"],
            item["truncated_ratio"],
            item["step"],
        ),
    )
    report = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_data_split": "validation",
        "test_metrics_read": False,
        "rule": formal_config["checkpoint_selection"],
        "selected": selected,
        "selected_checkpoint": fingerprint_directory(Path(selected["checkpoint"])),
        "candidates": candidates,
        "metrics_source": str(metrics_path.resolve()),
    }
    write_json(output_path, report)
    return report


def build_evaluation_plan(
    *,
    formal_config: Mapping[str, Any],
    test_data: Path,
    evaluation_config: Path,
    output_root: Path,
    selection_report: Path,
    models: Mapping[str, tuple[str, str | None]],
) -> dict[str, Any]:
    validate_formal_config(formal_config)
    if set(models) != set(FORMAL_ROLES):
        raise ExperimentError(f"models must contain exactly {FORMAL_ROLES}")
    protocol = load_yaml(evaluation_config)
    expected_protocol = _formal_protocol(formal_config)
    if protocol != expected_protocol:
        raise ExperimentError(
            f"evaluation config differs from the formal protocol: {protocol!r}"
        )
    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    if selection.get("selection_data_split") != "validation":
        raise ExperimentError("checkpoint must be selected on validation")
    commands = {}
    for role in FORMAL_ROLES:
        model, revision = models[role]
        directory = output_root / role
        command = [
            "python",
            "-m",
            "miniopd.evaluation",
            "run",
            "--config",
            str(evaluation_config),
            "--backend",
            "vllm",
            "--model",
            model,
            "--input",
            str(test_data),
            "--output",
            str(directory / "predictions.jsonl"),
            "--summary",
            str(directory / "summary.json"),
        ]
        if revision is not None:
            command.extend(("--model-revision", revision))
        commands[role] = command
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_report": str(selection_report.resolve()),
        "test_data": str(test_data.resolve()),
        "evaluation_config": str(evaluation_config.resolve()),
        "protocol": protocol,
        "commands": commands,
    }


def _validate_formal_evaluation(
    role: str,
    directory: Path,
    expected_rows: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    predictions_path = directory / "predictions.jsonl"
    summary_path = directory / "summary.json"
    if not predictions_path.is_file() or not summary_path.is_file():
        raise ExperimentError(f"{role} evaluation artifacts are incomplete")
    rows = read_jsonl(predictions_path)
    if len(rows) != expected_rows:
        raise ExperimentError(
            f"{role} has {len(rows)} test rows; expected {expected_rows}"
        )
    expected_ids = [f"gsm8k-test-{index:05d}" for index in range(expected_rows)]
    if [row.get("sample_id") for row in rows] != expected_ids:
        raise ExperimentError(f"{role} test IDs are missing, duplicated, or reordered")
    if any(row.get("split") != "test" for row in rows):
        raise ExperimentError(f"{role} evaluation contains non-test rows")
    models = {str(row.get("model")) for row in rows}
    if len(models) != 1:
        raise ExperimentError(f"{role} evaluation mixes model identities")
    for row in rows:
        generation = row.get("generation_config", {})
        if any(
            generation.get(key) != protocol[key]
            for key in (
                "temperature",
                "top_p",
                "do_sample",
                "max_new_tokens",
                "batch_size",
                "dtype",
                "seed",
            )
        ):
            raise ExperimentError(f"{role} did not use the formal greedy BF16 config")
        if row.get("runtime_message") is not None:
            raise ExperimentError(f"{role} contains runtime failures")
    calculated = summarize_records(rows)
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("exact_match", "correct", "total", "tokens_per_second"):
        if saved.get(key) != calculated.get(key):
            raise ExperimentError(f"{role} saved summary mismatch for {key}")
    return {
        "role": role,
        "model": models.pop(),
        "summary": calculated,
        "predictions": str(predictions_path.resolve()),
    }


def _average_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ExperimentError(f"invalid efficiency metric {key}")
    return sum(values) / len(values) if values else None


def _same_model_reference(saved: str, planned: str) -> bool:
    """Compare model IDs literally and local model paths canonically."""
    if saved == planned:
        return True
    saved_path = Path(saved)
    planned_path = Path(planned)
    if not (saved_path.is_absolute() or planned_path.is_absolute()):
        return False
    return saved_path.resolve() == planned_path.resolve()


def summarize_efficiency(
    training_run_dir: Path,
    selection: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    status_path = training_run_dir / "run_status.json"
    if not status_path.is_file():
        raise ExperimentError("formal training run_status.json is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("exit_code") != 0:
        raise ExperimentError("formal training run is not complete")
    metrics_path = training_run_dir / "metrics.jsonl"
    metrics = read_jsonl(metrics_path) if metrics_path.is_file() else []
    if not metrics:
        raise ExperimentError("formal training metrics.jsonl is missing or empty")
    gpu_count = int(status.get("visible_gpu_count") or 0)
    wall_seconds = float(status.get("wall_seconds") or 0)
    if gpu_count <= 0 or wall_seconds <= 0:
        raise ExperimentError("GPU count and training wall time must be recorded")
    peak_memory = 0
    for row in metrics:
        for device in row.get("gpu_memory_bytes", []):
            peak_memory = max(
                peak_memory,
                int(device.get("max_allocated", 0)),
                int(device.get("max_reserved", 0)),
            )
    history_path = training_run_dir / "run_history.jsonl"
    if not history_path.is_file():
        raise ExperimentError("formal run_history.jsonl is missing")
    history = read_jsonl(history_path)
    throughput = {
        "rollout_tokens_per_second": _average_metric(
            metrics, "rollout_tokens_per_second"
        ),
        "teacher_scoring_tokens_per_second": _average_metric(
            metrics, "teacher_scoring_tokens_per_second"
        ),
        "training_tokens_per_second": _average_metric(
            metrics, "training_tokens_per_second"
        ),
    }
    if peak_memory <= 0 or any(value is None for value in throughput.values()):
        raise ExperimentError("formal efficiency metrics are incomplete")
    inference_speed = {
        role: value["summary"]["tokens_per_second"]
        for role, value in evaluations.items()
    }
    if any(
        value is None or not math.isfinite(value) or value <= 0
        for value in inference_speed.values()
    ):
        raise ExperimentError("formal inference speed metrics are incomplete")
    return {
        "gpu_hours": wall_seconds * gpu_count / 3600,
        "wall_seconds": wall_seconds,
        "visible_gpu_count": gpu_count,
        "peak_gpu_memory_bytes": peak_memory,
        "peak_gpu_memory_coverage": sorted(
            {
                str(device.get("scope", "unspecified"))
                for row in metrics
                for device in row.get("gpu_memory_bytes", [])
            }
        ),
        **throughput,
        "training_metric_sources": sorted(
            {
                str(row["record_type"])
                for row in metrics
                if row.get("record_type") is not None
            }
        ),
        "selected_checkpoint_size_bytes": selection["selected_checkpoint"][
            "total_size_bytes"
        ],
        "inference_tokens_per_second": inference_speed,
        "run_history": history,
        "failed_or_interrupted_attempts": [
            row for row in history if row.get("status") in {"failed", "interrupted"}
        ],
    }


def build_final_report(
    *,
    formal_config: Mapping[str, Any],
    selection_report: Path,
    evaluation_plan: Path,
    training_run_dir: Path,
    evaluation_dirs: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    validate_formal_config(formal_config)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if set(evaluation_dirs) != set(FORMAL_ROLES):
        raise ExperimentError(f"evaluation dirs must contain exactly {FORMAL_ROLES}")
    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    if selection.get("selection_data_split") != "validation":
        raise ExperimentError("selected checkpoint is not validation-derived")
    expected_rows = int(formal_config["final_evaluation"]["expected_rows"])
    protocol = _formal_protocol(formal_config)
    evaluations = {
        role: _validate_formal_evaluation(
            role, evaluation_dirs[role], expected_rows, protocol
        )
        for role in FORMAL_ROLES
    }
    plan = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    planned_selection = Path(str(plan.get("selection_report", ""))).resolve()
    if planned_selection != selection_report.resolve():
        raise ExperimentError("evaluation plan uses a different checkpoint selection")
    commands = plan.get("commands", {})
    if set(commands) != set(FORMAL_ROLES):
        raise ExperimentError("evaluation plan does not contain all four formal roles")
    for role in FORMAL_ROLES:
        command = commands[role]
        if not isinstance(command, list) or "--model" not in command:
            raise ExperimentError(f"{role} evaluation plan has no model argument")
        planned_model = str(command[command.index("--model") + 1])
        if not _same_model_reference(evaluations[role]["model"], planned_model):
            raise ExperimentError(
                f"{role} saved predictions do not match the evaluation plan model"
            )
    if plan.get("protocol") != protocol:
        raise ExperimentError("saved evaluation plan differs from formal protocol")
    if plan["protocol"] != load_yaml(Path(str(plan["evaluation_config"]))):
        raise ExperimentError("saved evaluation plan protocol no longer matches YAML")
    scores = {
        role: value["summary"]["exact_match"] for role, value in evaluations.items()
    }
    denominator = scores["teacher"] - scores["base"]
    gains = {
        "sft_gain": scores["sft"] - scores["base"],
        "opd_additional_gain": scores["opd"] - scores["sft"],
        "gap_recovery": (
            (scores["opd"] - scores["base"]) / denominator if denominator != 0 else None
        ),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "gains": gains,
        "evaluations": evaluations,
        "checkpoint_selection": selection,
        "evaluation_plan": str(evaluation_plan.resolve()),
        "efficiency": summarize_efficiency(training_run_dir, selection, evaluations),
        "all_values_from_saved_artifacts": True,
        "test_used_for_checkpoint_selection": False,
    }
    write_json(output_path, report)
    return report


def load_formal_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    validate_formal_config(config)
    return config
