"""Finalize auditable diagnostics and efficiency metrics for a formal OPD run."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from miniopd.diagnostics.finalize_e2e import (
    _forward,
    _load_rollouts,
    _trajectory_batch,
    _write_jsonl,
    parse_step_metrics,
)
from miniopd.diagnostics.opd import DiagnosticsConfig, compute_periodic_diagnostics
from miniopd.evaluation.io import write_json
from miniopd.integration.e2e_smoke import record_tensor_audit
from miniopd.integration.verl_opd import load_yaml

GIB = 1024**3


def normalize_formal_step_metrics(
    rows: Sequence[Mapping[str, Any]], *, rollout_samples_per_step: int
) -> list[dict[str, Any]]:
    """Normalize verl console metrics without inventing unavailable measurements."""
    if rollout_samples_per_step <= 0:
        raise ValueError("rollout_samples_per_step must be positive")
    normalized = []
    for row in rows:
        required = {
            "step",
            "response_length/mean",
            "timing_s/gen",
            "timing_s/update_actor",
            "perf/total_num_tokens",
            "actor/perf/max_memory_allocated_gb",
            "actor/perf/max_memory_reserved_gb",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"step {row.get('step')} is missing metrics: {missing}")
        rollout_seconds = float(row["timing_s/gen"])
        training_seconds = float(row["timing_s/update_actor"])
        response_tokens = float(row["response_length/mean"]) * (
            rollout_samples_per_step
        )
        training_tokens = int(row["perf/total_num_tokens"])
        allocated = float(row["actor/perf/max_memory_allocated_gb"])
        reserved = float(row["actor/perf/max_memory_reserved_gb"])
        values = (
            rollout_seconds,
            training_seconds,
            response_tokens,
            training_tokens,
            allocated,
            reserved,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in values):
            raise ValueError(f"step {row['step']} contains invalid efficiency metrics")
        normalized.append(
            {
                **dict(row),
                "record_type": "training_step_recovered_from_console_log",
                "rollout_samples": rollout_samples_per_step,
                "rollout_response_tokens": response_tokens,
                "rollout_seconds": rollout_seconds,
                "rollout_tokens_per_second": response_tokens / rollout_seconds,
                "training_tokens": training_tokens,
                "training_seconds": training_seconds,
                "training_tokens_per_second": training_tokens / training_seconds,
                "gpu_memory_bytes": [
                    {
                        "device": 0,
                        "scope": "actor_training_worker",
                        "max_allocated": round(allocated * GIB),
                        "max_reserved": round(reserved * GIB),
                    }
                ],
            }
        )
    return normalized


def validate_step_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    final_step: int,
    allowed_missing_steps: Sequence[int],
    checkpoint_root: Path,
) -> list[int]:
    """Require complete coverage except explicitly evidenced checkpointed steps."""
    if final_step <= 0:
        raise ValueError("final_step must be positive")
    steps = [int(row["step"]) for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError("parsed metrics contain duplicate steps")
    expected = set(range(1, final_step + 1))
    unexpected = sorted(set(steps) - expected)
    if unexpected:
        raise ValueError(f"metrics contain unexpected steps: {unexpected}")
    missing = sorted(expected - set(steps))
    allowed = sorted({int(step) for step in allowed_missing_steps})
    if missing != allowed:
        raise ValueError(f"missing steps {missing}; explicitly allowed {allowed}")
    for step in missing:
        checkpoint = checkpoint_root / f"global_step_{step}"
        if not checkpoint.is_dir():
            raise ValueError(f"missing step {step} has no checkpoint evidence")
    return missing


@torch.no_grad()
def _score_teacher_once(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    sampled = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    )
    if not torch.isfinite(sampled.squeeze(-1).masked_select(response_mask)).all():
        raise FloatingPointError("teacher benchmark produced non-finite log-probs")


@torch.no_grad()
def benchmark_teacher_scoring(
    *,
    model_path: Path,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
    device: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Measure teacher sampled-token scoring on a fixed saved trajectory batch."""
    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("invalid teacher benchmark iteration counts")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.eval()
    ids = input_ids.to(device)
    attention_mask = ids.ne(pad_token_id)
    targets = ids[:, 1:]
    mask = response_mask.to(device)

    for _ in range(warmup_iterations):
        _score_teacher_once(model, ids, attention_mask, targets, mask)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for _ in range(measured_iterations):
        _score_teacher_once(model, ids, attention_mask, targets, mask)
    torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    response_tokens_per_iteration = int(response_mask.sum().item())
    total_tokens = response_tokens_per_iteration * measured_iterations
    result = {
        "record_type": "posthoc_fixed_trajectory_teacher_scoring_benchmark",
        "metric_scope": "sampled_response_tokens",
        "device": device,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "response_tokens_per_iteration": response_tokens_per_iteration,
        "teacher_scoring_tokens": total_tokens,
        "teacher_scoring_seconds": seconds,
        "teacher_scoring_tokens_per_second": total_tokens / seconds,
        "benchmark_gpu_memory_bytes": {
            "max_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    del model, ids, attention_mask, targets, mask
    torch.cuda.empty_cache()
    return result


def _refuse_existing(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite formal artifacts: {existing}")


def finalize(
    *,
    run_dir: Path,
    log_paths: Sequence[Path],
    student_model: Path,
    teacher_model: Path,
    config_path: Path,
    rollout_step: int,
    final_step: int,
    allowed_missing_steps: Sequence[int],
    rollout_samples_per_step: int,
    sample_size: int,
    teacher_device: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    outputs = [
        run_dir / "metrics.jsonl",
        run_dir / "metrics_provenance.json",
        run_dir / "teacher_scoring_benchmark.json",
        run_dir / "diagnostics.jsonl",
        run_dir / "diagnostic_fixed_batch.json",
        run_dir / "tensor_audit.jsonl",
    ]
    _refuse_existing(outputs)
    unique_logs = list(dict.fromkeys(path.resolve() for path in log_paths))
    missing_logs = [str(path) for path in unique_logs if not path.is_file()]
    if missing_logs:
        raise FileNotFoundError(f"formal training logs do not exist: {missing_logs}")
    parsed = parse_step_metrics(unique_logs)
    missing_steps = validate_step_coverage(
        parsed,
        final_step=final_step,
        allowed_missing_steps=allowed_missing_steps,
        checkpoint_root=run_dir / "checkpoints",
    )
    metrics = normalize_formal_step_metrics(
        parsed, rollout_samples_per_step=rollout_samples_per_step
    )

    rollout_path = run_dir / "rollouts" / f"{rollout_step}.jsonl"
    rollout_rows = _load_rollouts(rollout_path, sample_size)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(student_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("diagnostic tokenizer must define pad_token_id")
    input_ids, response_mask = _trajectory_batch(tokenizer, rollout_rows)

    benchmark = benchmark_teacher_scoring(
        model_path=teacher_model,
        input_ids=input_ids,
        response_mask=response_mask,
        pad_token_id=tokenizer.pad_token_id,
        device=teacher_device,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    benchmark["teacher_model"] = str(teacher_model.resolve())
    benchmark["rollout_artifact"] = str(rollout_path.resolve())
    benchmark["rollout_step"] = rollout_step
    metrics.append(dict(benchmark))

    started = time.monotonic()
    student_logits = _forward(
        student_model,
        input_ids,
        "cuda:0",
        pad_token_id=tokenizer.pad_token_id,
    )
    teacher_logits = _forward(
        teacher_model,
        input_ids,
        teacher_device,
        pad_token_id=tokenizer.pad_token_id,
    )
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        shared_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        if int(input_ids.max()) >= shared_vocab:
            raise ValueError(
                "trajectory includes a token outside the shared vocabulary"
            )
        student_logits = student_logits[..., :shared_vocab]
        teacher_logits = teacher_logits[..., :shared_vocab]
    diagnostic_seconds = time.monotonic() - started
    targets = input_ids[:, 1:]
    student_log_probs = (
        torch.log_softmax(student_logits.float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    teacher_log_probs = (
        torch.log_softmax(teacher_logits.float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    token_reward = (teacher_log_probs - student_log_probs).detach()
    opd_config = load_yaml(config_path)
    diagnostics_config = DiagnosticsConfig.from_mapping(opd_config["diagnostics"])
    periodic = compute_periodic_diagnostics(
        step=rollout_step,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_reward=token_reward,
        response_mask=response_mask,
        config=diagnostics_config,
        diagnostic_seconds=diagnostic_seconds,
        reference_step_seconds=mean(float(row["timing_s/step"]) for row in parsed),
    )
    sample_ids = [
        str(row.get("sample_id") or f"step{rollout_step}-row-{index}")
        for index, row in enumerate(rollout_rows)
    ]
    periodic.update(
        {
            "sample_ids": sample_ids,
            "student_checkpoint": str(student_model.resolve()),
            "teacher_model": str(teacher_model.resolve()),
            "artifact_origin": "posthoc_fixed_saved_trajectories",
        }
    )

    temporary_audit = run_dir / ".tensor_audit.jsonl.tmp"
    record_tensor_audit(
        temporary_audit,
        step=rollout_step,
        input_ids=input_ids,
        shifted_logits=student_logits,
        target_ids=targets,
        response_mask=response_mask,
        teacher_log_probs=teacher_log_probs.detach(),
        old_student_log_probs=student_log_probs.detach(),
        advantage=token_reward,
        current_student_log_probs=student_log_probs,
    )

    _write_jsonl(run_dir / "metrics.jsonl", metrics)
    _write_jsonl(run_dir / "diagnostics.jsonl", [periodic])
    os.replace(temporary_audit, run_dir / "tensor_audit.jsonl")
    write_json(
        run_dir / "diagnostic_fixed_batch.json",
        {
            "sample_ids": sample_ids,
            "sample_size": sample_size,
            "rollout_artifact": str(rollout_path.resolve()),
            "rollout_step": rollout_step,
        },
    )
    write_json(run_dir / "teacher_scoring_benchmark.json", benchmark)
    provenance = {
        "training_step_records": len(parsed),
        "expected_final_step": final_step,
        "missing_steps": missing_steps,
        "missing_step_evidence": {
            str(step): str((run_dir / "checkpoints" / f"global_step_{step}").resolve())
            for step in missing_steps
        },
        "log_paths": [str(path) for path in unique_logs],
        "rollout_samples_per_step": rollout_samples_per_step,
        "rollout_throughput_formula": (
            "response_length/mean * rollout_samples_per_step / timing_s/gen"
        ),
        "training_throughput_formula": (
            "perf/total_num_tokens / timing_s/update_actor"
        ),
        "teacher_throughput_source": (
            "posthoc fixed saved-trajectory sampled-token scoring benchmark"
        ),
        "old_log_prob_not_used_as_teacher_timing": True,
        "peak_memory_coverage": "actor_training_worker_only",
    }
    write_json(run_dir / "metrics_provenance.json", provenance)
    return {
        "training_step_metrics": len(parsed),
        "missing_steps": missing_steps,
        "teacher_scoring_tokens_per_second": benchmark[
            "teacher_scoring_tokens_per_second"
        ],
        "diagnostic_records": 1,
        "tensor_audit_records": 1,
        "outputs": [str(path) for path in outputs],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--log",
        dest="log_paths",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--rollout-step", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument(
        "--allow-missing-step",
        dest="allowed_missing_steps",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--rollout-samples-per-step", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measured-iterations", type=int, default=3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(finalize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
