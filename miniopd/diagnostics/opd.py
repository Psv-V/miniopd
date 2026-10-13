"""Masked OPD training and distribution diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from miniopd.evaluation.io import append_jsonl
from miniopd.losses.masking import masked_mean


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Configuration for diagnostics that are excluded from optimization."""

    enabled: bool = True
    interval_steps: int = 50
    sample_size: int = 4
    top_k: int = 16
    exact_entropy: bool = True
    position_segments: int = 3
    fixed_batch_seed: int = 42
    max_overhead_ratio: float = 0.10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DiagnosticsConfig:
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        if self.interval_steps <= 0:
            raise ValueError("diagnostics.interval_steps must be positive")
        if self.sample_size <= 0:
            raise ValueError("diagnostics.sample_size must be positive")
        if self.top_k <= 0:
            raise ValueError("diagnostics.top_k must be positive")
        if self.position_segments != 3:
            raise ValueError("MiniOPD currently requires exactly three segments")
        if not self.exact_entropy:
            raise ValueError(
                "this implementation computes exact entropy; approximate entropy "
                "must use a separately named metric"
            )
        if not 0 <= self.max_overhead_ratio <= 1:
            raise ValueError("diagnostics.max_overhead_ratio must be in [0, 1]")

    def should_run(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be non-negative")
        return self.enabled and step % self.interval_steps == 0


@dataclass(frozen=True)
class DiagnosticsRecorder:
    """Append-only files for light step metrics and periodic diagnostics."""

    run_dir: Path
    config: DiagnosticsConfig

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def diagnostics_path(self) -> Path:
        return self.run_dir / "diagnostics.jsonl"

    @property
    def fixed_batch_path(self) -> Path:
        return self.run_dir / "diagnostic_fixed_batch.json"

    def initialize_fixed_batch(self, candidate_ids: Sequence[str]) -> list[str]:
        selected = select_fixed_sample_ids(
            candidate_ids,
            sample_size=self.config.sample_size,
            seed=self.config.fixed_batch_seed,
        )
        payload = {
            "sample_ids": selected,
            "sample_size": self.config.sample_size,
            "seed": self.config.fixed_batch_seed,
        }
        if self.fixed_batch_path.exists():
            existing = json.loads(self.fixed_batch_path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("existing diagnostic fixed batch does not match")
            return selected
        self.fixed_batch_path.parent.mkdir(parents=True, exist_ok=True)
        self.fixed_batch_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return selected

    def record_step(self, record: Mapping[str, Any]) -> None:
        if "step" not in record:
            raise ValueError("step metric record is missing step")
        append_jsonl(self.metrics_path, [record])

    def record_periodic(self, record: Mapping[str, Any]) -> None:
        if "step" not in record:
            raise ValueError("periodic diagnostic record is missing step")
        step = int(record["step"])
        if not self.config.should_run(step):
            raise ValueError(f"step {step} is outside the diagnostic schedule")
        append_jsonl(self.diagnostics_path, [record])


def select_fixed_sample_ids(
    sample_ids: Sequence[str], *, sample_size: int, seed: int
) -> list[str]:
    """Select the same diagnostic examples across runs without mutable RNG state."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    if sample_size > len(sample_ids):
        raise ValueError("sample_size exceeds available sample IDs")

    def key(sample_id: str) -> bytes:
        return hashlib.sha256(
            f"miniopd-diagnostics:{seed}:{sample_id}".encode()
        ).digest()

    return sorted(sample_ids, key=key)[:sample_size]


def position_segment_masks(
    response_mask: torch.Tensor, *, segments: int = 3
) -> torch.Tensor:
    """Split each response by its valid-token rank, never by padded width."""
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, token]")
    if segments != 3:
        raise ValueError("MiniOPD position diagnostics require three segments")
    valid = response_mask.bool()
    lengths = valid.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("every response must contain at least one valid token")
    ranks = valid.to(torch.int64).cumsum(dim=1) - 1
    segment_ids = torch.div(
        ranks * segments,
        lengths.unsqueeze(1),
        rounding_mode="floor",
    ).clamp_max(segments - 1)
    return torch.stack(
        [valid & segment_ids.eq(index) for index in range(segments)], dim=0
    )


def _selected(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and response_mask must have identical shapes")
    selected = values.detach().float().masked_select(mask.bool())
    if selected.numel() == 0:
        raise ValueError("diagnostic mask selects no values")
    if not torch.isfinite(selected).all():
        raise FloatingPointError("diagnostic values contain NaN or Inf")
    return selected


def _mean_flags(values: torch.Tensor | Sequence[bool], name: str) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    return float(tensor.mean().item())


def summarize_step_metrics(
    *,
    step: int,
    token_reward: torch.Tensor,
    response_mask: torch.Tensor,
    policy_loss: float | torch.Tensor,
    distillation_loss: float | torch.Tensor,
    gradient_norm: float | torch.Tensor,
    rollout_correct: torch.Tensor | Sequence[bool],
    truncated: torch.Tensor | Sequence[bool],
    format_error: torch.Tensor | Sequence[bool],
    timing_seconds: Mapping[str, float],
    token_counts: Mapping[str, int],
    gpu_memory_bytes: Sequence[Mapping[str, int]] = (),
) -> dict[str, Any]:
    """Summarize existing step tensors without performing another model forward."""
    selected = _selected(token_reward, response_mask)
    response_lengths = response_mask.bool().sum(dim=1).float()
    timings = {}
    throughput = {}
    for phase in ("rollout", "teacher_scoring", "training"):
        seconds = float(timing_seconds.get(phase, 0.0))
        count = int(token_counts.get(phase, 0))
        if seconds < 0 or count < 0:
            raise ValueError("timings and token counts must be non-negative")
        timings[f"{phase}_seconds"] = seconds
        throughput[f"{phase}_tokens_per_second"] = (
            count / seconds if seconds > 0 else None
        )
    return {
        "step": step,
        "token_reward_mean": float(selected.mean().item()),
        "token_reward_std": float(selected.std(unbiased=False).item()),
        "token_reward_min": float(selected.min().item()),
        "token_reward_max": float(selected.max().item()),
        "policy_loss": float(torch.as_tensor(policy_loss).detach().item()),
        "distillation_loss": float(torch.as_tensor(distillation_loss).detach().item()),
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().item()),
        "response_length_mean": float(response_lengths.mean().item()),
        "response_length_min": int(response_lengths.min().item()),
        "response_length_max": int(response_lengths.max().item()),
        "rollout_accuracy": _mean_flags(rollout_correct, "rollout_correct"),
        "truncated_ratio": _mean_flags(truncated, "truncated"),
        "format_error_ratio": _mean_flags(format_error, "format_error"),
        **timings,
        **throughput,
        "gpu_memory_bytes": [dict(item) for item in gpu_memory_bytes],
    }


def _distribution_snapshot(logits: torch.Tensor, *, k: int) -> dict[str, torch.Tensor]:
    """Reduce full-vocabulary logits on their owning device, then transfer to CPU."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, token, vocab]")
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError("top_k exceeds vocabulary size")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    top_values, top_ids = log_probs.topk(k, dim=-1)
    if not torch.isfinite(entropy).all():
        raise FloatingPointError("entropy contains NaN or Inf")
    return {
        "entropy": entropy.detach().cpu(),
        "top_log_probs": top_values.detach().cpu(),
        "top_ids": top_ids.detach().cpu(),
    }


def _topk_per_token(
    student: Mapping[str, torch.Tensor],
    teacher: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    student_ids = student["top_ids"]
    teacher_ids = teacher["top_ids"]
    student_top_logp = student["top_log_probs"]
    teacher_top_logp = teacher["top_log_probs"]
    shared = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2)).any(dim=-1)
    matches = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2))
    teacher_logp_at_student_ids = (
        matches.float() * teacher_top_logp.unsqueeze(-2)
    ).sum(dim=-1)
    shared_float = shared.float()
    shared_count = shared_float.sum(dim=-1)
    advantage_sum = (
        (teacher_logp_at_student_ids - student_top_logp) * shared_float
    ).sum(dim=-1)
    advantage = torch.where(
        shared_count > 0,
        advantage_sum / shared_count.clamp_min(1),
        torch.zeros_like(advantage_sum),
    )
    return {
        "overlap_ratio": shared_count / student_ids.shape[-1],
        "overlap_token_advantage": advantage,
        "student_overlap_probability_mass": (student_top_logp.exp() * shared_float).sum(
            dim=-1
        ),
        "teacher_overlap_probability_mass": (
            teacher_logp_at_student_ids.exp() * shared_float
        ).sum(dim=-1),
    }


def _optional_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not mask.any():
        return None
    return float(masked_mean(values, mask).item())


@torch.no_grad()
def compute_periodic_diagnostics(
    *,
    step: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_reward: torch.Tensor,
    response_mask: torch.Tensor,
    config: DiagnosticsConfig,
    diagnostic_seconds: float | None = None,
    reference_step_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute exact-entropy and Top-k diagnostics outside the training graph."""
    config.validate()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits shapes must match")
    if student_logits.shape[:2] != response_mask.shape:
        raise ValueError("response_mask must match logits batch/token dimensions")
    if token_reward.shape != response_mask.shape:
        raise ValueError("token_reward and response_mask shapes must match")
    mask = response_mask.detach().bool().cpu()
    reward = token_reward.detach().float().cpu()
    if not mask.any():
        raise ValueError("response_mask selects no tokens")

    student = _distribution_snapshot(student_logits, k=config.top_k)
    teacher = _distribution_snapshot(teacher_logits, k=config.top_k)
    student_entropy = student["entropy"]
    teacher_entropy = teacher["entropy"]
    topk = _topk_per_token(student, teacher)
    result: dict[str, Any] = {
        "step": step,
        "sample_size": student_logits.shape[0],
        "valid_response_tokens": int(mask.sum().item()),
        "entropy_kind": "exact_full_vocabulary",
        "top_k": config.top_k,
        "student_entropy_exact": float(masked_mean(student_entropy, mask).item()),
        "teacher_entropy_exact": float(masked_mean(teacher_entropy, mask).item()),
        "absolute_entropy_gap_exact": float(
            masked_mean((student_entropy - teacher_entropy).abs(), mask).item()
        ),
        "token_reward_mean": float(masked_mean(reward, mask).item()),
    }
    result.update(
        {
            f"top{config.top_k}_{name}": float(masked_mean(value, mask).item())
            for name, value in topk.items()
        }
    )
    segment_names = ("front", "middle", "back")
    segment_masks = position_segment_masks(mask, segments=config.position_segments)
    segments = {}
    for name, segment_mask in zip(segment_names, segment_masks, strict=True):
        segments[name] = {
            "valid_tokens": int(segment_mask.sum().item()),
            "student_entropy_exact": _optional_masked_mean(
                student_entropy, segment_mask
            ),
            "teacher_entropy_exact": _optional_masked_mean(
                teacher_entropy, segment_mask
            ),
            "token_reward_mean": _optional_masked_mean(reward, segment_mask),
            f"top{config.top_k}_overlap_ratio": _optional_masked_mean(
                topk["overlap_ratio"], segment_mask
            ),
        }
    result["segments"] = segments
    if diagnostic_seconds is not None or reference_step_seconds is not None:
        if diagnostic_seconds is None or reference_step_seconds is None:
            raise ValueError("both diagnostic and reference durations are required")
        if diagnostic_seconds < 0 or reference_step_seconds <= 0:
            raise ValueError("invalid diagnostic overhead durations")
        ratio = diagnostic_seconds / reference_step_seconds
        result["diagnostic_seconds"] = diagnostic_seconds
        result["diagnostic_overhead_ratio"] = ratio
        result["diagnostic_overhead_within_budget"] = ratio <= config.max_overhead_ratio
    return result
