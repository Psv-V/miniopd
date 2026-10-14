"""Finalize Stage 13 diagnostics from completed verl E2E smoke artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from miniopd.data.gsm8k import SYSTEM_PROMPT
from miniopd.diagnostics.opd import (
    DiagnosticsConfig,
    compute_periodic_diagnostics,
)
from miniopd.evaluation.io import write_json
from miniopd.integration.e2e_smoke import record_tensor_audit

_STEP_LINE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
_METRIC = re.compile(r"(?:^|\s+-\s+)([^:]+):([^\s]+)")


def _number(value: str) -> int | float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value}")
    return int(number) if number.is_integer() else number


def parse_step_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    """Parse the final console metric line for every completed training step."""
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _STEP_LINE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            values: dict[str, Any] = {"step": step, "source": str(path)}
            for metric in _METRIC.finditer(match.group(2)):
                value = _number(metric.group(2))
                if value is not None:
                    values[metric.group(1).strip()] = value
            records[step] = values
    return [records[step] for step in sorted(records)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_rollouts(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < limit:
        raise ValueError(f"{path} has {len(rows)} rows; need {limit}")
    return rows[:limit]


def _trajectory_batch(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences: list[list[int]] = []
    response_starts: list[int] = []
    prefix = f"system\n{SYSTEM_PROMPT}\nuser\n"
    for row in rows:
        rendered = str(row["input"])
        if not rendered.startswith(prefix):
            raise ValueError("rollout input does not match MiniOPD's fixed prompt")
        question = rendered[len(prefix) :]
        prompt_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_ids = tokenizer.encode(str(row["output"]), add_special_tokens=False)
        if len(response_ids) < 128 and tokenizer.eos_token_id is not None:
            response_ids.append(tokenizer.eos_token_id)
        response_starts.append(len(prompt_ids))
        sequences.append([*prompt_ids, *response_ids])
    width = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long
    )
    response_mask = torch.zeros((len(sequences), width - 1), dtype=torch.bool)
    paired = zip(sequences, response_starts, strict=True)
    for index, (tokens, start) in enumerate(paired):
        input_ids[index, : len(tokens)] = torch.tensor(tokens)
        response_mask[index, start - 1 : len(tokens) - 1] = True
    return input_ids, response_mask


def _forward(
    model_path: Path,
    input_ids: torch.Tensor,
    device: str,
    *,
    pad_token_id: int,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.eval()
    # Qwen's teacher config does not declare a pad token even though the shared
    # tokenizer does.  The batch was padded by that tokenizer, so its id is the
    # authoritative value for both models.
    attention_mask = input_ids.ne(pad_token_id).to(device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask
        ).logits[:, :-1]
    output = logits.cpu()
    del logits, model
    torch.cuda.empty_cache()
    return output


def finalize(
    *, run_dir: Path, student_model: Path, teacher_model: Path, sample_size: int
) -> dict[str, Any]:
    metrics = parse_step_metrics(
        [run_dir / "phase1/run.log", run_dir / "phase2/run.log"]
    )
    if [row["step"] for row in metrics] != [1, 2, 3, 4, 5]:
        raise ValueError("expected completed step metrics 1 through 5")
    _write_jsonl(run_dir / "metrics.jsonl", metrics)

    rollout_path = run_dir / "rollouts/5.jsonl"
    rows = _load_rollouts(rollout_path, sample_size)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(student_model, local_files_only=True)
    input_ids, response_mask = _trajectory_batch(tokenizer, rows)
    started = time.monotonic()
    if tokenizer.pad_token_id is None:
        raise ValueError("diagnostic tokenizer must define pad_token_id")
    student_logits = _forward(
        student_model,
        input_ids,
        "cuda:0",
        pad_token_id=tokenizer.pad_token_id,
    )
    teacher_logits = _forward(
        teacher_model,
        input_ids,
        "cuda:1",
        pad_token_id=tokenizer.pad_token_id,
    )
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        shared_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
        if int(input_ids.max()) >= shared_vocab_size:
            raise ValueError(
                "trajectory contains a token outside the shared vocabulary"
            )
        student_logits = student_logits[..., :shared_vocab_size]
        teacher_logits = teacher_logits[..., :shared_vocab_size]
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
    config = DiagnosticsConfig(
        interval_steps=1,
        sample_size=sample_size,
        top_k=16,
        fixed_batch_seed=42,
        max_overhead_ratio=1.0,
    )
    periodic = compute_periodic_diagnostics(
        step=5,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_reward=token_reward,
        response_mask=response_mask,
        config=config,
        diagnostic_seconds=diagnostic_seconds,
        reference_step_seconds=float(metrics[-1]["timing_s/step"]),
    )
    periodic["sample_ids"] = [f"step5-row-{index}" for index in range(sample_size)]
    periodic["student_checkpoint"] = str(student_model.resolve())
    periodic["teacher_model"] = str(teacher_model.resolve())
    _write_jsonl(run_dir / "diagnostics.jsonl", [periodic])
    write_json(
        run_dir / "diagnostic_fixed_batch.json",
        {"sample_ids": periodic["sample_ids"], "sample_size": sample_size, "seed": 42},
    )
    record_tensor_audit(
        run_dir / "tensor_audit.jsonl",
        step=5,
        input_ids=input_ids,
        shifted_logits=student_logits,
        target_ids=targets,
        response_mask=response_mask,
        teacher_log_probs=teacher_log_probs.detach(),
        old_student_log_probs=student_log_probs.detach(),
        advantage=token_reward,
        current_student_log_probs=student_log_probs,
    )
    return {
        "metrics": len(metrics),
        "diagnostics": 1,
        "tensor_audits": 1,
        "diagnostic_seconds": diagnostic_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(finalize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
