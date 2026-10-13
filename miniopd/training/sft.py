"""Response-only full-parameter BF16 cold-start SFT."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from miniopd.evaluation.io import read_jsonl, write_json
from miniopd.evaluation.prompt import build_sft_messages

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTConfig:
    dtype: str = "bfloat16"
    full_parameter: bool = True
    epochs: float = 1.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    expected_world_size: int = 1
    effective_batch_size: int = 32
    gradient_checkpointing: bool = True
    max_length: int | str = "auto"
    auto_length_multiple: int = 64
    logging_steps: int = 10
    eval_steps: int = 20
    save_steps: int = 20
    save_total_limit: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.dtype != "bfloat16" or not self.full_parameter:
            raise ValueError("the primary SFT path is full-parameter BF16")
        if self.epochs != 1.0 or self.learning_rate != 1e-5:
            raise ValueError("initial SFT fixes epochs=1 and learning_rate=1e-5")
        actual = (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.expected_world_size
        )
        if actual != self.effective_batch_size or actual != 32:
            raise ValueError("configured effective batch size must equal 32")
        if not self.gradient_checkpointing:
            raise ValueError("initial SFT requires gradient checkpointing")
        if self.max_length != "auto" and (
            not isinstance(self.max_length, int) or self.max_length < 1
        ):
            raise ValueError("max_length must be 'auto' or a positive integer")
        if self.auto_length_multiple < 1:
            raise ValueError("auto_length_multiple must be positive")
        if self.eval_steps != self.save_steps:
            raise ValueError("eval_steps and save_steps must match for selection")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SFTConfig:
        unexpected = set(value).difference(cls.__dataclass_fields__)
        if unexpected:
            raise ValueError(f"unknown SFT config keys: {sorted(unexpected)}")
        return cls(**dict(value))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".parquet":
        try:
            import datasets
        except ImportError as error:
            raise RuntimeError("datasets is required to read parquet") from error
        return [dict(row) for row in datasets.Dataset.from_parquet(str(path))]
    raise ValueError("SFT data must be JSONL or parquet")


def validate_sft_splits(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject duplicate rows, wrong splits, and train/validation leakage."""
    if not train_rows or not validation_rows:
        raise ValueError("train and validation data must both be non-empty")
    if any(row.get("split") != "sft" for row in train_rows):
        raise ValueError("training data must contain only the fixed SFT split")
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("checkpoint selection must use only the validation split")

    def unique_values(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
        values = [str(row.get(key, "")) for row in rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError(f"missing or duplicate {key} values")
        return set(values)

    train_ids = unique_values(train_rows, "sample_id")
    validation_ids = unique_values(validation_rows, "sample_id")
    train_questions = unique_values(train_rows, "question")
    validation_questions = unique_values(validation_rows, "question")
    if train_ids & validation_ids or train_questions & validation_questions:
        raise ValueError("SFT train and validation data leak into each other")
    if any(not sample_id.startswith("gsm8k-train-") for sample_id in train_ids):
        raise ValueError("SFT training may only use GSM8K train rows")
    if any(not sample_id.startswith("gsm8k-train-") for sample_id in validation_ids):
        raise ValueError("validation may only use held-out GSM8K train rows")


def _training_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("SFT row must contain messages")
    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    roles = [message["role"] for message in normalized]
    if roles == ["system", "user", "assistant"]:
        return normalized
    if roles == ["system", "user"] and row.get("raw_answer"):
        reference = str(row.get("reference_answer", "")).strip()
        solution = str(row["raw_answer"]).rsplit("####", 1)[0].rstrip()
        response = f"{solution}\nThe final answer is \\boxed{{{reference}}}."
        return build_sft_messages(str(row["question"]), response)
    raise ValueError("SFT messages must have system, user, assistant roles")


def tokenize_conversation(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    """Tokenize one conversation and mask every non-assistant target token."""
    messages = _training_messages(row)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if not full_ids[: len(prompt_ids)] == prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the SFT sequence")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("SFT row has no assistant target tokens")
    return {
        "sample_id": str(row.get("sample_id", "")),
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        "prompt_tokens": len(prompt_ids),
        "total_tokens": len(full_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
    }


def _percentile(sorted_values: Sequence[int], probability: float) -> int:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of empty data")
    index = max(0, math.ceil(probability * len(sorted_values)) - 1)
    return int(sorted_values[index])


def length_report(
    encoded: Sequence[Mapping[str, Any]], multiple: int = 64
) -> dict[str, Any]:
    if not encoded:
        raise ValueError("SFT dataset is empty")
    lengths = sorted(int(row["total_tokens"]) for row in encoded)
    maximum = lengths[-1]
    recommended = math.ceil(maximum / multiple) * multiple
    return {
        "count": len(lengths),
        "min": lengths[0],
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": maximum,
        "recommended_max_length": recommended,
        "recommendation_rule": f"observed maximum rounded up to {multiple}",
    }


def inspect_sft_lengths(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    model_revision: str,
    output_path: Path,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for length inspection") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, trust_remote_code=trust_remote_code
    )
    report = length_report([tokenize_conversation(tokenizer, row) for row in rows])
    report.update({"model": model_id, "model_revision": model_revision})
    write_json(output_path, report)
    return report


class _EncodedDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


class ResponseOnlyCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for SFT") from error
        width = min(
            self.max_length, max(len(feature["input_ids"]) for feature in features)
        )

        def padded(name: str, pad_value: int) -> list[list[int]]:
            rows = []
            for feature in features:
                values = list(feature[name])[:width]
                rows.append(values + [pad_value] * (width - len(values)))
            return rows

        return {
            "input_ids": torch.tensor(padded("input_ids", self.pad_token_id)),
            "attention_mask": torch.tensor(padded("attention_mask", 0)),
            "labels": torch.tensor(padded("labels", IGNORE_INDEX)),
        }


def _write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_sft_training(
    *,
    train_input: Path,
    validation_input: Path,
    output_dir: Path,
    model_id: str,
    model_revision: str,
    config: SFTConfig,
    smoke: bool = False,
    resume_from_checkpoint: Path | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run SFT; GPU execution is intentionally deferred to the training machine."""
    try:
        import torch
        import yaml
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "SFT requires torch, transformers, accelerate, and PyYAML"
        ) from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-parameter BF16 SFT requires a BF16-capable CUDA GPU")

    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = load_rows(train_input)
    validation_rows = load_rows(validation_input)
    validate_sft_splits(train_rows, validation_rows)
    if smoke:
        train_rows = train_rows[:32]
        validation_rows = validation_rows[:32]
    train_encoded = [tokenize_conversation(tokenizer, row) for row in train_rows]
    validation_encoded = [
        tokenize_conversation(tokenizer, row) for row in validation_rows
    ]
    lengths = length_report(train_encoded, config.auto_length_multiple)
    max_length = (
        lengths["recommended_max_length"]
        if config.max_length == "auto"
        else int(config.max_length)
    )
    if max_length < lengths["max"]:
        raise ValueError(
            f"max_length={max_length} truncates training data; "
            f"observed max={lengths['max']}"
        )
    model_limit = int(getattr(tokenizer, "model_max_length", max_length))
    if model_limit < 10**9 and max_length > model_limit:
        raise ValueError(
            f"resolved max_length={max_length} exceeds model limit={model_limit}"
        )

    resolved = {
        **config.as_dict(),
        "resolved_max_length": max_length,
        "model_id": model_id,
        "model_revision": model_revision,
        "train_input": str(train_input.resolve()),
        "validation_input": str(validation_input.resolve()),
        "smoke": smoke,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    write_json(output_dir / "length_report.json", lengths)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config.expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, expected {config.expected_world_size}; "
            "this would change effective batch size"
        )
    checkpoint_dir = output_dir / "checkpoints"
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.epochs,
        max_steps=5 if smoke else -1,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        logging_steps=1 if smoke else config.logging_steps,
        eval_strategy="steps",
        eval_steps=2 if smoke else config.eval_steps,
        save_strategy="steps",
        save_steps=2 if smoke else config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_EncodedDataset(train_encoded),
        eval_dataset=_EncodedDataset(validation_encoded),
        data_collator=ResponseOnlyCollator(tokenizer.pad_token_id, max_length),
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    resume_verified = False
    if smoke and resume_from_checkpoint is None:
        trainer.args.max_steps = 2
        trainer.train()
        smoke_checkpoint = checkpoint_dir / "checkpoint-2"
        if not smoke_checkpoint.is_dir():
            raise RuntimeError("smoke phase did not save checkpoint-2")
        trainer.args.max_steps = 5
        result = trainer.train(resume_from_checkpoint=str(smoke_checkpoint))
        resume_verified = True
    else:
        result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )
    wall_seconds = time.perf_counter() - started
    final_dir = output_dir / "final_model"
    trainer.model.config.use_cache = True
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    _write_metrics(output_dir / "metrics.jsonl", trainer.state.log_history)

    best_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(
        final_dir, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    ).cuda()
    reloaded.eval()
    prompt = train_rows[0]["messages"][:2]
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(reloaded.device)
    with torch.inference_mode():
        generated = reloaded.generate(inputs, max_new_tokens=32)
    verification = tokenizer.decode(
        generated[0, inputs.shape[1] :], skip_special_tokens=True
    )
    if not verification.strip():
        raise RuntimeError(
            "independently reloaded SFT model generated an empty response"
        )
    write_json(output_dir / "load_and_generate_check.json", {"response": verification})

    summary = {
        "scope": "coldstart_sft_smoke" if smoke else "coldstart_sft_full",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "model_id": model_id,
        "model_revision": model_revision,
        "best_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "train_runtime_seconds": result.metrics.get("train_runtime", wall_seconds),
        "wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds * torch.cuda.device_count() / 3600,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "final_model": str(final_dir.resolve()),
        "resolved_max_length": max_length,
        "effective_batch_size": config.effective_batch_size,
        "resume_verified": resume_verified,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
