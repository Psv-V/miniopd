"""Cold-start SFT command line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from miniopd.training.sft import (
    SFTConfig,
    inspect_sft_lengths,
    load_rows,
    run_sft_training,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miniopd.training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lengths = subparsers.add_parser("lengths", help="inspect real SFT token lengths")
    lengths.add_argument("--train-input", type=Path, required=True)
    lengths.add_argument("--model-config", type=Path, required=True)
    lengths.add_argument("--output", type=Path, required=True)
    lengths.add_argument("--trust-remote-code", action="store_true")

    train = subparsers.add_parser("sft", help="run full cold-start SFT")
    train.add_argument("--train-input", type=Path, required=True)
    train.add_argument("--validation-input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-config", type=Path, required=True)
    train.add_argument(
        "--model-path",
        type=Path,
        help=(
            "load weights from a verified local snapshot while preserving "
            "the canonical model ID and revision"
        ),
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume-from-checkpoint", type=Path)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    model = _load_yaml(args.model_config)
    if model.get("role") != "base_student":
        raise ValueError("SFT must initialize from the base student config")
    if args.command == "lengths":
        report = inspect_sft_lengths(
            load_rows(args.train_input),
            model_id=str(model["model_id"]),
            model_revision=str(model["revision"]),
            output_path=args.output,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        report = run_sft_training(
            train_input=args.train_input,
            validation_input=args.validation_input,
            output_dir=args.output_dir,
            model_id=str(model["model_id"]),
            model_revision=str(model["revision"]),
            model_path=args.model_path,
            config=SFTConfig.from_mapping(_load_yaml(args.config)),
            smoke=args.smoke,
            resume_from_checkpoint=args.resume_from_checkpoint,
            trust_remote_code=args.trust_remote_code,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
