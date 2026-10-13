"""CLI for formal OPD experiment selection and reporting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from miniopd.evaluation.io import write_json
from miniopd.experiments.formal import (
    build_evaluation_plan,
    build_final_report,
    build_validation_metrics,
    load_formal_config,
    select_validation_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miniopd.experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    metrics = commands.add_parser("build-validation-metrics")
    metrics.add_argument("--config", type=Path, required=True)
    metrics.add_argument("--validation-dir", type=Path, required=True)
    metrics.add_argument("--checkpoint-root", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)

    select = commands.add_parser("select-checkpoint")
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--metrics", type=Path, required=True)
    select.add_argument("--checkpoint-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--forbidden-test-root", type=Path, action="append", default=[])

    plan = commands.add_parser("evaluation-plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--test-data", type=Path, required=True)
    plan.add_argument("--evaluation-config", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--selection-report", type=Path, required=True)
    plan.add_argument("--base-model", required=True)
    plan.add_argument("--base-revision", required=True)
    plan.add_argument("--sft-model", required=True)
    plan.add_argument("--opd-model", required=True)
    plan.add_argument("--teacher-model", required=True)
    plan.add_argument("--teacher-revision", required=True)
    plan.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("final-report")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--selection-report", type=Path, required=True)
    report.add_argument("--evaluation-plan", type=Path, required=True)
    report.add_argument("--training-run-dir", type=Path, required=True)
    report.add_argument("--base-dir", type=Path, required=True)
    report.add_argument("--sft-dir", type=Path, required=True)
    report.add_argument("--opd-dir", type=Path, required=True)
    report.add_argument("--teacher-dir", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-validation-metrics":
        value = build_validation_metrics(
            validation_dir=args.validation_dir,
            checkpoint_root=args.checkpoint_root,
            output_path=args.output,
            formal_config=load_formal_config(args.config),
        )
    elif args.command == "select-checkpoint":
        value = select_validation_checkpoint(
            metrics_path=args.metrics,
            checkpoint_root=args.checkpoint_root,
            formal_config=load_formal_config(args.config),
            output_path=args.output,
            forbidden_test_roots=args.forbidden_test_root,
        )
    elif args.command == "evaluation-plan":
        value = build_evaluation_plan(
            formal_config=load_formal_config(args.config),
            test_data=args.test_data,
            evaluation_config=args.evaluation_config,
            output_root=args.output_root,
            selection_report=args.selection_report,
            models={
                "base": (args.base_model, args.base_revision),
                "sft": (args.sft_model, None),
                "opd": (args.opd_model, None),
                "teacher": (args.teacher_model, args.teacher_revision),
            },
        )
        write_json(args.output, value)
    else:
        value = build_final_report(
            formal_config=load_formal_config(args.config),
            selection_report=args.selection_report,
            evaluation_plan=args.evaluation_plan,
            training_run_dir=args.training_run_dir,
            evaluation_dirs={
                "base": args.base_dir,
                "sft": args.sft_dir,
                "opd": args.opd_dir,
                "teacher": args.teacher_dir,
            },
            output_path=args.output,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
