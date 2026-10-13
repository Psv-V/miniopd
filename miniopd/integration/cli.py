"""CLI for Stage 12 verl integration."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from miniopd.evaluation.io import append_jsonl, write_json
from miniopd.integration.e2e_smoke import check_e2e_artifacts, run_e2e_smoke
from miniopd.integration.verl_opd import (
    IntegrationError,
    build_verl_command,
    check_artifacts,
    launch,
    load_yaml,
    validate_checkpoint,
    validate_recipe,
    validate_runtime,
    validate_verl_checkout,
    validate_verl_data,
    write_resolved_yaml,
)


def _record_run_status(run_dir: Path, record: dict) -> None:
    write_json(run_dir / "run_status.json", record)
    append_jsonl(run_dir / "run_history.jsonl", [record])


def _latest_checkpoint_step(run_dir: Path) -> int:
    steps = []
    for path in (run_dir / "checkpoints").glob("global_step_*"):
        try:
            steps.append(int(path.name.removeprefix("global_step_")))
        except ValueError:
            continue
    return max(steps, default=0)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verl-lock", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miniopd.integration")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _shared(preflight)
    preflight.add_argument("--runtime", action="store_true")
    run = commands.add_parser("launch")
    _shared(run)
    check = commands.add_parser("check-artifacts")
    check.add_argument("--mode", choices=("smoke", "full"), required=True)
    check.add_argument("--run-dir", type=Path, required=True)
    e2e = commands.add_parser("e2e-smoke")
    e2e.add_argument("--config", type=Path, required=True)
    e2e.add_argument("--smoke-config", type=Path, required=True)
    e2e.add_argument("--verl-lock", type=Path, required=True)
    e2e.add_argument("--verl-root", type=Path, required=True)
    e2e.add_argument("--student-checkpoint", type=Path, required=True)
    e2e.add_argument("--teacher-model", type=Path, required=True)
    e2e.add_argument("--train-data", type=Path, required=True)
    e2e.add_argument("--validation-data", type=Path, required=True)
    e2e.add_argument("--manifest", type=Path, required=True)
    e2e.add_argument("--run-dir", type=Path, required=True)
    check_e2e = commands.add_parser("check-e2e")
    check_e2e.add_argument("--smoke-config", type=Path, required=True)
    check_e2e.add_argument("--run-dir", type=Path, required=True)
    return parser


def _prepare(args: argparse.Namespace, runtime: bool) -> tuple[list[str], dict]:
    config = load_yaml(args.config)
    validate_recipe(config)
    lock = load_yaml(args.verl_lock)
    report = {
        "mode": args.mode,
        "recipe": "passed",
        "verl_commit": validate_verl_checkout(args.verl_root, lock, args.mode),
        "checkpoint": validate_checkpoint(
            args.student_checkpoint, role="student checkpoint"
        ),
        "teacher_model": validate_checkpoint(args.teacher_model, role="teacher model"),
        "data": validate_verl_data(
            args.train_data,
            args.validation_data,
            args.manifest,
            train_count=int(config["data"]["train_count"]),
            validation_count=int(config["data"]["validation_count"]),
        ),
    }
    if runtime:
        report["runtime"] = validate_runtime(
            args.student_checkpoint,
            args.teacher_model,
            int(config["resources"]["expected_total_gpus"]),
        )
    command = build_verl_command(
        config,
        mode=args.mode,
        student_checkpoint=args.student_checkpoint,
        teacher_model=args.teacher_model,
        train_data=args.train_data,
        validation_data=args.validation_data,
        run_dir=args.run_dir,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "preflight.json", report)
    write_resolved_yaml(args.run_dir / "resolved_config.yaml", config)
    write_json(args.run_dir / "launch_command.json", {"argv": command})
    return command, report


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "check-artifacts":
        report = check_artifacts(args.run_dir, args.mode)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "check-e2e":
        report = check_e2e_artifacts(args.run_dir, load_yaml(args.smoke_config))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "e2e-smoke":
        report = run_e2e_smoke(
            opd_config_path=args.config,
            smoke_config_path=args.smoke_config,
            verl_lock_path=args.verl_lock,
            verl_root=args.verl_root,
            student_checkpoint=args.student_checkpoint,
            teacher_model=args.teacher_model,
            train_source=args.train_data,
            validation_source=args.validation_data,
            manifest=args.manifest,
            run_dir=args.run_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    command, report = _prepare(args, runtime=args.command == "launch" or args.runtime)
    print(json.dumps(report, indent=2, sort_keys=True))
    separator = " \\" + "\n  "
    print("Launch command:\n" + separator.join(command))
    if args.command == "launch":
        started = time.time()
        attempt = datetime.now(timezone.utc).isoformat()
        base_status = {
            "attempt": attempt,
            "mode": args.mode,
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "visible_gpu_count": report.get("runtime", {}).get("cuda_device_count"),
        }
        _record_run_status(args.run_dir, {**base_status, "status": "running"})
        try:
            exit_code = launch(
                command,
                args.run_dir,
                cwd=args.verl_root,
                sync_version_offset=(
                    _latest_checkpoint_step(args.run_dir) if args.mode == "full" else 0
                ),
            )
            if exit_code:
                _record_run_status(
                    args.run_dir,
                    {
                        **base_status,
                        "status": "failed",
                        "exit_code": exit_code,
                        "wall_seconds": time.time() - started,
                    },
                )
                raise SystemExit(exit_code)
            acceptance = check_artifacts(args.run_dir, args.mode)
            if not acceptance["passed"]:
                raise IntegrationError(
                    "training exited but Stage 12 artifacts are incomplete"
                )
        except BaseException as error:
            if not isinstance(error, SystemExit):
                _record_run_status(
                    args.run_dir,
                    {
                        **base_status,
                        "status": "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "wall_seconds": time.time() - started,
                    },
                )
            raise
        _record_run_status(
            args.run_dir,
            {
                **base_status,
                "status": "complete",
                "exit_code": 0,
                "wall_seconds": time.time() - started,
            },
        )
