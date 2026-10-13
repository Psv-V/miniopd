"""Command line entry point for cold-start teacher generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from miniopd.generation.coldstart import ColdStartConfig, run_coldstart_generation


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load generation config") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("generation config must be a YAML mapping")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miniopd.generation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    model = _load_yaml(args.model_config)
    if model.get("role") != "teacher":
        raise ValueError("cold-start generation requires a teacher model config")
    summary = run_coldstart_generation(
        input_path=args.input,
        output_dir=args.output_dir,
        model_id=str(model["model_id"]),
        model_revision=str(model["revision"]),
        config=ColdStartConfig.from_mapping(_load_yaml(args.config)),
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
