"""Command line entry points for model evaluation and offline rescoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from miniopd.evaluation.io import load_evaluation_examples, write_json
from miniopd.evaluation.pipeline import run_evaluation, score_jsonl
from miniopd.evaluation.schema import GenerationConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m miniopd.evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate and score a model")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--summary", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--model-revision")
    run.add_argument("--config", type=Path)
    run.add_argument("--backend", choices=("transformers", "vllm"))
    run.add_argument("--batch-size", type=int)
    run.add_argument("--max-new-tokens", type=int)
    run.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"))
    run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model repository code (Transformers only)",
    )

    score = subparsers.add_parser("score", help="deterministically rescore JSONL")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--summary", type=Path, required=True)

    preeval = subparsers.add_parser(
        "preeval", help="run the fixed GSM8K test first-100 GPU check"
    )
    preeval.add_argument("--input", type=Path, required=True)
    preeval.add_argument("--output-dir", type=Path, required=True)
    preeval.add_argument("--model", required=True)
    preeval.add_argument("--config", type=Path, required=True)
    preeval.add_argument("--trust-remote-code", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="run the full official GSM8K BF16 benchmark"
    )
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--model-config", type=Path, required=True)
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--trust-remote-code", action="store_true")

    compatibility = subparsers.add_parser(
        "tokenizer-check", help="compare locked student and teacher tokenizers"
    )
    compatibility.add_argument("--student-config", type=Path, required=True)
    compatibility.add_argument("--teacher-config", type=Path, required=True)
    compatibility.add_argument("--output", type=Path, required=True)
    compatibility.add_argument("--trust-remote-code", action="store_true")
    return parser


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required when --config is used") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("evaluation config must be a YAML mapping")
    return dict(value)


def _run(args: argparse.Namespace) -> None:
    settings = _load_yaml(args.config)
    allowed = {
        "backend",
        "temperature",
        "top_p",
        "do_sample",
        "max_new_tokens",
        "batch_size",
        "dtype",
    }
    unexpected = set(settings).difference(allowed)
    if unexpected:
        raise ValueError(f"unknown evaluation config keys: {sorted(unexpected)}")
    backend_name = args.backend or settings.get("backend")
    if backend_name not in {"transformers", "vllm"}:
        raise ValueError("select transformers or vllm with --backend or --config")
    dtype = args.dtype or str(settings.get("dtype", "auto"))
    config = GenerationConfig(
        temperature=float(settings.get("temperature", 0)),
        top_p=float(settings.get("top_p", 1)),
        do_sample=bool(settings.get("do_sample", False)),
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(settings.get("max_new_tokens", 512))
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else int(settings.get("batch_size", 8))
        ),
        dtype=dtype,
    )
    if backend_name == "transformers":
        from miniopd.evaluation.backends import TransformersBackend

        backend = TransformersBackend(
            args.model,
            revision=args.model_revision,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
    else:
        from miniopd.evaluation.backends import VLLMBackend

        backend = VLLMBackend(
            args.model,
            revision=args.model_revision,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    summary = run_evaluation(
        load_evaluation_examples(args.input),
        backend,
        args.output,
        config=config,
    )
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "score":
        score_jsonl(args.input, args.output, args.summary)
    elif args.command == "preeval":
        from miniopd.evaluation.preeval import (
            PreEvaluationConfig,
            print_summary,
            run_preevaluation,
        )

        summary = run_preevaluation(
            input_path=args.input,
            output_dir=args.output_dir,
            model_id=args.model,
            config=PreEvaluationConfig.from_mapping(_load_yaml(args.config)),
            trust_remote_code=args.trust_remote_code,
        )
        print_summary(summary)
    elif args.command == "benchmark":
        from miniopd.evaluation.benchmark import (
            BenchmarkConfig,
            ModelSpec,
            print_json,
            run_benchmark,
        )

        summary = run_benchmark(
            input_path=args.input,
            output_dir=args.output_dir,
            model=ModelSpec.from_mapping(_load_yaml(args.model_config)),
            config=BenchmarkConfig.from_mapping(_load_yaml(args.config)),
            trust_remote_code=args.trust_remote_code,
        )
        print_json(summary)
    else:
        from miniopd.evaluation.benchmark import (
            ModelSpec,
            check_tokenizer_compatibility,
            print_json,
        )

        report = check_tokenizer_compatibility(
            ModelSpec.from_mapping(_load_yaml(args.student_config)),
            ModelSpec.from_mapping(_load_yaml(args.teacher_config)),
            trust_remote_code=args.trust_remote_code,
        )
        write_json(args.output, report)
        print_json(report)
