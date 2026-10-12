"""Reproducible GSM8K snapshotting, splitting, validation, and conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_TRAIN_SIZE = 7_473
GSM8K_TEST_SIZE = 1_319
SPLIT_SEED = 42
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class DataValidationError(ValueError):
    """Raised when source data or generated artifacts violate the contract."""


@dataclass(frozen=True)
class SplitSizes:
    """Sizes of the three subsets derived from the official train split."""

    sft: int
    opd_train: int
    validation: int

    @property
    def total(self) -> int:
        return self.sft + self.opd_train + self.validation

    def as_dict(self) -> dict[str, int]:
        return {
            "sft": self.sft,
            "opd_train": self.opd_train,
            "validation": self.validation,
        }


DEFAULT_SPLIT_SIZES = SplitSizes(sft=3_000, opd_train=3_473, validation=1_000)


def normalize_question(question: str) -> str:
    """Normalize a question for duplicate detection and stable hashing."""
    if not isinstance(question, str) or not question.strip():
        raise DataValidationError("question must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", question)
    return " ".join(normalized.split()).casefold()


def question_hash(question: str) -> str:
    """Return the SHA-256 of a normalized question."""
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def extract_ground_truth(answer: str) -> str:
    """Extract the final GSM8K answer following the last ``####`` marker."""
    if not isinstance(answer, str) or not answer.strip():
        raise DataValidationError("answer must be a non-empty string")
    matches = re.findall(r"####\s*([^\r\n]+)", answer)
    if not matches:
        raise DataValidationError("answer does not contain a '####' final answer")
    ground_truth = matches[-1].strip().replace(",", "")
    if not ground_truth:
        raise DataValidationError("answer has an empty value after '####'")
    return ground_truth


def validate_raw_split(
    records: Sequence[Mapping[str, Any]],
    source_split: str,
    expected_count: int | None = None,
) -> None:
    """Validate the raw GSM8K schema, values, and normalized uniqueness."""
    if expected_count is not None and len(records) != expected_count:
        raise DataValidationError(
            f"{source_split} has {len(records)} rows; expected {expected_count}"
        )

    seen_questions: dict[str, int] = {}
    for index, record in enumerate(records):
        if "question" not in record or "answer" not in record:
            raise DataValidationError(
                f"{source_split}[{index}] must contain question and answer"
            )
        question = record["question"]
        answer = record["answer"]
        if not isinstance(question, str):
            raise DataValidationError(f"{source_split}[{index}].question is not str")
        if not isinstance(answer, str):
            raise DataValidationError(f"{source_split}[{index}].answer is not str")

        digest = question_hash(question)
        extract_ground_truth(answer)
        if digest in seen_questions:
            first = seen_questions[digest]
            raise DataValidationError(
                f"normalized duplicate in {source_split}: rows {first} and {index}"
            )
        seen_questions[digest] = index


def _split_order_key(seed: int, original_index: int) -> bytes:
    payload = f"miniopd-gsm8k:{seed}:{original_index}".encode()
    return hashlib.sha256(payload).digest()


def _assign_train_splits(
    train_count: int,
    seed: int,
    sizes: SplitSizes,
) -> dict[int, str]:
    if sizes.total != train_count:
        raise DataValidationError(
            f"split sizes total {sizes.total}; official train has {train_count} rows"
        )
    if min(sizes.sft, sizes.opd_train, sizes.validation) <= 0:
        raise DataValidationError("all derived split sizes must be positive")

    ordered = sorted(range(train_count), key=lambda index: _split_order_key(seed, index))
    assignments: dict[int, str] = {}
    sft_end = sizes.sft
    opd_end = sft_end + sizes.opd_train
    for position, original_index in enumerate(ordered):
        if position < sft_end:
            split = "sft"
        elif position < opd_end:
            split = "opd_train"
        else:
            split = "validation"
        assignments[original_index] = split
    return assignments


def build_manifest(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    dataset_revision: str,
    code_version: str,
    seed: int = SPLIT_SEED,
    sizes: SplitSizes = DEFAULT_SPLIT_SIZES,
) -> list[dict[str, Any]]:
    """Build an auditable, deterministic manifest for every GSM8K row."""
    if not dataset_revision.strip():
        raise DataValidationError("dataset_revision must be non-empty")
    if not code_version.strip():
        raise DataValidationError("code_version must be non-empty")

    validate_raw_split(train_records, "train", sizes.total)
    validate_raw_split(test_records, "test", len(test_records))
    assignments = _assign_train_splits(len(train_records), seed, sizes)

    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(train_records):
        manifest.append(
            {
                "sample_id": f"gsm8k-train-{index:05d}",
                "source_split": "train",
                "original_index": index,
                "split": assignments[index],
                "question_hash": question_hash(str(record["question"])),
                "dataset_revision": dataset_revision,
                "seed": seed,
                "code_version": code_version,
            }
        )

    for index, record in enumerate(test_records):
        manifest.append(
            {
                "sample_id": f"gsm8k-test-{index:05d}",
                "source_split": "test",
                "original_index": index,
                "split": "test",
                "question_hash": question_hash(str(record["question"])),
                "dataset_revision": dataset_revision,
                "seed": seed,
                "code_version": code_version,
            }
        )

    validate_manifest(manifest, sizes=sizes, test_size=len(test_records))
    return manifest


def validate_manifest(
    manifest: Sequence[Mapping[str, Any]],
    *,
    sizes: SplitSizes = DEFAULT_SPLIT_SIZES,
    test_size: int = GSM8K_TEST_SIZE,
) -> None:
    """Check counts, identity uniqueness, and normalized-question leakage."""
    expected_counts = {**sizes.as_dict(), "test": test_size}
    actual_counts = Counter(str(row.get("split")) for row in manifest)
    if actual_counts != Counter(expected_counts):
        raise DataValidationError(
            f"manifest counts {dict(actual_counts)}; expected {expected_counts}"
        )

    identities: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    hashes: dict[str, str] = {}
    train_indices: set[int] = set()
    test_indices: set[int] = set()

    for row in manifest:
        required = {
            "sample_id",
            "source_split",
            "original_index",
            "split",
            "question_hash",
            "dataset_revision",
            "seed",
            "code_version",
        }
        missing = required.difference(row)
        if missing:
            raise DataValidationError(f"manifest row missing fields: {sorted(missing)}")

        source_split = str(row["source_split"])
        split = str(row["split"])
        index = int(row["original_index"])
        sample_id = str(row["sample_id"])
        digest = str(row["question_hash"])

        if source_split == "train":
            if split not in sizes.as_dict():
                raise DataValidationError(f"train row assigned to invalid split {split}")
            train_indices.add(index)
        elif source_split == "test":
            if split != "test":
                raise DataValidationError("official test row assigned outside test")
            test_indices.add(index)
        else:
            raise DataValidationError(f"unknown source split {source_split}")

        identity = (source_split, index)
        if identity in identities:
            raise DataValidationError(f"duplicate source identity {identity}")
        identities.add(identity)

        if sample_id in sample_ids:
            raise DataValidationError(f"duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)

        if digest in hashes:
            raise DataValidationError(
                f"question leakage between {hashes[digest]} and {sample_id}"
            )
        hashes[digest] = sample_id

        if not str(row["dataset_revision"]).strip():
            raise DataValidationError("empty dataset_revision in manifest")
        if not str(row["code_version"]).strip():
            raise DataValidationError("empty code_version in manifest")

    if train_indices != set(range(sizes.total)):
        raise DataValidationError("manifest does not cover every train index exactly once")
    if test_indices != set(range(test_size)):
        raise DataValidationError("manifest does not cover every test index exactly once")


def manifest_jsonl_bytes(manifest: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize a manifest canonically so repeated builds have identical bytes."""
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in manifest
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_prompt(question: str) -> list[dict[str, str]]:
    """Build the single prompt contract shared by generation and evaluation."""
    if not isinstance(question, str) or not question.strip():
        raise DataValidationError("cannot build a prompt from an empty question")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def _resolve_manifest_record(
    row: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    source = str(row["source_split"])
    records = train_records if source == "train" else test_records
    index = int(row["original_index"])
    if not 0 <= index < len(records):
        raise DataValidationError(f"manifest index out of range: {source}[{index}]")
    record = records[index]
    actual_hash = question_hash(str(record["question"]))
    if actual_hash != row["question_hash"]:
        raise DataValidationError(f"question hash mismatch for {row['sample_id']}")
    return record


def _selected_rows(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    split: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    selected = []
    for row in manifest:
        if row["split"] == split:
            selected.append(
                (row, _resolve_manifest_record(row, train_records, test_records))
            )
    return selected


def build_evaluation_records(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Convert one split into the backend-independent evaluation schema."""
    if split not in {"validation", "test"}:
        raise DataValidationError("evaluation split must be validation or test")
    output = []
    for row, record in _selected_rows(train_records, test_records, manifest, split):
        question = str(record["question"])
        output.append(
            {
                "sample_id": row["sample_id"],
                "data_source": GSM8K_DATASET,
                "split": split,
                "messages": build_prompt(question),
                "question": question,
                "reference_answer": extract_ground_truth(str(record["answer"])),
                "raw_answer": record["answer"],
                "question_hash": row["question_hash"],
                "original_index": row["original_index"],
            }
        )
    return output


def build_teacher_rollout_inputs(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create prompts for teacher generation over the cold-start SFT subset."""
    output = []
    for row, record in _selected_rows(
        train_records, test_records, manifest, "sft"
    ):
        question = str(record["question"])
        output.append(
            {
                "sample_id": row["sample_id"],
                "split": "sft",
                "messages": build_prompt(question),
                "question": question,
                "reference_answer": extract_ground_truth(str(record["answer"])),
                "question_hash": row["question_hash"],
                "original_index": row["original_index"],
            }
        )
    return output


def build_sft_records(
    teacher_inputs: Sequence[Mapping[str, Any]],
    teacher_responses: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Join verified teacher responses with prompts to form SFT messages."""
    expected_ids = {str(item["sample_id"]) for item in teacher_inputs}
    response_ids = set(teacher_responses)
    missing = expected_ids.difference(response_ids)
    unexpected = response_ids.difference(expected_ids)
    if missing or unexpected:
        raise DataValidationError(
            "teacher response IDs do not match inputs: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    output = []
    for item in teacher_inputs:
        sample_id = str(item["sample_id"])
        response = teacher_responses[sample_id]
        if not isinstance(response, str) or not response.strip():
            raise DataValidationError(f"empty teacher response for {sample_id}")
        messages = [dict(message) for message in item["messages"]]
        messages.append({"role": "assistant", "content": response.strip()})
        output.append(
            {
                "sample_id": sample_id,
                "messages": messages,
                "reference_answer": item["reference_answer"],
                "question_hash": item["question_hash"],
                "original_index": item["original_index"],
            }
        )
    return output


def build_verl_opd_records(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert the OPD subset to verl's prompt parquet schema."""
    output = []
    for row, record in _selected_rows(
        train_records, test_records, manifest, "opd_train"
    ):
        question = str(record["question"])
        output.append(
            {
                "data_source": GSM8K_DATASET,
                "prompt": build_prompt(question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": extract_ground_truth(str(record["answer"])),
                },
                "extra_info": {
                    "sample_id": row["sample_id"],
                    "split": "opd_train",
                    "original_index": row["original_index"],
                    "question": question,
                    "question_hash": row["question_hash"],
                    "raw_answer": record["answer"],
                },
            }
        )
    return output


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_datasets() -> Any:
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face datasets is required on the data-preparation machine"
        ) from error
    return datasets


def _require_exact_revision(revision: str) -> str:
    if not _REVISION_RE.fullmatch(revision):
        raise DataValidationError(
            "revision must be a full 40-character Hugging Face commit SHA"
        )
    return revision.lower()


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write(path, (payload + "\n").encode("utf-8"))


def _write_parquet(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = _import_datasets()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".parquet", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        datasets.Dataset.from_list([dict(record) for record in records]).to_parquet(
            temporary_path
        )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def verify_raw_snapshot(snapshot_dir: Path, revision: str) -> dict[str, Any]:
    """Verify immutable raw files against their stored revision and checksums."""
    revision = _require_exact_revision(revision)
    metadata_path = snapshot_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DataValidationError(f"missing raw metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != GSM8K_DATASET:
        raise DataValidationError("raw snapshot dataset ID mismatch")
    if metadata.get("config") != GSM8K_CONFIG:
        raise DataValidationError("raw snapshot config mismatch")
    if metadata.get("revision") != revision:
        raise DataValidationError("raw snapshot revision mismatch")

    expected_counts = {"train": GSM8K_TRAIN_SIZE, "test": GSM8K_TEST_SIZE}
    if metadata.get("split_counts") != expected_counts:
        raise DataValidationError("raw snapshot split counts mismatch")

    for split in ("train", "test"):
        path = snapshot_dir / f"{split}.parquet"
        if not path.is_file():
            raise DataValidationError(f"missing raw file: {path}")
        expected_hash = metadata.get("sha256", {}).get(path.name)
        if not expected_hash or sha256_file(path) != expected_hash:
            raise DataValidationError(f"checksum mismatch for {path}")
    return metadata


def create_raw_snapshot(
    *,
    revision: str,
    raw_root: Path,
    code_version: str,
) -> Path:
    """Download one exact GSM8K revision and save an immutable raw snapshot."""
    revision = _require_exact_revision(revision)
    if not code_version.strip():
        raise DataValidationError("code_version must be non-empty")
    snapshot_dir = raw_root / "gsm8k" / revision
    if snapshot_dir.exists():
        verify_raw_snapshot(snapshot_dir, revision)
        return snapshot_dir

    datasets = _import_datasets()
    dataset = datasets.load_dataset(
        GSM8K_DATASET,
        GSM8K_CONFIG,
        revision=revision,
    )
    train_records = [dict(row) for row in dataset["train"]]
    test_records = [dict(row) for row in dataset["test"]]
    validate_raw_split(train_records, "train", GSM8K_TRAIN_SIZE)
    validate_raw_split(test_records, "test", GSM8K_TEST_SIZE)

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{revision}.", dir=snapshot_dir.parent)
    )
    try:
        train_path = temporary_dir / "train.parquet"
        test_path = temporary_dir / "test.parquet"
        datasets.Dataset.from_list(train_records).to_parquet(train_path)
        datasets.Dataset.from_list(test_records).to_parquet(test_path)
        metadata = {
            "dataset": GSM8K_DATASET,
            "config": GSM8K_CONFIG,
            "revision": revision,
            "split_counts": {
                "train": len(train_records),
                "test": len(test_records),
            },
            "schema": {
                split: {name: str(feature) for name, feature in data.features.items()}
                for split, data in dataset.items()
            },
            "sha256": {
                train_path.name: sha256_file(train_path),
                test_path.name: sha256_file(test_path),
            },
            "datasets_version": datasets.__version__,
            "code_version": code_version,
        }
        (temporary_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, snapshot_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    verify_raw_snapshot(snapshot_dir, revision)
    return snapshot_dir


def load_raw_snapshot(
    snapshot_dir: Path, revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a verified local snapshot without contacting the Hub."""
    verify_raw_snapshot(snapshot_dir, revision)
    datasets = _import_datasets()
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": str(snapshot_dir / "train.parquet"),
            "test": str(snapshot_dir / "test.parquet"),
        },
    )
    train_records = [dict(row) for row in dataset["train"]]
    test_records = [dict(row) for row in dataset["test"]]
    validate_raw_split(train_records, "train", GSM8K_TRAIN_SIZE)
    validate_raw_split(test_records, "test", GSM8K_TEST_SIZE)
    return train_records, test_records


def prepare_gsm8k(
    *,
    snapshot_dir: Path,
    intermediate_root: Path,
    final_root: Path,
    revision: str,
    code_version: str,
    seed: int = SPLIT_SEED,
) -> dict[str, Path]:
    """Create the manifest and all formats available before teacher generation."""
    revision = _require_exact_revision(revision)
    train_records, test_records = load_raw_snapshot(snapshot_dir, revision)
    manifest = build_manifest(
        train_records,
        test_records,
        dataset_revision=revision,
        code_version=code_version,
        seed=seed,
    )

    intermediate_dir = intermediate_root / "gsm8k" / revision
    final_dir = final_root / "gsm8k" / revision
    paths = {
        "manifest": intermediate_dir / "split_manifest.jsonl",
        "manifest_metadata": intermediate_dir / "metadata.json",
        "evaluation_validation": final_dir / "evaluation_validation.parquet",
        "evaluation_test": final_dir / "evaluation_test.parquet",
        "teacher_rollout_input": final_dir / "teacher_rollout_input.parquet",
        "opd_train": final_dir / "opd_train.parquet",
        "final_metadata": final_dir / "metadata.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    manifest_payload = manifest_jsonl_bytes(manifest)
    _atomic_write(paths["manifest"], manifest_payload)
    _write_json(
        paths["manifest_metadata"],
        {
            "dataset": GSM8K_DATASET,
            "revision": revision,
            "seed": seed,
            "code_version": code_version,
            "counts": dict(Counter(row["split"] for row in manifest)),
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        },
    )

    _write_parquet(
        build_evaluation_records(
            train_records, test_records, manifest, split="validation"
        ),
        paths["evaluation_validation"],
    )
    _write_parquet(
        build_evaluation_records(train_records, test_records, manifest, split="test"),
        paths["evaluation_test"],
    )
    _write_parquet(
        build_teacher_rollout_inputs(train_records, test_records, manifest),
        paths["teacher_rollout_input"],
    )
    _write_parquet(
        build_verl_opd_records(train_records, test_records, manifest),
        paths["opd_train"],
    )

    generated_files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if name not in {"final_metadata", "manifest_metadata"}
    }
    _write_json(
        paths["final_metadata"],
        {
            "dataset": GSM8K_DATASET,
            "revision": revision,
            "seed": seed,
            "code_version": code_version,
            "system_prompt": SYSTEM_PROMPT,
            "files": generated_files,
            "note": (
                "SFT messages are created after teacher generation with "
                "build_sft_records; no teacher answers are fabricated here."
            ),
        },
    )
    return paths


def git_commit(project_root: Path) -> str:
    """Return the exact code version used to produce data artifacts."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("the MiniOPD repository needs an initial Git commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("commit data-preparation code before generating artifacts")
    return commit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare one immutable GSM8K revision"
    )
    parser.add_argument("--revision", required=True, help="full 40-character SHA")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="create and verify only the raw snapshot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    code_version = git_commit(args.project_root.resolve())
    data_root = args.data_root.resolve()
    snapshot_dir = create_raw_snapshot(
        revision=args.revision,
        raw_root=data_root / "raw",
        code_version=code_version,
    )
    print(f"verified raw snapshot: {snapshot_dir}")
    if args.download_only:
        return
    outputs = prepare_gsm8k(
        snapshot_dir=snapshot_dir,
        intermediate_root=data_root / "intermediate",
        final_root=data_root / "final",
        revision=args.revision,
        code_version=code_version,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

