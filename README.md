# MiniOPD

Low-cost on-policy distillation for small-model mathematical reasoning.

## Configuration

- Experiment configurations live under `configs/` and use YAML.
- Configurations are grouped by workflow only when that workflow is implemented,
  such as `evaluation/`, `sft/`, and `opd/`.
- Commands accept a configuration through `--config <path>`.
- Each run must save the fully resolved configuration used for that run.
- Training reuses verl and Hydra/OmegaConf configuration facilities instead of
  introducing a separate configuration framework.

## Outputs and logging

Each run writes to `results/<run_name>/`:

- `resolved_config.yaml`: the effective configuration;
- `run.log`: human-readable progress, warnings, and errors;
- `metrics.jsonl`: machine-readable metrics, one JSON object per line.

Logs are shown in the terminal and saved to disk. Training metrics should reuse
verl's existing tracking interfaces; MiniOPD does not duplicate its training
logging system.

## Artifact naming

Use one run name across checkpoints and results:

```text
<stage>-<model>-<method>-s<seed>
```

Examples:

```text
sft-qwen05b-coldstart-s42
opd-qwen05b-sampled-k1-s42
eval-qwen05b-base-s42
```

Append `-run02`, `-run03`, and so on only when repeating the same configuration.

- `checkpoints/<run_name>/` stores model and optimizer state.
- `results/<run_name>/` stores raw metrics, predictions, logs, and the resolved
  configuration.
- `reports/` stores curated, human-readable analysis that refers to the relevant
  run name.

Large checkpoints, local data, and raw results are ignored by Git. Curated
reports and the code or configuration needed to reproduce them are versioned.

## GSM8K evaluation

Formal evaluation uses greedy decoding from
`configs/evaluation/gsm8k_greedy.yaml`. Transformers and vLLM write the same
resumable JSONL schema and use the same answer parser and metric aggregation.

```bash
python -m miniopd.evaluation run \
  --config configs/evaluation/gsm8k_greedy.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input data/final/gsm8k/<revision>/evaluation_test.parquet \
  --output results/<run_name>/predictions.jsonl \
  --summary results/<run_name>/summary.json
```

The command safely skips sample IDs already present in `predictions.jsonl`.
Resuming with a different model, backend, or generation configuration is
rejected. Saved predictions can be rescored without model inference:

```bash
python -m miniopd.evaluation score \
  --input results/<run_name>/predictions.jsonl \
  --output results/<run_name>/rescored.jsonl \
  --summary results/<run_name>/rescored_summary.json
```

## GPU first-100 pre-evaluation

Before training, run the student and teacher separately on the first 100 rows of
the official GSM8K test split. The fixed configuration enforces BF16, greedy
decoding, and `max_new_tokens=512`.

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866
INPUT=data/final/gsm8k/$REVISION/evaluation_test.parquet

python -m miniopd.evaluation preeval \
  --config configs/evaluation/gsm8k_preeval_100_vllm.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input "$INPUT" \
  --output-dir results/preeval-qwen05b-base-greedy100-s42

python -m miniopd.evaluation preeval \
  --config configs/evaluation/gsm8k_preeval_100_vllm.yaml \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --input "$INPUT" \
  --output-dir results/preeval-qwen15bmath-teacher-greedy100-s42
```

Each directory contains `predictions.jsonl`, `summary.json`, the resolved
configuration, CUDA/package environment metadata, all 100 tokenizer-rendered
prompts, and a deterministic `review_queue.json`. The review queue includes
random correct and wrong examples plus every format error and truncated output.
These runs are labeled as pre-evaluations and are not official full-test scores.
