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

## Full BF16 benchmark

After the first-100 pre-evaluation passes, verify that the locked student and
teacher tokenizers have the same vocabulary and special-token IDs:

```bash
python -m miniopd.evaluation tokenizer-check \
  --student-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --teacher-config configs/models/qwen2.5-math-1.5b-instruct.yaml \
  --output results/tokenizer_compatibility.json
```

Then run each locked model revision on all 1319 official GSM8K test rows:

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866
INPUT=data/final/gsm8k/$REVISION/evaluation_test.parquet

python -m miniopd.evaluation benchmark \
  --config configs/evaluation/gsm8k_full_bf16_vllm.yaml \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --input "$INPUT" \
  --output-dir results/eval-qwen05b-base-full-bf16-s42

python -m miniopd.evaluation benchmark \
  --config configs/evaluation/gsm8k_full_bf16_vllm.yaml \
  --model-config configs/models/qwen2.5-math-1.5b-instruct.yaml \
  --input "$INPUT" \
  --output-dir results/eval-qwen15bmath-teacher-full-bf16-s42
```

The benchmark verifies the exact Hugging Face model commit, all 1319 ordered
test IDs, writable isolated output storage, free disk space, CUDA/BF16, one-item
generation, and small-batch generation before starting the resumable full run.
It records package versions, GPU identity/topology, rendered prompt samples,
per-example predictions, aggregate metrics, and the manual-review queue. The
summary explicitly identifies the score as this project's own full-test result.

## Cold-start teacher data

Generate one sampled teacher solution for every row in the fixed 3000-example
SFT split. The raw JSONL is append-only and resumable; the filtered JSONL is a
deterministic derivative and never replaces failed generations.

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866

python -m miniopd.generation \
  --config configs/generation/coldstart_teacher_qwen15b.yaml \
  --model-config configs/models/qwen2.5-math-1.5b-instruct.yaml \
  --input "data/final/gsm8k/$REVISION/teacher_rollout_input.parquet" \
  --output-dir results/coldstart-qwen15bmath-s42
```

The output contains all raw generations, the retained `sft_train.jsonl`, filter
statistics, the resolved configuration, and a deterministic manual-review queue.

## Cold-start SFT

First inspect the actual token distribution. `max_length: auto` resolves to the
observed maximum rounded up to 64, so training does not silently truncate a
teacher trajectory.

```bash
python -m miniopd.training lengths \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --train-input results/coldstart-qwen15bmath-s42/sft_train.jsonl \
  --output results/coldstart-qwen15bmath-s42/token_lengths.json
```

Run the required 32-row, five-step smoke test before the full run:

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866

python -m miniopd.training sft --smoke \
  --config configs/sft/qwen05b_coldstart_full.yaml \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --train-input results/coldstart-qwen15bmath-s42/sft_train.jsonl \
  --validation-input "data/final/gsm8k/$REVISION/evaluation_validation.parquet" \
  --output-dir results/sft-qwen05b-coldstart-smoke-s42
```

The smoke test trains to step 2, saves a checkpoint, resumes to step 5, reloads
the exported model independently, and verifies generation. Remove `--smoke` and
use a new output directory for the full one-epoch run. Prompt tokens are masked
with `-100`; only teacher assistant tokens contribute to the SFT loss. Checkpoint
selection uses validation loss, never the official test split.

After the full SFT run, evaluate the independently loadable `final_model` with
the same full-test greedy pipeline used for the base model:

```bash
python -m miniopd.evaluation run \
  --config configs/evaluation/gsm8k_greedy.yaml \
  --backend vllm \
  --model results/sft-qwen05b-coldstart-s42/final_model \
  --input "data/final/gsm8k/$REVISION/evaluation_test.parquet" \
  --output results/eval-qwen05b-sft-full-bf16-s42/predictions.jsonl \
  --summary results/eval-qwen05b-sft-full-bf16-s42/summary.json
```

## Sampled-token OPD algorithm tests

Before integrating Qwen and verl, the core math is tested with hand-checkable CPU
tensors and tiny PyTorch modules:

```bash
make opd-test
```

The tests cover next-token logits shift, response/EOS masking, sampled-token k1
reward signs and clamps, stop-gradient and frozen-teacher behavior, Top-k overlap,
entropy gap, and actor-to-rollout weight synchronization. The reference mask
includes the first response EOS and excludes prompt, padding, and all positions
after that EOS. `opd_tensor_records` records shape, dtype, device, gradient state,
and semantics for every tensor that must be checked again during verl integration.

This in-process sync test does not claim to validate FSDP-to-vLLM transfer; that
cross-process GPU check belongs to the next verl integration stage.
