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
  --model-path /path/to/verified/Qwen2.5-Math-1.5B-Instruct \
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

## verl sampled-token OPD integration

MiniOPD keeps only the experiment contract, preflight checks, and acceptance
logic. The trainer remains the official upstream verl implementation; no verl
source is copied into this repository. The current candidate commit is recorded
in `configs/verl/upstream_candidate.yaml` and is deliberately marked
`pending_gpu_smoke`. Full training is refused until that exact commit passes the
GPU smoke test and the lock is changed to `status: verified` with the same SHA in
`verified_commit`.

The data-preparation command creates both `opd_train.parquet` and
`opd_validation.parquet` from the same locked manifest.

On the two-GPU machine, clone and check out the exact verl candidate, download
the teacher model at its locked revision, and point `STUDENT` at the exported
cold-start SFT `final_model`. GPU 0 owns the student actor plus its vLLM rollout;
GPU 1 owns the frozen teacher vLLM scorer.

Run the one-batch acceptance test first:

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866
CUDA_VISIBLE_DEVICES=0,1 scripts/run_opd_smoke.sh \
  --config configs/opd/qwen05b_sampled_k1.yaml \
  --verl-lock configs/verl/upstream_candidate.yaml \
  --verl-root /path/to/verl \
  --student-checkpoint /path/to/sft/final_model \
  --teacher-model /path/to/Qwen2.5-Math-1.5B-Instruct \
  --train-data "data/final/gsm8k/$REVISION/opd_train.parquet" \
  --validation-data "data/final/gsm8k/$REVISION/opd_validation.parquet" \
  --manifest "data/intermediate/gsm8k/$REVISION/split_manifest.jsonl" \
  --run-dir results/opd-qwen05b-sampled-k1-smoke-s42
```

Preflight requires exactly two visible BF16 GPUs, complete local model files,
identical student/teacher vocabularies and special-token IDs, exact data counts
and manifest membership, and the exact verl Git commit. The smoke run performs
one optimizer step and must leave evidence for rollout, teacher scoring,
backward/update, checkpoint saving, rollout dumping, and both initial and
post-update actor-to-rollout synchronization. Confirm or rerun its audit with:

```bash
python -m miniopd.integration check-artifacts --mode smoke \
  --run-dir results/opd-qwen05b-sampled-k1-smoke-s42
```

Only after that report passes should the verl lock be marked verified and the
same arguments be passed to `scripts/run_opd_train.sh`. Full mode uses all 3473
OPD prompts for one epoch; it never consumes the SFT, validation, or test rows as
training data.

## OPD training diagnostics

The OPD recipe contains a separate `diagnostics` section. Every-step metrics use
already available reward, response-mask, loss, timing, correctness, and memory
values, so they require no additional model forward. They are appended to
`metrics.jsonl`.

Every 50 steps, a fixed four-example batch is eligible for the heavier
diagnostic path. It computes student/teacher Top-16 overlap, overlap-token
advantage, overlap probability mass, exact full-vocabulary entropy, absolute
entropy gap, and front/middle/back response summaries. These values are appended
to `diagnostics.jsonl`; `diagnostic_fixed_batch.json` locks the selected sample
IDs so results remain comparable across checkpoints.

All periodic tensor calculations run under `torch.no_grad()`, return detached
Python values, and reuse the same response mask as sampled-token OPD. Prompt,
padding, and post-EOS positions therefore cannot enter a metric. The diagnostic
record also reports its wall-time overhead and whether it remains below the
configured 10% budget. These diagnostics are observational only and are never
added to the policy or distillation loss.

The diagnostic primitives and artifact auditor are implemented locally, but
the exact callback that exports live verl tensors is intentionally not declared
complete before the candidate commit passes GPU smoke. Until that adapter emits
`metrics.jsonl`, `diagnostics.jsonl`, and `tensor_audit.jsonl`, the end-to-end
audit fails instead of silently accepting incomplete observability. The same
GPU adapter must persist an explicit per-sample `truncated` flag in validation
dumps; checkpoint selection refuses to infer missing truncation evidence.

## End-to-end OPD smoke test

The end-to-end smoke test uses a deterministic 32-row training subset and an
independent eight-row validation subset. It starts a fresh process for two
optimizer steps, requires `global_step_2`, then starts a second process with
`resume_mode=resume_path` and continues to step five. The final step runs
validation and saves `global_step_5`.

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866
CUDA_VISIBLE_DEVICES=0,1 scripts/run_opd_e2e_smoke.sh \
  --config configs/opd/qwen05b_sampled_k1.yaml \
  --smoke-config configs/opd/e2e_smoke.yaml \
  --verl-lock configs/verl/upstream_candidate.yaml \
  --verl-root /path/to/verl \
  --student-checkpoint /path/to/sft/final_model \
  --teacher-model /path/to/Qwen2.5-Math-1.5B-Instruct \
  --train-data "data/final/gsm8k/$REVISION/opd_train.parquet" \
  --validation-data "data/final/gsm8k/$REVISION/opd_validation.parquet" \
  --manifest "data/intermediate/gsm8k/$REVISION/split_manifest.jsonl" \
  --run-dir results/opd-qwen05b-e2e-smoke-s42
```

Use a new run directory for every attempt. The command exits nonzero if either
process fails, the explicit resume is not observed, a checkpoint or validation
artifact is missing, actor-to-rollout synchronization is incomplete, tensor
shift/mask/no-grad evidence is absent, diagnostics are missing, or a logged
numeric value contains NaN/Inf. The complete audit is written to
`e2e_artifact_report.json` and can be rerun without training:

```bash
python -m miniopd.integration check-e2e \
  --smoke-config configs/opd/e2e_smoke.yaml \
  --run-dir results/opd-qwen05b-e2e-smoke-s42
```

## Formal sampled-token OPD and final report

After the end-to-end smoke test passes, run the existing full launcher with a
new formal run directory. Full mode consumes all 3473 OPD prompts for one epoch,
validates and checkpoints every 20 steps, logs eight validation generations,
and resumes automatically after an interruption. Every attempt is retained in
`run_history.jsonl`; the latest state is in `run_status.json`.

```bash
REVISION=740312add88f781978c0658806c59bc2815b9866
CUDA_VISIBLE_DEVICES=0,1 scripts/run_opd_train.sh \
  --config configs/opd/qwen05b_sampled_k1.yaml \
  --verl-lock configs/verl/upstream_candidate.yaml \
  --verl-root /path/to/verl \
  --student-checkpoint /path/to/sft/final_model \
  --teacher-model /path/to/Qwen2.5-Math-1.5B-Instruct \
  --train-data "data/final/gsm8k/$REVISION/opd_train.parquet" \
  --validation-data "data/final/gsm8k/$REVISION/opd_validation.parquet" \
  --manifest "data/intermediate/gsm8k/$REVISION/split_manifest.jsonl" \
  --run-dir results/opd-qwen05b-sampled-k1-s42
```

Select a checkpoint before any formal test prediction exists. Selection uses
validation exact match only, followed by lower format-error ratio, lower
truncation ratio, and earlier step as deterministic tie-breakers:

```bash
python -m miniopd.experiments build-validation-metrics \
  --config configs/experiments/formal_sampled_opd.yaml \
  --validation-dir results/opd-qwen05b-sampled-k1-s42/validation \
  --checkpoint-root results/opd-qwen05b-sampled-k1-s42/checkpoints \
  --output results/opd-qwen05b-sampled-k1-s42/validation_metrics.jsonl

python -m miniopd.experiments select-checkpoint \
  --config configs/experiments/formal_sampled_opd.yaml \
  --metrics results/opd-qwen05b-sampled-k1-s42/validation_metrics.jsonl \
  --checkpoint-root results/opd-qwen05b-sampled-k1-s42/checkpoints \
  --forbidden-test-root results/final-gsm8k \
  --output results/opd-qwen05b-sampled-k1-s42/checkpoint_selection.json
```

After exporting the selected OPD checkpoint to a loadable model directory,
create the four-model evaluation plan. Base, SFT, OPD, and teacher all use the
same 1319 test rows, greedy decoding, BF16, and 512 output tokens:

```bash
python -m miniopd.experiments evaluation-plan \
  --config configs/experiments/formal_sampled_opd.yaml \
  --test-data "data/final/gsm8k/$REVISION/evaluation_test.parquet" \
  --evaluation-config configs/evaluation/gsm8k_greedy.yaml \
  --selection-report results/opd-qwen05b-sampled-k1-s42/checkpoint_selection.json \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --base-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --sft-model /path/to/sft/final_model \
  --opd-model /path/to/selected-opd/final_model \
  --teacher-model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --teacher-revision aafeb0fc6f22cbf0eaeed126eff8be45b0360a35 \
  --output-root results/final-gsm8k \
  --output results/final-gsm8k/evaluation_plan.json
```

Run the four saved commands, then build the final artifact-derived report:

```bash
python -m miniopd.experiments final-report \
  --config configs/experiments/formal_sampled_opd.yaml \
  --selection-report results/opd-qwen05b-sampled-k1-s42/checkpoint_selection.json \
  --evaluation-plan results/final-gsm8k/evaluation_plan.json \
  --training-run-dir results/opd-qwen05b-sampled-k1-s42 \
  --base-dir results/final-gsm8k/base \
  --sft-dir results/final-gsm8k/sft \
  --opd-dir results/final-gsm8k/opd \
  --teacher-dir results/final-gsm8k/teacher \
  --output reports/final_results.json
```

The validation builder rescans every saved response with MiniOPD's shared answer
parser and requires an explicit per-sample truncation flag. It will not silently
treat missing truncation evidence as zero. The final report recalculates each
score from all 1319 saved predictions, verifies both the saved evaluation plan
and shared decoding protocol, and computes SFT gain, additional OPD gain, gap
recovery, GPU hours, peak memory, three training throughputs, checkpoint size,
and four-model inference speed. Failed and interrupted attempts remain visible
in the embedded run history.
