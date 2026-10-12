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
