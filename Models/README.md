# VRAVEN Model Artefacts

## Current suites

| Directory | Status | Purpose |
|---|---|---|
| [`publication_suite_5epoch/`](publication_suite_5epoch/) | Complete: 26/26 final checkpoints | Primary cross-domain VRAVEN functional suite; one seed, at most five epochs, 94% validation-accuracy stop |
| `publication_suite/` | Interrupted by request: 19/130 final checkpoints preserved | Earlier five-seed run; not the primary suite |
| `_smoke_publication_suite_all/` | Complete smoke test: 26/26 paths | Two-batch architecture, MPS, saving and reconstruction test |
| `_smoke_publication_suite/` | Complete smoke test: WDBC only | Initial end-to-end trainer validation |

Use `publication_suite_5epoch/` for the next VRAVEN analyses.
The `CURRENT` shortcut resolves to that suite.

## Reproduce the five-epoch suite

From the repository root:

```bash
/opt/homebrew/bin/python3.10 scripts/train_publication_model_suite.py \
  --device mps \
  --seeds 11 \
  --max-epochs 5 \
  --patience 5 \
  --target-validation-accuracy 0.94 \
  --output-dir Models/publication_suite_5epoch
```

The command is resumable and skips complete artefacts unless `--force` is supplied.

## Artefact contract

Each final checkpoint directory contains:

- `model.pt`: weights and reconstructable architecture declaration;
- `meta.json`: device, environment, model, dataset and input-domain metadata;
- `metrics.json`: validation-selected held-out test measurements;
- `history.json`: epoch history;
- `preprocessing.json`: train-only fitted preprocessing state;
- `split_indices.npz`: exact paired train/validation/test identifiers;
- `reference.npz`: stratified evaluation and calibration material for VRAVEN;
- `RUN_COMPLETE.json`: hashes and completion status;
- `checkpoints/`: lightweight epoch states for model-evolution analysis.

## Scientific boundary

The five-epoch suite tests model and VRAVEN coverage, not architecture-optimal performance. Models that fail the recorded macro-F1 quality gate must not support publication-level accuracy, robustness or cross-architecture claims. The test metric is evaluated once after selecting the best epoch by validation macro-F1. The 94% stopping rule uses validation accuracy and can be optimistic on imbalanced datasets, so macro-F1 remains the checkpoint-selection and quality-gate metric.
