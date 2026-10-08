# Developing VRAVEN from source

## What this repository contains

This workspace implements the new VRAVEN research direction:

```text
Trained model
    → compiler-level computation capture
    → VRAVEN Mechanism IR
    → internal concept discovery
    → cross-layer concept evolution
    → graph-compiled causal mechanism
    → executable decision programme
    → contrastive explanation
    → counterfactual mechanism proof
    → validity and dark-matter certificate
```

## Develop from source

```bash
git clone https://github.com/vraven-ai/vraven.git
cd vraven
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[analysis,dev]"
vraven --version
vraven doctor
```

Run validation:

```bash
PYTHONPATH=src:. pytest -q
bash scripts/build_package.sh
```

## Run the supplied Kronodroid example

```bash
PYTHONPATH=src:. python examples/kronodroid/decompile_cnn1d.py --profile quick
```

Open:

```text
vraven_outputs/compiler_decompiler/kronodroid_cnn1d_decision_proof/vraven_decision_proof.html
```

## Run the built-in demonstration

```bash
vraven decompile-demo \
  --profile quick \
  --report-format html \
  --output vraven_decompile_demo.html
```

## Use an existing state-dictionary model

A `.pt` state dictionary contains weights, not an arbitrary Python architecture. Supply a model factory:

```bash
vraven reverse model.pt \
  --model-factory my_project.models:Classifier \
  --factory-kwargs '{"input_dim":476,"num_classes":2}' \
  --reference reference.npy \
  --labels labels.npy \
  --target 1 \
  --class-names Benign,Malware \
  --profile standard \
  --report-format html \
  --output vraven_decision_proof.html
```

## Read first

```text
README.md
docs/decompiler/ARCHITECTURE.md
docs/decompiler/IMPLEMENTATION_GAP_CLOSURE.md
docs/decompiler/MATHEMATICAL_FOUNDATIONS.md
docs/decompiler/RESEARCH_SOURCE_INDEX.md
docs/decompiler/LIMITATIONS.md
docs/IMPLEMENTATION_STATUS.md
```

## Research status

This release demonstrates an implemented compiler-assisted causal-decompilation workflow. It does not claim universal model reversal, unique semantic recovery, globally minimal programmes or domain-valid counterfactuals without external validation.
