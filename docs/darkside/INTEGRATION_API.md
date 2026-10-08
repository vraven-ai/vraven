# Integration API

## Merge into the existing workspace

Perform a preflight:

```bash
python /path/to/VRAVEN_DarkSide_Extensions/scripts/merge_darkside.py \
  /path/to/VRAVEN_DarkSide_Extensions \
  --target /path/to/VRAVEN \
  --dry-run
```

Apply the merge:

```bash
python /path/to/VRAVEN_DarkSide_Extensions/scripts/merge_darkside.py \
  /path/to/VRAVEN_DarkSide_Extensions \
  --target /path/to/VRAVEN
```

The script copies only `src/vraven/darkside`, `test_darkside*.py` and `darkside_*.py` examples. It does not replace the existing VRAVEN package.

## Live model intervention session

```python
from vraven.darkside import DarkSideSession, InputFeatureGroup

session = DarkSideSession(model, sample, target=1)

circuit = session.decision_circuitry(
    layer_names=("encoder.0", "encoder.2"),
    group_size=8,
    relative_score_tolerance=0.05,
)

shortcut = session.shortcut_detection(
    [InputFeatureGroup((12,), label="LOLBIN indicator")]
)
```

## Build an aligned dataset

```python
from vraven.darkside import TraceDataset

traces = [
    vraven.explain(model, sample, target=1, mode="gradient").trace
    for sample in samples
]

dataset = TraceDataset.from_traces(
    traces,
    true_labels=labels,
    sample_ids=sample_hashes,
)
```

Batching is supported when each trace represents one logical sample or when the caller deliberately aggregates each trace to one row. Paired analyses require exact sample-ID equality and ordering.

## Cohort analyses

```python
from vraven.darkside import (
    analyse_failure_anatomy,
    analyse_representation_drift,
    audit_unused_capacity,
)

failure = analyse_failure_anatomy(dataset, failure="fp", reference="tp")
capacity = audit_unused_capacity(dataset)
drift = analyse_representation_drift(checkpoint_a, checkpoint_b)
```

## Repeated-circuit multiplicity

```python
from vraven.darkside import analyse_circuit_multiplicity

multiplicity = analyse_circuit_multiplicity(
    repeated_binary_circuit_masks,
    component_labels=aligned_component_names,
    stable_threshold=0.80,
)
```

## Visual modes

```python
result.figure(mode="interactive_dark").show()
result.figure(mode="publication_light").write_html("figure.html")
```

Static SVG, PDF or PNG export can use VRAVEN's existing export helper or Plotly Kaleido after the result figure has been created.

## Optional top-level export

No edit to `vraven/__init__.py` is required. The stable import is:

```python
from vraven.darkside import DarkSideSession, TraceDataset, recommend_features
```
