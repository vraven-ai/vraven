# Extension architecture

VRAVEN DarkSide is an isolated package beneath the existing VRAVEN namespace.

```text
Existing VRAVEN model adapter and trace engine
                    │
        ┌───────────┴───────────┐
        │                       │
 Live model path          Aligned trace path
        │                       │
 DarkSideSession          TraceDataset
 InterventionEngine       sample IDs and cohorts
        │                       │
 ┌──────┼─────────┐       ┌─────┼─────────────────────────────┐
 │      │         │       │     │      │      │      │       │
Circuit Shortcut Synergy Drift Failure Stability Capacity Rerouting
 │                 │       │                         │
Mediation       Multiplicity and transfer       Cost and anomaly
        └───────────┬───────────┘
                    │
              AnalysisResult
                    │
        ┌───────────┴───────────┐
        │                       │
 interactive_dark       publication_light
```

## Live model path

The live model path uses temporary PyTorch hooks and explicit activation interventions. Every intervention runs in deterministic evaluation mode and restores the training flags of every module afterwards. Hooks are removed in `finally` blocks.

```text
DarkSideSession
    ├── decision_circuitry()
    ├── shortcut_detection()
    ├── causal_synergy()
    └── activation_mediation()
```

## Aligned trace path

`TraceDataset` converts multiple VRAVEN traces into sample-aligned matrices. Exact ordered sample IDs are required for paired analyses. The constructor rejects duplicate IDs, shape mismatch and non-finite measurements.

```text
TraceDataset
    ├── representations[layer]      shape: samples × sketch width
    ├── unit_influences[layer]      shape: samples × units
    ├── unit_energies[layer]        shape: samples × units
    ├── activation_rms[layer]       shape: samples
    ├── timing_ms[layer]            shape: samples
    └── fingerprints                shape: samples × fingerprint width
```

## Repeated discovery path

Several analyses operate on repeated outputs rather than model hooks:

```text
Repeated circuit masks  -> circuit multiplicity
Repeated attributions   -> explanation uncertainty
Aligned method vectors  -> method disagreement
Aligned effect vectors  -> mechanism transfer
```

This separation makes variability and non-uniqueness explicit rather than hiding them inside one selected explanation.

## Result contract

Every public analysis returns `AnalysisResult`:

```python
AnalysisResult(
    analysis="stable_machine_identifier",
    metrics={...},
    tables={...},
    metadata={"claim_type": "..."},
    warnings=[...],
    limitations=[...],
)
```

The result is JSON serialisable and can generate one standalone dark or publication figure from the same numerical data.

## No required changes outside the namespace

The stable import is:

```python
from vraven.darkside import DarkSideSession, TraceDataset
```

The extension does not require changes to `vraven/__init__.py`, the CLI or the existing trace schema.
