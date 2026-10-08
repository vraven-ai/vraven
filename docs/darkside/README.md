# VRAVEN DarkSide Extensions

Version 1.0.0. Merge-only research extension for VRAVEN 0.5.x.

VRAVEN DarkSide is a merge-only extension for advanced internal auditing of deep neural networks. It does not replace or rewrite the main VRAVEN package. The archive contributes one isolated namespace:

```text
src/vraven/darkside/
```

The extension moves beyond a single feature-importance vector. It studies internal causal sufficiency, shortcut reliance, alternative circuits, class and failure cohorts, representation drift, perturbation collapse, explanation disagreement, capacity use, evasion rerouting, runtime efficiency and atypical high-confidence mechanisms.

## Included analyses

The first ten items correspond directly to the requested dark-side angles. Eight further analyses were added because they expose interaction, multiplicity and reliability questions that ordinary feature attribution does not answer.

| No. | Analysis | Stable API |
|---:|---|---|
| 1 | Decision circuitry | `discover_decision_circuitry` |
| 2 | Shortcut reliance | `detect_shortcut_reliance` |
| 3 | Class-conditional internals | `compare_class_conditionals` |
| 4 | Representation drift | `analyse_representation_drift` |
| 5 | Failure anatomy | `analyse_failure_anatomy` |
| 6 | Stability geometry | `analyse_stability_geometry` |
| 7 | Cross-method disagreement | `analyse_method_disagreement` |
| 8 | Unused capacity | `audit_unused_capacity` |
| 9 | Adversarial or evasion rerouting | `analyse_adversarial_rerouting` |
| 10 | Cost–evidence trade-off | `analyse_cost_evidence_tradeoff` |
| 11 | Prediction depth | `analyse_prediction_depth` and `layerwise_shortcut_probe` |
| 12 | Circuit multiplicity | `analyse_circuit_multiplicity` |
| 13 | Causal synergy and redundancy | `analyse_causal_synergy` |
| 14 | Activation mediation and repair | `analyse_activation_mediation` |
| 15 | Topological representation shift | `analyse_topological_shift` |
| 16 | Explanation uncertainty | `analyse_explanation_uncertainty` |
| 17 | Mechanism transfer | `analyse_mechanism_transfer` |
| 18 | High-confidence internal anomaly | `analyse_high_confidence_ood` |
| 19 | Internal Linear anatomy | `inspect_internal_anatomy` / `print_internal_anatomy` |
| 20 | Measured circuit wiring | `build_circuit_wiring` |
| 21 | Class-conditional unit field | `map_class_conditional_unit_field` |

Analyses 19–21 expose the network’s measured internals rather than another attribution scatter: printable `W·a` anatomy, a node–edge wiring graph whose edge mass is `Σ|W_ji|·|a_i|` on a retained ε-core, and a dense layer×unit class-contrast field.

The analyses are not intended to run together by default. Use `recommend_features(...)` to select a focused audit for security, malware, training, model engineering, research or assurance work.

Every result family returns a standalone figure in both `interactive_dark` and `publication_light` modes. The two modes use the same measured values and differ only in visual presentation.

## Safe merge

Run a dry check first:

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

The script performs a complete preflight before changing the target. It refuses to replace existing DarkSide files unless `--overwrite` is supplied. When overwrite is explicitly enabled, the old files are copied to a timestamped directory under:

```text
VRAVEN/.vraven_darkside_backups/
```

## Installation after merging

From the existing VRAVEN repository:

```bash
python -m pip install -e ".[analysis,publication]"
```

Run the extension tests:

```bash
PYTHONPATH=src python -m pytest -q tests/test_darkside*.py
```

## Minimal live-model audit

```python
import torch
from vraven.darkside import DarkSideSession, InputFeatureGroup

model = MyModel().eval()
sample = torch.load("sample.pt", map_location="cpu", weights_only=True)

session = DarkSideSession(model, sample, target=1)

circuit = session.decision_circuitry(
    layer_names=("encoder.0", "encoder.2"),
    group_size=8,
    relative_score_tolerance=0.05,
)

shortcut = session.shortcut_detection(
    [
        InputFeatureGroup((17,), label="LOLBIN flag"),
        InputFeatureGroup(tuple(range(80, 96)), label="API cluster"),
    ]
)

print(circuit.summary())
print(shortcut.summary())
circuit.figure(mode="interactive_dark").show()
circuit.figure(mode="publication_light").write_html("circuit_publication.html")
```

## Dataset-level audit

```python
import vraven
from vraven.darkside import TraceDataset, analyse_failure_anatomy

traces = [
    vraven.explain(model, sample, target=1, mode="gradient").trace
    for sample in samples
]

dataset = TraceDataset.from_traces(
    traces,
    true_labels=true_labels,
    sample_ids=sample_hashes,
)

result = analyse_failure_anatomy(
    dataset,
    failure="fp",
    reference="tp",
)

result.figure(mode="publication_light").show()
```

## Focused feature selection

```python
from vraven.darkside import recommend_features

for feature in recommend_features("malware"):
    print(feature.name, feature.cost, feature.claim_type)
```

## Validation evidence

The clean merge validation recorded 35 DarkSide tests passing and 90 tests passing across the merged VRAVEN workspace. See `VALIDATION_REPORT.md` and `docs/COVERAGE_REPORT.txt` for the exact environment, limitations and branch-aware coverage.

## Read before publication

Start with:

```text
docs/QUICK_RECIPES.md
docs/FEATURE_CATALOGUE.md
docs/MATHEMATICS.md
docs/RESEARCH_BASIS.md
docs/LIMITATIONS.md
docs/USAGE_BY_COMMUNITY.md
```

## Scientific boundary

VRAVEN DarkSide has a broader analytical scope than standard SHAP or LIME outputs because it can inspect internal representations and perform declared interventions. This does not establish universal superiority. A valid comparison must hold the model, sample, target, baseline, intervention budget and evaluation metric constant. Observational quantities such as activation, gradient, probe accuracy, CKA and attention remain separate from intervention-based evidence.
