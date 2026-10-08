# Quick recipes for all DarkSide analyses

These examples assume the extension has been merged into an existing VRAVEN workspace. They are deliberately independent. Run only the analyses required by the scientific question.

## Shared setup

```python
import numpy as np
import torch
import vraven

from vraven.darkside import DarkSideSession, TraceDataset

model = MyModel().eval()
samples = [sample_0, sample_1, sample_2]
reports = [vraven.explain(model, x, target=1, mode="gradient") for x in samples]

dataset = TraceDataset.from_reports(
    reports,
    true_labels=[0, 1, 1],
    sample_ids=["sha256-0", "sha256-1", "sha256-2"],
)
```

For cohort geometry, drift, evasion and failure analyses, use substantially larger and representative audit cohorts than this compact example.

## 1. Decision circuitry

```python
session = DarkSideSession(model, samples[0], target=1)

result = session.decision_circuitry(
    layer_names=("encoder.0", "encoder.2"),
    group_size=8,
    relative_score_tolerance=0.05,
    baseline="zero",
    maximum_groups=128,
)
```

Use for a bounded question such as: which tested internal unit groups can be removed while preserving the selected prediction and target score within the declared tolerance?

## 2. Shortcut reliance

```python
from vraven.darkside import InputFeatureGroup

result = session.shortcut_detection(
    [
        InputFeatureGroup((17,), label="LOLBIN flag"),
        InputFeatureGroup(tuple(range(80, 96)), label="API cluster"),
    ],
    baseline="zero",
)
```

The feature groups must have domain meaning. A local intervention effect is evidence of dependence, not proof that the group is a spurious shortcut.

## 3. Class-conditional internals

```python
from vraven.darkside import compare_class_conditionals

result = compare_class_conditionals(
    dataset,
    class_a="benign",
    class_b="malware",
    label_source="true",
    top_units=25,
)
```

## 4. Representation drift

```python
from vraven.darkside import analyse_representation_drift

result = analyse_representation_drift(
    checkpoint_a_dataset,
    checkpoint_b_dataset,
    sliced_projections=128,
    seed=42,
)
```

The two datasets must contain identical ordered sample IDs and equal representation shapes at the compared layers.

## 5. Failure anatomy

```python
from vraven.darkside import analyse_failure_anatomy

false_positive_anatomy = analyse_failure_anatomy(
    dataset,
    failure="fp",
    reference="tp",
    positive_label=1,
)
```

Valid cohort names are `tp`, `fp`, `fn` and `tn`.

## 6. Stability geometry

```python
from vraven.darkside import analyse_stability_geometry

result = analyse_stability_geometry(
    clean_dataset,
    perturbed_dataset,
    collapse_threshold=0.70,
    neighbours=10,
)
```

The output identifies the earliest tested layer where the transparent composite stability score falls below the declared threshold. Inspect every component rather than relying only on the composite.

## 7. Cross-method disagreement

```python
from vraven.darkside import analyse_method_disagreement

result = analyse_method_disagreement(
    {
        "input_x_gradient": input_x_gradient_vector,
        "lrp": lrp_vector,
        "shap": shap_vector,
        "causal_group_effect": causal_vector,
    },
    top_k=20,
    sign_threshold=1e-8,
)
```

All vectors must refer to the same indexed explanatory surface. Do not compare a feature vector with a layer vector without a declared aggregation that makes the coordinates comparable.

## 8. Unused capacity

```python
from vraven.darkside import audit_unused_capacity

result = audit_unused_capacity(
    dataset,
    activity_threshold=1e-6,
    low_activity_fraction=0.95,
    high_impact_quantile=0.90,
)
```

The categories are audit-dataset diagnostics, not proof that parameters can be safely removed.

## 9. Adversarial or evasion rerouting

```python
from vraven.darkside import analyse_adversarial_rerouting

result = analyse_adversarial_rerouting(
    clean_dataset,
    evasive_dataset,
    input_l0=changed_feature_counts,
    input_l2=input_l2_distances,
    neighbours=10,
)
```

The clean and altered rows must be paired exactly.

## 10. Cost–evidence trade-off

```python
from vraven.darkside import analyse_cost_evidence_tradeoff

result = analyse_cost_evidence_tradeoff(
    dataset,
    evidence="absolute_influence",
)
```

The result reports a Pareto frontier. Timing recorded through hooks includes instrumentation cost, so use device synchronisation and repeated profiling for publication-grade GPU measurements.

## 11. Prediction depth and shortcut probes

```python
from vraven.darkside import analyse_prediction_depth, layerwise_shortcut_probe

prediction_depth = analyse_prediction_depth(
    reference_dataset,
    query_dataset,
    neighbours=5,
)

shortcut_probe = layerwise_shortcut_probe(
    reference_dataset,
    shortcut_labels=scanner_or_source_labels,
    folds=5,
    seed=42,
)
```

Prediction depth uses local k-nearest-neighbour probes. Shortcut probing measures decodability, not causal use.

## 12. Circuit multiplicity

```python
from vraven.darkside import analyse_circuit_multiplicity

result = analyse_circuit_multiplicity(
    {
        "seed_1": circuit_mask_seed_1,
        "seed_2": circuit_mask_seed_2,
        "baseline_mean": circuit_mask_mean_baseline,
        "perturbation_1": circuit_mask_perturbed_sample,
    },
    component_labels=aligned_component_names,
    confidence=0.95,
    stable_threshold=0.80,
)
```

The output distinguishes a repeatedly included empirical core from alternative routes and threshold-ambiguous components. It is not a formal verification certificate.

## 13. Causal synergy and redundancy

```python
from vraven.darkside import UnitGroup

result = session.causal_synergy(
    [
        UnitGroup("encoder.2", (0, 1, 2, 3), label="community_A"),
        UnitGroup("encoder.2", (4, 5, 6, 7), label="community_B"),
    ],
    baseline="zero",
    maximum_pairs=100,
)
```

Positive interaction is super-additive under the declared intervention. Negative interaction can indicate redundancy or compensation.

## 14. Activation mediation and repair

```python
result = session.activation_mediation(
    corrupted_sample,
    layer_names=("encoder.0", "encoder.2", "classifier"),
    include_reverse_noising=True,
)
```

The method reports clean-to-corrupt recovery and optional reverse damage. It does not silently interpret patch recovery as a pure natural indirect effect.

## 15. Topological representation shift

```python
from vraven.darkside import analyse_topological_shift

result = analyse_topological_shift(
    reference_dataset,
    shifted_dataset,
    include_h1=False,
    maximum_points=512,
    seed=42,
)
```

Exact H0 death times are implemented through minimum spanning trees. Optional H1 summaries require `ripser` and remain lifetime-distribution comparisons rather than a complete persistence-diagram distance.

## 16. Explanation uncertainty

```python
from vraven.darkside import analyse_explanation_uncertainty

result = analyse_explanation_uncertainty(
    repeated_attribution_matrix,
    top_k=20,
    confidence=0.95,
    bootstrap_repeats=2000,
    seed=42,
    element_labels=feature_or_unit_names,
)
```

Rows may represent retraining seeds, stochastic explainer runs, controlled perturbations or resampled backgrounds. State exactly what was repeated.

## 17. Mechanism transfer

```python
from vraven.darkside import analyse_mechanism_transfer

result = analyse_mechanism_transfer(
    {
        "checkpoint_10": effect_vector_checkpoint_10,
        "checkpoint_50": effect_vector_checkpoint_50,
        "new_domain": effect_vector_new_domain,
    },
    top_k=20,
)
```

Component ordering must be genuinely aligned across conditions.

## 18. High-confidence internal anomaly

```python
from vraven.darkside import analyse_high_confidence_ood

result = analyse_high_confidence_ood(
    reference_fingerprints,
    query_fingerprints,
    query_confidence=query_probabilities,
    confidence_threshold=0.95,
    anomaly_percentile_threshold=0.99,
)
```

This is an internal anomaly screen. It is not proof that an input is adversarial, malicious or out of distribution.

## Save and render any result

```python
result.save_json("darkside_result.json")
result.figure(mode="interactive_dark").show()
result.figure(mode="publication_light").write_html("darkside_publication.html")
```
