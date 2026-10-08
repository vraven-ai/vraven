from __future__ import annotations

import numpy as np

from vraven.darkside import (
    TraceDataset,
    analyse_adversarial_rerouting,
    analyse_cost_evidence_tradeoff,
    analyse_failure_anatomy,
    analyse_prediction_depth,
    analyse_representation_drift,
    analyse_stability_geometry,
    analyse_topological_shift,
    audit_unused_capacity,
    compare_class_conditionals,
    layerwise_shortcut_probe,
)


def make_dataset(*, shifted: float = 0.0, seed: int = 1) -> TraceDataset:
    rng = np.random.default_rng(seed)
    count = 48
    labels = np.array([0] * 24 + [1] * 24, dtype=object)
    predicted = labels.copy()
    predicted[[2, 4, 26, 28]] = [1, 1, 0, 0]
    first = np.vstack([rng.normal(0.0 + shifted, 1.0, (24, 8)), rng.normal(1.0 + shifted, 1.0, (24, 8))])
    second = np.tanh(first @ rng.normal(size=(8, 8))) + shifted
    influences = {"layer1#0": rng.normal(size=(count, 8)), "layer2#0": rng.normal(size=(count, 8))}
    energies = {name: np.abs(values) + 0.01 for name, values in influences.items()}
    return TraceDataset(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        true_labels=labels, predicted_labels=predicted, target_scores=rng.normal(size=count),
        representations={"layer1#0": first, "layer2#0": second}, unit_influences=influences,
        unit_energies=energies, activation_rms={"layer1#0": np.ones(count), "layer2#0": np.ones(count) * 2},
        timing_ms={"layer1#0": np.ones(count) * 0.2, "layer2#0": np.ones(count) * 0.5},
        fingerprints=rng.normal(size=(count, 20)) + shifted,
    )



def test_class_conditional_and_failure_anatomy() -> None:
    dataset = make_dataset()
    class_result = compare_class_conditionals(dataset, class_a=0, class_b=1)
    assert class_result.metrics["layers_compared"] == 2
    failure = analyse_failure_anatomy(dataset, failure="fp", reference="tp")
    assert failure.metrics["failure_samples"] == 2


def test_drift_stability_and_rerouting_profiles() -> None:
    reference = make_dataset(seed=2)
    comparison = make_dataset(seed=2, shifted=0.15)
    drift = analyse_representation_drift(reference, comparison, sliced_projections=16)
    stability = analyse_stability_geometry(reference, comparison, neighbours=4)
    rerouting = analyse_adversarial_rerouting(reference, comparison, neighbours=4)
    assert drift.metrics["shared_layers"] == 2
    assert stability.metrics["layers"] == 2
    assert rerouting.metrics["samples"] == 48


def test_capacity_cost_topology_and_prediction_depth() -> None:
    reference = make_dataset(seed=3)
    comparison = make_dataset(seed=3, shifted=0.05)
    capacity = audit_unused_capacity(reference)
    cost = analyse_cost_evidence_tradeoff(reference)
    topology = analyse_topological_shift(reference, comparison, maximum_points=32)
    depth = analyse_prediction_depth(reference, neighbours=3)
    assert capacity.metrics["units_audited"] == 16
    assert cost.metrics["layers"] == 2
    assert topology.metrics["layers"] == 2
    assert depth.metrics["layers"] == 2


def test_layerwise_shortcut_probe() -> None:
    dataset = make_dataset(seed=4)
    result = layerwise_shortcut_probe(dataset, dataset.true_labels, folds=3)
    assert 0.0 <= result.metrics["maximum_probe_auc"] <= 1.0
