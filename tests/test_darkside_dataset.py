from __future__ import annotations

import numpy as np
import pytest

from vraven.darkside import TraceDataset


def make_dataset(*, shifted: float = 0.0, seed: int = 1) -> TraceDataset:
    rng = np.random.default_rng(seed)
    count = 48
    labels = np.array([0] * 24 + [1] * 24, dtype=object)
    predicted = labels.copy()
    predicted[[2, 4, 26, 28]] = [1, 1, 0, 0]
    first = np.vstack(
        [rng.normal(0.0 + shifted, 1.0, (24, 8)), rng.normal(1.0 + shifted, 1.0, (24, 8))]
    )
    second = np.tanh(first @ rng.normal(size=(8, 8))) + shifted
    influences = {
        "layer1#0": rng.normal(size=(count, 8)),
        "layer2#0": rng.normal(size=(count, 8)),
    }
    energies = {name: np.abs(values) + 0.01 for name, values in influences.items()}
    return TraceDataset(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        true_labels=labels,
        predicted_labels=predicted,
        target_scores=rng.normal(size=count),
        representations={"layer1#0": first, "layer2#0": second},
        unit_influences=influences,
        unit_energies=energies,
        activation_rms={"layer1#0": np.ones(count), "layer2#0": np.ones(count) * 2},
        timing_ms={"layer1#0": np.ones(count) * 0.2, "layer2#0": np.ones(count) * 0.5},
        fingerprints=rng.normal(size=(count, 20)) + shifted,
    )


def test_subset_and_confusion_cohorts_are_aligned() -> None:
    dataset = make_dataset()
    fp = dataset.confusion_cohort("fp", positive_label=1)
    tp = dataset.confusion_cohort("tp", positive_label=1)
    assert len(fp.sample_ids) == 2
    assert len(tp.sample_ids) == 22
    assert fp.representations["layer1#0"].shape[0] == 2


def test_dataset_rejects_misaligned_matrix() -> None:
    dataset = make_dataset()
    with pytest.raises(ValueError):
        TraceDataset(
            sample_ids=dataset.sample_ids,
            true_labels=dataset.true_labels,
            predicted_labels=dataset.predicted_labels,
            target_scores=dataset.target_scores,
            representations={"bad": np.ones((2, 3))},
        )


def test_from_reports_builds_aligned_dataset_across_fast_and_gradient_traces() -> None:
    import torch

    import vraven

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    ).eval()
    reports = [
        vraven.explain(model, torch.randn(1, 4), target=1, mode="fast"),
        vraven.explain(model, torch.randn(1, 4), target=1, mode="gradient"),
    ]

    dataset = TraceDataset.from_reports(
        reports,
        true_labels=[0, 1],
        sample_ids=["fast-sample", "gradient-sample"],
    )

    assert dataset.sample_ids == ("fast-sample", "gradient-sample")
    assert dataset.fingerprints is not None
    assert dataset.fingerprints.shape[0] == 2
    assert dataset.layers
    for layer in dataset.layers:
        assert dataset.representations[layer].shape[0] == 2
        assert dataset.unit_influences[layer].shape[0] == 2
