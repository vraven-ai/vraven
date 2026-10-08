from __future__ import annotations

import numpy as np
import pytest
import torch

from vraven.darkside import (
    InputFeatureGroup,
    TraceDataset,
    UnitGroup,
    analyse_high_confidence_ood,
    analyse_representation_drift,
)
from vraven.darkside.interventions import InterventionEngine


class MixedModeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.hidden = torch.nn.Linear(3, 3)
        self.output = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(self.dropout(inputs))))


def _dataset(*, shift: float = 0.0) -> TraceDataset:
    rng = np.random.default_rng(31)
    count = 12
    representation = rng.normal(size=(count, 5)) + shift
    return TraceDataset(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        true_labels=np.asarray([0, 1] * 6, dtype=object),
        predicted_labels=np.asarray([0, 1] * 6, dtype=object),
        target_scores=rng.normal(size=count),
        representations={"hidden#0": representation},
        fingerprints=rng.normal(size=(count, 8)) + shift,
    )


def test_intervention_engine_restores_mixed_training_flags() -> None:
    model = MixedModeModel()
    model.train()
    model.dropout.eval()
    original = {name: module.training for name, module in model.named_modules()}

    engine = InterventionEngine(model, torch.ones(1, 3), target=1)
    engine.evaluate()
    engine.capture(("hidden",))
    engine.intervene((UnitGroup("hidden", (0,), axis=-1),))

    restored = {name: module.training for name, module in model.named_modules()}
    assert restored == original


def test_group_contracts_reject_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        UnitGroup("hidden", ())
    with pytest.raises(ValueError):
        UnitGroup("hidden", (1, 1))
    with pytest.raises(ValueError):
        InputFeatureGroup((-1,))
    with pytest.raises(ValueError):
        InputFeatureGroup((0,), tensor_index=-1)


def test_trace_dataset_rejects_duplicate_sample_ids() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="unique"):
        TraceDataset(
            sample_ids=("duplicate",) * len(dataset.sample_ids),
            true_labels=dataset.true_labels,
            predicted_labels=dataset.predicted_labels,
            target_scores=dataset.target_scores,
            representations=dataset.representations,
        )


def test_drift_index_is_dimensionless_and_bounded() -> None:
    result = analyse_representation_drift(_dataset(), _dataset(shift=0.25), sliced_projections=16)
    row = result.tables["layers"][0]
    assert 0.0 <= row["cka_similarity"] <= 1.0
    assert 0.0 <= row["procrustes_distance"] <= 1.0
    assert 0.0 <= row["mmd2_rbf_bounded"] <= 1.0
    assert 0.0 <= row["sliced_wasserstein_bounded"] <= 1.0
    assert 0.0 <= row["drift_index"] <= 1.0


def test_ood_anomaly_threshold_is_configurable() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(80, 6))
    query = np.vstack([rng.normal(size=(1, 6)), np.full((1, 6), 8.0)])

    strict = analyse_high_confidence_ood(
        reference,
        query,
        query_confidence=[0.99, 0.99],
        anomaly_percentile_threshold=0.999,
    )
    permissive = analyse_high_confidence_ood(
        reference,
        query,
        query_confidence=[0.99, 0.99],
        anomaly_percentile_threshold=0.90,
    )

    assert permissive.metrics["flagged_high_confidence_anomalies"] >= strict.metrics[
        "flagged_high_confidence_anomalies"
    ]
    assert permissive.metadata["anomaly_percentile_threshold"] == 0.90
