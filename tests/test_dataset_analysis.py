from __future__ import annotations

import json

import numpy as np
import torch

import vraven


def _model(seed: int = 31) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(5, 12),
        torch.nn.GELU(),
        torch.nn.Linear(12, 3),
    )


def _scientific_arrays(figure) -> list[dict[str, object]]:
    payload = json.loads(figure.to_json())
    keys = ("x", "y", "z", "customdata", "text")
    return [{key: trace.get(key) for key in keys if key in trace} for trace in payload["data"]]


def test_explain_dataset_fingerprints_and_round_trip(tmp_path) -> None:
    torch.manual_seed(32)
    data = torch.randn(6, 5)
    report = vraven.explain_dataset(
        _model(),
        data,
        mode="fast",
        max_samples=6,
        fingerprint_dimensions=32,
    )

    assert len(report.reports) == 6
    assert report.fingerprints.shape == (6, 32)
    similarity = report.similarity_matrix()
    assert similarity.shape == (6, 6)
    assert np.allclose(np.diag(similarity), 1.0)
    assert report.outlier_scores(neighbours=2).shape == (6,)

    light = report.reasoning_landscape(mode="publication_light")
    dark = report.reasoning_landscape(mode="interactive_dark")
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert dark.layout.paper_bgcolor != "#FFFFFF"
    assert _scientific_arrays(light) == _scientific_arrays(dark)

    path = report.save(tmp_path / "dataset_report.json")
    loaded = vraven.VRavenDatasetReport.load(path)
    assert loaded.sample_ids == report.sample_ids
    assert np.allclose(loaded.fingerprints, report.fingerprints)


def test_dataset_item_preserves_ids_labels_and_targets() -> None:
    items = [
        vraven.DatasetItem(
            inputs=torch.full((1, 5), float(index)),
            target=index % 3,
            label=f"cohort_{index % 2}",
            sample_id=f"item-{index}",
        )
        for index in range(4)
    ]
    report = vraven.explain_dataset(
        _model(),
        items,
        mode="fast",
        max_samples=None,
        fingerprint_dimensions=24,
    )
    assert report.sample_ids == ("item-0", "item-1", "item-2", "item-3")
    assert report.labels == ("cohort_0", "cohort_1", "cohort_0", "cohort_1")
    figure = report.reasoning_landscape(colour_by="label", mode="publication_light")
    assert figure.layout.meta["scientific"]["group_name"] == "label"


def test_dataset_outlier_view_has_declared_neighbourhood() -> None:
    torch.manual_seed(21)
    report = vraven.explain_dataset(_model(), torch.randn(7, 5), mode="fast", max_samples=7)
    figure = report.reasoning_outliers(neighbours=3, mode="publication_light")
    metadata = figure.layout.meta["scientific"]
    assert metadata["resolved_neighbours"] == 3
    assert len(figure.data[0].y) == 7
