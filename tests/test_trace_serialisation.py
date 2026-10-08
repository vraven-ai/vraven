from __future__ import annotations

import torch

import vraven


def test_trace_round_trip(tmp_path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    report = vraven.explain(model, torch.randn(3, 3), target=1, mode="gradient")
    path = report.save_trace(tmp_path / "trace.json")
    loaded = vraven.VRavenReport.load_trace(path)
    assert loaded.summary()["layers_captured"] == report.summary()["layers_captured"]
    assert loaded.trace.prediction is not None
    assert loaded.trace.prediction.predicted_classes == report.trace.prediction.predicted_classes
    assert loaded.reasoning_fingerprint().shape == (128,)
