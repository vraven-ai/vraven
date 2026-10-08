from __future__ import annotations

import torch

import vraven


def test_input_fingerprint_is_stable_and_changes_with_input() -> None:
    torch.manual_seed(30)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    first = torch.randn(2, 4)
    report_a = vraven.explain(model, first)
    report_b = vraven.explain(model, first.clone())
    report_c = vraven.explain(model, first + 0.25)
    key = "input_fingerprint_hash"
    assert report_a.trace.metadata[key] == report_b.trace.metadata[key]
    assert report_a.trace.metadata[key] != report_c.trace.metadata[key]
    assert report_a.trace.metadata["target_score_reduction"] == "batch_mean"
