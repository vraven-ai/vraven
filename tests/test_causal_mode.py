from __future__ import annotations

import torch

import vraven


def test_causal_mode_runs_temporary_interventions_and_restores_model() -> None:
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )
    model.train()
    report = vraven.explain(
        model,
        torch.randn(5, 4),
        target=1,
        mode="causal",
        causal_top_k=2,
    )
    assert model.training is True
    assert report.trace.causal_interventions
    assert all(item.strategy == "zero" for item in report.trace.causal_interventions)
    figure = report.causal_pathway(mode="interactive_dark")
    assert figure.layout.paper_bgcolor != "#FFFFFF"
