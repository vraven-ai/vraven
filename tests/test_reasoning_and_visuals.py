from __future__ import annotations

import json

import torch

import vraven


def _report() -> vraven.VRavenReport:
    torch.manual_seed(9)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 12),
        torch.nn.GELU(),
        torch.nn.Linear(12, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 3),
    )
    return vraven.explain(model, torch.randn(18, 6), target=2, mode="gradient")


def _data_signature(figure) -> str:
    payload = json.loads(figure.to_json())
    scientific = []
    for trace in payload["data"]:
        scientific.append(
            {key: trace.get(key) for key in ("type", "x", "y", "z", "customdata", "text") if key in trace}
        )
    return json.dumps(scientific, sort_keys=True)


def test_core_visuals_support_identical_data_in_both_modes() -> None:
    report = _report()
    visuals = (
        "activation_energy",
        "decision_flow",
        "spectral_trace",
        "neuron_impact",
        "reasoning_dna",
        "conflict_field",
        "evidence_trajectory",
        "input_attribution",
        "reasoning_graph",
    )
    for visual in visuals:
        publication = report.figure(visual, mode="publication_light")
        dark = report.figure(visual, mode="interactive_dark")
        assert publication.layout.paper_bgcolor == "#FFFFFF"
        assert dark.layout.paper_bgcolor != "#FFFFFF"
        assert _data_signature(publication) == _data_signature(dark)
        assert publication.layout.meta["vraven"]["standalone"] is True


def test_latent_flight_uses_one_shared_projection() -> None:
    report = _report()
    figure = report.latent_flight(mode="publication_light", max_layers=4)
    metadata = figure.layout.meta["scientific"]
    assert "one PCA basis" in metadata["normalisation"]
    assert len(metadata["selected_layers"]) == 4
