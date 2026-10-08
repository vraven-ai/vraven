from __future__ import annotations

import json

import torch

import vraven


def _scientific_signature(figure) -> str:
    payload = json.loads(figure.to_json())
    return json.dumps(
        [
            {key: trace.get(key) for key in ("type", "x", "y", "customdata", "text") if key in trace}
            for trace in payload["data"]
        ],
        sort_keys=True,
    )


def test_community_graph_is_deterministic_and_dual_mode() -> None:
    torch.manual_seed(22)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 18),
        torch.nn.Tanh(),
        torch.nn.Linear(18, 3),
    )
    report = vraven.explain(model, torch.randn(20, 6), target=1, mode="gradient")
    light = report.community_graph(layer=0, n_clusters=5, mode="publication_light")
    dark = report.community_graph(layer=0, n_clusters=5, mode="interactive_dark")
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert dark.layout.paper_bgcolor != "#FFFFFF"
    assert _scientific_signature(light) == _scientific_signature(dark)
    assert light.layout.meta["scientific"]["community_semantics"] == "latent_activation_pattern_group"
    assert light.layout.meta["scientific"]["causal"] is False
