from __future__ import annotations

import json

import torch

import vraven


def _model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3),
    )


def _signature(figure) -> str:
    payload = json.loads(figure.to_json())
    return json.dumps(
        [
            {key: trace.get(key) for key in ("type", "x", "y", "customdata") if key in trace}
            for trace in payload["data"]
        ],
        sort_keys=True,
    )


def test_checkpoint_evolution_tracks_similarity_to_final() -> None:
    torch.manual_seed(18)
    result = vraven.compare_checkpoints(
        [_model(1), _model(2), _model(3)],
        torch.randn(1, 5),
        target=2,
        checkpoint_names=["epoch_1", "epoch_5", "epoch_20"],
        mode="gradient",
    )
    assert result.similarity_to_final.shape == (3,)
    assert abs(result.similarity_to_final[-1] - 1.0) < 1e-8
    assert result.summary()["checkpoint_count"] == 3
    light = result.figure(mode="publication_light")
    dark = result.figure(mode="interactive_dark")
    assert _signature(light) == _signature(dark)
    assert light.layout.meta["scientific"]["final_checkpoint"] == "epoch_20"
