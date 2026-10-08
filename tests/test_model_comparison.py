from __future__ import annotations

import json

import numpy as np
import torch

import vraven


def _model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 2),
    )


def _scientific_arrays(figure) -> list[dict[str, object]]:
    payload = json.loads(figure.to_json())
    keys = ("x", "y", "z", "customdata", "text")
    return [{key: trace.get(key) for key in keys if key in trace} for trace in payload["data"]]


def test_identical_models_have_identical_reasoning_fingerprints() -> None:
    first = _model(44)
    second = _model(45)
    second.load_state_dict(first.state_dict())
    torch.manual_seed(46)
    data = torch.randn(5, 4)

    comparison = vraven.compare_models(
        first,
        second,
        data,
        max_samples=5,
        fingerprint_dimensions=40,
        left_name="reference",
        right_name="candidate",
    )

    assert np.allclose(comparison.fingerprint_similarity, 1.0)
    assert comparison.prediction_agreement.all()
    assert np.allclose(comparison.confidence_delta, 0.0)
    summary = comparison.summary()
    assert summary["prediction_agreement_rate"] == 1.0

    light = comparison.alignment(mode="publication_light")
    dark = comparison.alignment(mode="interactive_dark")
    assert _scientific_arrays(light) == _scientific_arrays(dark)
    assert light.layout.meta["scientific"]["left_model"] == "reference"


def test_checkpoint_evolution_uses_declared_order() -> None:
    models = [_model(51), _model(52), _model(53)]
    torch.manual_seed(54)
    inputs = torch.randn(1, 4)
    evolution = vraven.compare_checkpoints(
        models,
        inputs,
        checkpoint_names=["epoch_1", "epoch_10", "epoch_50"],
        fingerprint_dimensions=36,
    )

    assert evolution.checkpoint_names == ("epoch_1", "epoch_10", "epoch_50")
    assert evolution.similarity_to_final.shape == (3,)
    assert np.isclose(evolution.similarity_to_final[-1], 1.0)
    light = evolution.figure(mode="publication_light")
    dark = evolution.figure(mode="interactive_dark")
    assert _scientific_arrays(light) == _scientific_arrays(dark)
    assert light.layout.meta["scientific"]["final_checkpoint"] == "epoch_50"
