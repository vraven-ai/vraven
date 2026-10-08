"""Tests for LRP, t-SNE projection and new mockup figures."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import vraven
from vraven.reasoning.projection import dataset_tsne, layer_tsne, shared_pca


def _tiny_classifier() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


def test_lrp_linear_relu_conservation_reasonable() -> None:
    model = _tiny_classifier()
    inputs = torch.randn(6, 8)
    result = vraven.explain_lrp(model, inputs, target=1)
    assert result.layers
    assert result.input_relevance is not None
    assert abs(result.conservation_ratio - 1.0) < 0.75
    figure = vraven.VRavenReport(
        vraven.explain(model, inputs, target=1, mode="fast").trace,
        lrp_result=result,
    ).relevance_flow(mode="interactive_dark")
    assert figure.layout.meta["scientific"]["visual_id"] == "vraven.relevance_flow"


def test_lrp_rejects_unsupported_only_model() -> None:
    from vraven.core.errors import CapabilityError

    model = nn.Sequential(nn.GELU())
    with pytest.raises(CapabilityError):
        vraven.explain_lrp(model, torch.randn(4, 8), target=0)


def test_shared_pca_and_layer_tsne() -> None:
    model = _tiny_classifier()
    inputs = torch.randn(20, 8)
    report = vraven.explain(model, inputs, target=1, mode="gradient")
    pca = shared_pca(report.trace, max_layers=3)
    assert pca.projection == "shared_pca"
    assert len(pca.coordinates) >= 1
    tsne = layer_tsne(
        report.trace, max_layers=3, perplexity=5.0, random_seed=3, n_iter=120
    )
    assert "tsne" in tsne.projection
    assert tsne.random_seed == 3
    fig = report.latent_flight(
        method="tsne",
        max_layers=3,
        perplexity=5.0,
        mode="publication_light",
    )
    assert fig.layout.meta["scientific"]["method"] == "tsne"


def test_dataset_tsne_and_landscape_highlight() -> None:
    matrix = np.random.default_rng(0).normal(size=(30, 16))
    coords, meta = dataset_tsne(matrix, perplexity=5.0, random_seed=11, n_iter=120)
    assert coords.shape == (30, 2)
    assert meta["projection"] == "dataset_tsne"
    from vraven.visualisation.reasoning_dataset import reasoning_landscape_figure

    figure = reasoning_landscape_figure(
        matrix,
        groups=["a"] * 15 + ["b"] * 15,
        method="tsne",
        highlight_index=3,
        perplexity=5.0,
        mode="interactive_dark",
    )
    assert figure.layout.meta["scientific"]["method"] == "tsne"


def test_neuron_importance_and_sankey_and_perturbation() -> None:
    model = _tiny_classifier()
    inputs = torch.randn(12, 8)
    report = vraven.explain(model, inputs, target=1, mode="gradient")
    importance = report.neuron_importance(top_k=8, mode="interactive_dark")
    assert importance.layout.meta["scientific"]["visual_id"] == "vraven.neuron_importance"
    sankey = report.decision_sankey(mode="interactive_dark")
    assert sankey.layout.meta["scientific"]["visual_id"] == "vraven.decision_sankey"
    heatmap = report.feature_sequence_heatmap(
        feature_names=[f"f{i}" for i in range(8)],
        mode="publication_light",
    )
    assert heatmap.layout.meta["scientific"]["visual_id"] == "vraven.feature_sequence_heatmap"

    stability = vraven.run_perturbation_stability(
        model, inputs, target=1, repeats=2, levels=[0.0, 0.5, 1.0]
    )
    report.perturbation_result = stability
    figure = report.perturbation_stability(mode="interactive_dark")
    assert figure.layout.meta["scientific"]["visual_id"] == "vraven.perturbation_stability"


def test_counterfactual_ui_figure() -> None:
    model = _tiny_classifier()
    inputs = torch.randn(1, 8)
    result = vraven.generate_counterfactual(model, inputs, target_class=2, steps=25)
    report = vraven.explain(model, inputs, target=2, mode="fast")
    report.counterfactual_result = result
    figure = report.counterfactual_delta(
        feature_names=[f"f{i}" for i in range(8)],
        class_names=["a", "b", "c"],
        mode="interactive_dark",
    )
    assert figure.layout.meta["scientific"]["visual_id"] == "vraven.counterfactual_delta"
