from __future__ import annotations

import numpy as np

from vraven.decompiler import (
    CaptureConfig,
    ConceptDiscoveryConfig,
    ConceptDiscoveryEngine,
    ConceptEvolutionBuilder,
    EvolutionConfig,
    ModelCaptureRouter,
)
from vraven.decompiler.runtime import FXValueRecorder


def test_concepts_and_evolution_are_deterministic(tiny_model, tiny_data):
    inputs, labels = tiny_data
    captured = ModelCaptureRouter(CaptureConfig()).capture(tiny_model, (inputs[:2],), {})
    config = ConceptDiscoveryConfig(method="pca", max_nodes=3, concepts_per_node=3, seed=31)
    first = ConceptDiscoveryEngine(config).discover(captured, (inputs,), labels=labels)
    second = ConceptDiscoveryEngine(config).discover(captured, (inputs,), labels=labels)
    assert [concept.id for concept in first.concepts] == [concept.id for concept in second.concepts]
    assert first.score_matrix()[1] == second.score_matrix()[1]
    evolution = ConceptEvolutionBuilder(EvolutionConfig()).build(first)
    assert evolution.states
    assert all(0.0 <= edge.transport_mass <= 1.0 for edge in evolution.edges)


def test_unseen_activation_scoring_matches_fitted_reference_coordinates(tiny_model, tiny_data):
    inputs, labels = tiny_data
    captured = ModelCaptureRouter(CaptureConfig()).capture(tiny_model, (inputs[:2],), {})
    atlas = ConceptDiscoveryEngine(
        ConceptDiscoveryConfig(method="pca", max_nodes=3, concepts_per_node=3, seed=31)
    ).discover(captured, (inputs,), labels=labels)
    assert captured.graph_module is not None
    record = FXValueRecorder(
        captured.graph_module,
        selected_nodes=[item.node_id for item in atlas.concept_sets],
        detach=True,
        move_to_cpu=True,
    ).execute((inputs[:1],), {})
    for concept_set in atlas.concept_sets:
        scored = concept_set.score_activation(record.values[concept_set.node_id])
        np.testing.assert_allclose(
            scored.detach().cpu().numpy()[0],
            concept_set.scores[0],
            rtol=1e-5,
            atol=1e-5,
        )
