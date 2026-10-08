from __future__ import annotations

import numpy as np
import torch

from vraven.darkside import (
    InputFeatureGroup,
    TraceDataset,
    UnitGroup,
    analyse_activation_mediation,
    analyse_adversarial_rerouting,
    analyse_causal_synergy,
    analyse_circuit_multiplicity,
    analyse_cost_evidence_tradeoff,
    analyse_explanation_uncertainty,
    analyse_failure_anatomy,
    analyse_high_confidence_ood,
    analyse_mechanism_transfer,
    analyse_method_disagreement,
    analyse_prediction_depth,
    analyse_representation_drift,
    analyse_stability_geometry,
    analyse_topological_shift,
    audit_unused_capacity,
    build_circuit_wiring,
    compare_class_conditionals,
    detect_shortcut_reliance,
    discover_decision_circuitry,
    inspect_internal_anatomy,
    map_class_conditional_unit_field,
)


class VisualToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4, bias=False)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(4))
            self.output.weight.copy_(
                torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.1, 0.1]])
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu(self.hidden(inputs)))


def _dataset(*, shifted: float = 0.0, seed: int = 73) -> TraceDataset:
    rng = np.random.default_rng(seed)
    count = 40
    labels = np.array([0] * 20 + [1] * 20, dtype=object)
    predicted = labels.copy()
    predicted[[1, 3, 21, 23]] = [1, 1, 0, 0]
    first = np.vstack(
        [
            rng.normal(0.0 + shifted, 1.0, (20, 8)),
            rng.normal(1.0 + shifted, 1.0, (20, 8)),
        ]
    )
    second = np.tanh(first @ rng.normal(size=(8, 8))) + shifted
    influences = {
        "layer1#0": rng.normal(size=(count, 8)),
        "layer2#0": rng.normal(size=(count, 8)),
    }
    energies = {name: np.abs(values) + 0.01 for name, values in influences.items()}
    return TraceDataset(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        true_labels=labels,
        predicted_labels=predicted,
        target_scores=rng.normal(size=count),
        representations={"layer1#0": first, "layer2#0": second},
        unit_influences=influences,
        unit_energies=energies,
        activation_rms={"layer1#0": np.ones(count), "layer2#0": np.ones(count) * 2},
        timing_ms={"layer1#0": np.ones(count) * 0.2, "layer2#0": np.ones(count) * 0.5},
        fingerprints=rng.normal(size=(count, 20)) + shifted,
    )


def _axis_payload(figure):
    payload = []
    for trace in figure.data:
        row = {}
        for name in ("x", "y", "z"):
            value = getattr(trace, name, None)
            if value is None:
                row[name] = None
            else:
                row[name] = np.asarray(value, dtype=object).tolist()
        payload.append(row)
    return payload


def test_every_analysis_renders_standalone_dark_and_publication_figures() -> None:
    model = VisualToyModel().eval()
    sample = torch.tensor([[2.0, 1.0, 0.2, 0.1]])
    clean = _dataset(seed=73)
    shifted = _dataset(seed=73, shifted=0.15)
    repeated = np.random.default_rng(73).normal(size=(20, 8))

    circuit = discover_decision_circuitry(
        model,
        sample,
        target=1,
        layer_names=("hidden", "output"),
        group_size=1,
        relative_score_tolerance=0.20,
    )
    results = [
        circuit,
        detect_shortcut_reliance(
            model,
            sample,
            target=1,
            feature_groups=(
                InputFeatureGroup((0,), label="dominant"),
                InputFeatureGroup((2, 3), label="weak"),
            ),
        ),
        compare_class_conditionals(clean, class_a=0, class_b=1),
        analyse_representation_drift(clean, shifted, sliced_projections=8),
        analyse_failure_anatomy(clean, failure="fp", reference="tp"),
        analyse_stability_geometry(clean, shifted, neighbours=3),
        analyse_method_disagreement(
            {
                "gradient": np.array([4.0, 3.0, 2.0, 1.0]),
                "lrp": np.array([4.0, -3.0, 1.0, 2.0]),
                "causal": np.array([1.0, 4.0, 3.0, 2.0]),
            },
            top_k=2,
        ),
        audit_unused_capacity(clean),
        analyse_adversarial_rerouting(clean, shifted, neighbours=3),
        analyse_cost_evidence_tradeoff(clean),
        analyse_prediction_depth(clean, neighbours=3),
        analyse_circuit_multiplicity(
            np.asarray(
                [
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1],
                ]
            ),
            confidence=0.80,
            stable_threshold=0.50,
        ),
        analyse_causal_synergy(
            model,
            sample,
            target=1,
            groups=(
                UnitGroup("hidden", (0,), label="strong"),
                UnitGroup("hidden", (1,), label="secondary"),
            ),
        ),
        analyse_activation_mediation(
            model,
            sample,
            torch.zeros_like(sample),
            target=1,
            layer_names=("hidden",),
        ),
        analyse_topological_shift(clean, shifted, maximum_points=32),
        analyse_explanation_uncertainty(repeated, top_k=3, bootstrap_repeats=50),
        analyse_mechanism_transfer(
            {
                "checkpoint_a": repeated[0],
                "checkpoint_b": repeated[1],
                "checkpoint_c": repeated[2],
            },
            top_k=3,
        ),
        analyse_high_confidence_ood(
            clean.fingerprints,
            np.vstack([clean.fingerprints[:1], np.full((1, clean.fingerprints.shape[1]), 8.0)]),
            query_confidence=[0.99, 0.99],
        ),
        inspect_internal_anatomy(model, sample, target=1, top_units=4),
        build_circuit_wiring(model, sample, circuit),
        map_class_conditional_unit_field(clean, class_a=0, class_b=1),
    ]

    assert len(results) == 21
    for result in results:
        dark = result.figure(mode="interactive_dark")
        light = result.figure(mode="publication_light")
        assert dark.layout.paper_bgcolor == "#030A12", result.analysis
        assert light.layout.paper_bgcolor == "#FFFFFF", result.analysis
        assert len(dark.data) >= 1, result.analysis
        assert len(dark.data) == len(light.data), result.analysis
        assert _axis_payload(dark) == _axis_payload(light), result.analysis
