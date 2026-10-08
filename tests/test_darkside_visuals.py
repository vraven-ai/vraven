from __future__ import annotations

import numpy as np

from vraven.darkside import (
    AnalysisResult,
    TraceDataset,
    analyse_cost_evidence_tradeoff,
    analyse_explanation_uncertainty,
    analyse_representation_drift,
)


def _scientific_payload(figure):
    return [
        {
            "x": list(trace.x) if getattr(trace, "x", None) is not None else None,
            "y": list(trace.y) if getattr(trace, "y", None) is not None else None,
            "z": np.asarray(trace.z).tolist() if getattr(trace, "z", None) is not None else None,
        }
        for trace in figure.data
    ]

def make_dataset(*, shifted: float = 0.0, seed: int = 1) -> TraceDataset:
    rng = np.random.default_rng(seed)
    count = 48
    labels = np.array([0] * 24 + [1] * 24, dtype=object)
    predicted = labels.copy()
    predicted[[2, 4, 26, 28]] = [1, 1, 0, 0]
    first = np.vstack([rng.normal(0.0 + shifted, 1.0, (24, 8)), rng.normal(1.0 + shifted, 1.0, (24, 8))])
    second = np.tanh(first @ rng.normal(size=(8, 8))) + shifted
    influences = {"layer1#0": rng.normal(size=(count, 8)), "layer2#0": rng.normal(size=(count, 8))}
    energies = {name: np.abs(values) + 0.01 for name, values in influences.items()}
    return TraceDataset(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        true_labels=labels, predicted_labels=predicted, target_scores=rng.normal(size=count),
        representations={"layer1#0": first, "layer2#0": second}, unit_influences=influences,
        unit_energies=energies, activation_rms={"layer1#0": np.ones(count), "layer2#0": np.ones(count) * 2},
        timing_ms={"layer1#0": np.ones(count) * 0.2, "layer2#0": np.ones(count) * 0.5},
        fingerprints=rng.normal(size=(count, 20)) + shifted,
    )



def test_both_modes_preserve_scientific_values() -> None:
    reference = make_dataset(seed=11)
    comparison = make_dataset(seed=11, shifted=0.2)
    result = analyse_representation_drift(reference, comparison, sliced_projections=8)
    light = result.figure(mode="publication_light")
    dark = result.figure(mode="interactive_dark")
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert dark.layout.paper_bgcolor == "#030A12"
    assert _scientific_payload(light) == _scientific_payload(dark)


def test_capacity_and_uncertainty_figures_render() -> None:
    dataset = make_dataset(seed=12)
    cost = analyse_cost_evidence_tradeoff(dataset)
    uncertainty = analyse_explanation_uncertainty(
        np.random.default_rng(12).normal(size=(10, 8)), bootstrap_repeats=50
    )
    assert len(cost.figure(mode="interactive_dark").data) >= 1
    assert len(uncertainty.figure(mode="publication_light").data) == 1


def test_unknown_figure_is_rejected() -> None:
    result = AnalysisResult("unknown_analysis", metrics={})
    try:
        result.figure()
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown analysis should not produce an arbitrary figure")
