from __future__ import annotations

import numpy as np

from vraven.darkside import (
    analyse_explanation_uncertainty,
    analyse_high_confidence_ood,
    analyse_mechanism_transfer,
    analyse_method_disagreement,
    h0_persistence_deaths,
)


def test_method_disagreement_reports_rank_and_sign_structure() -> None:
    result = analyse_method_disagreement(
        {
            "input_grad": np.array([4.0, 3.0, 2.0, 1.0]),
            "lrp": np.array([4.0, -3.0, 1.0, 2.0]),
            "causal": np.array([1.0, 4.0, 3.0, 2.0]),
        },
        top_k=2,
    )
    assert result.metrics["methods"] == 3
    assert len(result.tables["pairs"]) == 3


def test_explanation_uncertainty_and_transfer() -> None:
    rng = np.random.default_rng(9)
    repeated = rng.normal(size=(30, 12))
    uncertainty = analyse_explanation_uncertainty(
        repeated, top_k=4, bootstrap_repeats=100
    )
    transfer = analyse_mechanism_transfer(
        {"checkpoint_a": repeated[0], "checkpoint_b": repeated[1], "checkpoint_c": repeated[2]},
        top_k=4,
    )
    assert uncertainty.metrics["runs"] == 30
    assert transfer.metrics["conditions"] == 3


def test_high_confidence_internal_anomaly() -> None:
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(100, 8))
    query = np.vstack([rng.normal(size=(2, 8)), np.full((1, 8), 15.0)])
    result = analyse_high_confidence_ood(
        reference, query, query_confidence=[0.95, 0.95, 0.999]
    )
    assert result.metrics["flagged_high_confidence_anomalies"] >= 1


def test_h0_persistence_deaths_match_line_spacing() -> None:
    values = np.array([[0.0], [1.0], [3.0]])
    deaths = h0_persistence_deaths(values)
    assert np.allclose(deaths, [1.0, 2.0])
