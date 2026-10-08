from __future__ import annotations

import numpy as np

from vraven.decompiler.adversarial import wilson_score_interval
from vraven.decompiler.contrastive import _decisive_depth
from vraven.decompiler.counterfactual import _analytic_boundary_candidates
from vraven.decompiler.programme import (
    MechanismProgram,
    ProgrammeFidelity,
    ProgrammeRule,
)


def _programme() -> MechanismProgram:
    return MechanismProgram(
        target=1,
        concept_ids=("c0", "c1"),
        concept_labels={"c0": "c0", "c1": "c1"},
        intercept=1.0,
        rules=[
            ProgrammeRule(
                id="R00",
                kind="main",
                concept_ids=("c0",),
                labels=("c0",),
                coefficient=2.0,
                term_mean=0.0,
                term_scale=1.0,
                evidence_tier="intervention_supported",
            ),
            ProgrammeRule(
                id="R01",
                kind="interaction",
                concept_ids=("c0", "c1"),
                labels=("c0", "c1"),
                coefficient=0.5,
                term_mean=0.0,
                term_scale=1.0,
                evidence_tier="intervention_supported",
            ),
        ],
        fidelity=ProgrammeFidelity(
            train_mae=0.0,
            validation_mae=0.0,
            train_r2=1.0,
            validation_r2=1.0,
            train_correlation=1.0,
            validation_correlation=1.0,
            train_decision_agreement=1.0,
            validation_decision_agreement=1.0,
        ),
    )


def test_decisive_depth_excludes_positive_intercept_only_state() -> None:
    cumulative = [(0.0, 0.8), (0.4, 0.2), (0.7, 1.4)]
    assert _decisive_depth(cumulative, exclude_intercept=True) == 0.4
    assert _decisive_depth(cumulative, exclude_intercept=False) == 0.0


def test_decisive_depth_can_resolve_evidence_at_depth_zero() -> None:
    cumulative = [(0.0, 0.8), (0.0, 1.2), (0.8, 1.5)]
    assert _decisive_depth(cumulative, exclude_intercept=True) == 0.0


def test_analytic_candidate_crosses_exact_multilinear_programme_boundary() -> None:
    programme = _programme()
    values = np.asarray([[1.0, 2.0]])
    candidates = _analytic_boundary_candidates(
        programme,
        values,
        "c0",
        lower=-2.0,
        upper=2.0,
    )
    assert len(candidates) == 2
    root, crossed = candidates
    assert np.isclose(programme.intervene(values, {"c0": root})[0], 0.0, atol=1e-8)
    assert programme.intervene(values, {"c0": crossed})[0] <= 0.0


def test_analytic_candidate_rejects_boundary_outside_empirical_interval() -> None:
    candidates = _analytic_boundary_candidates(
        _programme(),
        np.asarray([[1.0, 2.0]]),
        "c0",
        lower=0.0,
        upper=2.0,
    )
    assert candidates == []


def test_wilson_interval_remains_non_degenerate_at_perfect_accuracy() -> None:
    low, high = wilson_score_interval(12, 12)
    assert np.isclose(high, 1.0)
    assert 0.70 < low < 0.80
