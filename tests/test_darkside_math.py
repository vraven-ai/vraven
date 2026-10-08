from __future__ import annotations

import numpy as np

from vraven.darkside.math_utils import (
    effective_rank,
    energy_distance,
    hedges_g,
    kendall_tau_b,
    linear_cka,
    mmd2_rbf,
    neighbourhood_overlap,
    orthogonal_procrustes_distance,
    pareto_frontier,
    sign_agreement,
    sliced_wasserstein,
    spearman_correlation,
    topk_jaccard,
)


def test_linear_cka_is_invariant_to_orthogonal_feature_rotation() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(50, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    rotated = values @ q
    assert linear_cka(values, rotated) > 0.999999
    assert orthogonal_procrustes_distance(values, rotated) < 1e-10


def test_distribution_distances_are_zero_for_identical_sets() -> None:
    values = np.arange(24, dtype=float).reshape(8, 3)
    assert energy_distance(values, values) == 0.0
    assert mmd2_rbf(values, values) < 1e-12
    assert sliced_wasserstein(values, values, projections=16) < 1e-12


def test_rank_and_sign_agreement_metrics() -> None:
    left = np.array([1.0, -4.0, 3.0, 2.0])
    right = left.copy()
    assert spearman_correlation(left, right) > 0.999
    assert kendall_tau_b(left, right) > 0.999
    assert topk_jaccard(left, right, k=2) == 1.0
    assert sign_agreement(left, right) == 1.0


def test_effective_rank_and_hedges_g_are_finite() -> None:
    rng = np.random.default_rng(8)
    left = rng.normal(size=(30, 5))
    right = rng.normal(loc=0.7, size=(35, 5))
    effects = hedges_g(left, right)
    assert effects.shape == (5,)
    assert np.isfinite(effects).all()
    assert 0.0 < effective_rank(left) <= 5.0


def test_neighbourhood_and_pareto_frontier() -> None:
    values = np.arange(30, dtype=float).reshape(10, 3)
    assert neighbourhood_overlap(values, values, neighbours=3) == 1.0
    mask = pareto_frontier([1, 2, 3], [1, 2, 1])
    assert mask.tolist() == [True, True, False]
