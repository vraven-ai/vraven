"""Topological signatures of internal representation geometry."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import pairwise_distances


def h0_persistence_deaths(values: np.ndarray) -> np.ndarray:
    """Return exact H0 Vietoris-Rips death times via a minimum spanning tree."""

    distances = pairwise_distances(values)
    n = distances.shape[0]
    if n <= 1:
        return np.array([], dtype=np.float64)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    minimum = distances[0].copy()
    minimum[0] = np.inf
    deaths = []
    for _ in range(n - 1):
        candidates = np.where(~visited, minimum, np.inf)
        vertex = int(np.argmin(candidates))
        edge = float(candidates[vertex])
        if not np.isfinite(edge):
            break
        deaths.append(edge)
        visited[vertex] = True
        minimum = np.minimum(minimum, distances[vertex])
        minimum[visited] = np.inf
    return np.sort(np.asarray(deaths, dtype=np.float64))


def _quantile_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    count = max(left.size, right.size, 2)
    q = np.linspace(0.0, 1.0, count)
    left_values = np.interp(q, np.linspace(0.0, 1.0, max(left.size, 1)), left if left.size else [0.0])
    right_values = np.interp(q, np.linspace(0.0, 1.0, max(right.size, 1)), right if right.size else [0.0])
    return float(np.mean(np.abs(left_values - right_values)))


def analyse_topological_shift(
    reference: TraceDataset,
    comparison: TraceDataset,
    *,
    include_h1: bool = False,
    maximum_points: int = 512,
    seed: int = 42,
) -> AnalysisResult:
    """Compare H0 persistence and optionally H1 persistence across layers."""

    if reference.sample_ids != comparison.sample_ids:
        raise ValueError("Topological shift requires identical ordered sample_ids")
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two")
    rng = np.random.default_rng(seed)
    sample_count = len(reference.sample_ids)
    indices = (
        np.sort(rng.choice(sample_count, size=maximum_points, replace=False))
        if sample_count > maximum_points
        else np.arange(sample_count)
    )
    rows = []
    h1_available = False
    for layer in reference.layers:
        if layer not in comparison.representations:
            continue
        left = reference.representations[layer][indices]
        right = comparison.representations[layer][indices]
        deaths_left = h0_persistence_deaths(left)
        deaths_right = h0_persistence_deaths(right)
        row = {
            "layer": layer,
            "h0_wasserstein_1": _quantile_distance(deaths_left, deaths_right),
            "h0_total_persistence_reference": float(deaths_left.sum()),
            "h0_total_persistence_comparison": float(deaths_right.sum()),
            "h0_max_death_reference": float(deaths_left.max()) if deaths_left.size else 0.0,
            "h0_max_death_comparison": float(deaths_right.max()) if deaths_right.size else 0.0,
        }
        if include_h1:
            try:
                from ripser import ripser

                h1_left = ripser(left, maxdim=1)["dgms"][1]
                h1_right = ripser(right, maxdim=1)["dgms"][1]
                persistence_left = (
                    h1_left[:, 1] - h1_left[:, 0] if h1_left.size else np.array([])
                )
                persistence_right = (
                    h1_right[:, 1] - h1_right[:, 0] if h1_right.size else np.array([])
                )
                persistence_left = persistence_left[np.isfinite(persistence_left)]
                persistence_right = persistence_right[np.isfinite(persistence_right)]
                row["h1_lifetime_quantile_distance"] = _quantile_distance(
                    np.sort(persistence_left), np.sort(persistence_right)
                )
                h1_available = True
            except ImportError as exc:  # pragma: no cover
                raise ImportError("include_h1=True requires the optional ripser package") from exc
        rows.append(row)
    if not rows:
        raise ValueError("No shared representation layers were found")
    strongest = max(rows, key=lambda row: row["h0_wasserstein_1"])
    return AnalysisResult(
        analysis="topological_shift",
        metrics={
            "layers": len(rows),
            "samples_used": len(indices),
            "largest_h0_shift_layer": strongest["layer"],
            "largest_h0_wasserstein_1": strongest["h0_wasserstein_1"],
            "h1_included": h1_available,
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "representation topology diagnostic",
            "h0_method": "exact minimum-spanning-tree death times; empirical 1-Wasserstein distance between sorted H0 death distributions",
            "maximum_points": maximum_points,
            "seed": seed,
        },
        limitations=[
            "Persistence depends on the chosen metric and representation scaling.",
            "The optional H1 value compares lifetime distributions, not full persistence diagrams with diagonal matching.",
            "Compressed representation sketches may change fine topological structure.",
        ],
    )
