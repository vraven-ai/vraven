"""Layerwise representation drift across checkpoints, periods or data domains."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import (
    effective_rank,
    linear_cka,
    median_bandwidth,
    mmd2_rbf,
    orthogonal_procrustes_distance,
    participation_ratio,
    sliced_wasserstein,
    stable_rank,
)


def analyse_representation_drift(
    reference: TraceDataset,
    comparison: TraceDataset,
    *,
    sliced_projections: int = 128,
    seed: int = 42,
) -> AnalysisResult:
    """Compare aligned representations using complementary geometric measures."""

    if reference.sample_ids != comparison.sample_ids:
        raise ValueError("Representation drift requires identical ordered sample_ids")
    layers = [layer for layer in reference.layers if layer in comparison.representations]
    if not layers:
        raise ValueError("No shared representation layers were found")
    rows = []
    for layer in layers:
        left = reference.representations[layer]
        right = comparison.representations[layer]
        if left.shape != right.shape:
            raise ValueError(f"Aligned representation shape changed for {layer}")
        cka = float(np.clip(linear_cka(left, right), 0.0, 1.0))
        procrustes = float(np.clip(orthogonal_procrustes_distance(left, right), 0.0, 1.0))
        mmd = mmd2_rbf(left, right)
        swd = sliced_wasserstein(left, right, projections=sliced_projections, seed=seed)
        distance_scale = max(median_bandwidth(left, right), 1e-12)
        swd_relative = swd / distance_scale
        rows.append(
            {
                "layer": layer,
                "cka_similarity": cka,
                "procrustes_distance": procrustes,
                "mmd2_rbf": mmd,
                "mmd2_rbf_bounded": float(np.clip(mmd / 2.0, 0.0, 1.0)),
                "sliced_wasserstein": swd,
                "sliced_wasserstein_scale": distance_scale,
                "sliced_wasserstein_relative": swd_relative,
                "sliced_wasserstein_bounded": swd_relative / (1.0 + swd_relative),
                "effective_rank_reference": effective_rank(left),
                "effective_rank_comparison": effective_rank(right),
                "stable_rank_reference": stable_rank(left),
                "stable_rank_comparison": stable_rank(right),
                "participation_ratio_reference": participation_ratio(left),
                "participation_ratio_comparison": participation_ratio(right),
            }
        )
    for row in rows:
        row["rank_change"] = row["effective_rank_comparison"] - row["effective_rank_reference"]
        row["drift_index"] = (
            0.35 * (1.0 - row["cka_similarity"])
            + 0.25 * row["procrustes_distance"]
            + 0.20 * row["mmd2_rbf_bounded"]
            + 0.20 * row["sliced_wasserstein_bounded"]
        )
    maximum = max(rows, key=lambda row: row["drift_index"])
    return AnalysisResult(
        analysis="representation_drift",
        metrics={
            "shared_layers": len(rows),
            "maximum_drift_layer": maximum["layer"],
            "maximum_drift_index": maximum["drift_index"],
            "minimum_cka": min(row["cka_similarity"] for row in rows),
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "paired representation geometry comparison",
            "sample_alignment": "exact sample_ids",
            "sliced_projections": sliced_projections,
            "seed": seed,
            "drift_index_calibrated": False,
            "drift_index_components": "dimensionless bounded CKA, Procrustes, RBF-MMD and relative sliced-Wasserstein terms",
        },
        warnings=[
            "The composite drift index is a transparent ranking heuristic, not a calibrated probability."
        ],
        limitations=[
            "CKA can remain high despite changes that matter to a particular decision.",
            "Distribution distances depend on representation scaling and sample coverage.",
        ],
    )
