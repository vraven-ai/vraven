"""Geometry of internal-path stability under aligned perturbations."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import (
    cosine_similarity,
    linear_cka,
    neighbourhood_overlap,
    principal_angle_distance,
)


def analyse_stability_geometry(
    reference: TraceDataset,
    perturbed: TraceDataset,
    *,
    collapse_threshold: float = 0.50,
    neighbours: int = 10,
) -> AnalysisResult:
    """Locate the first depth at which an internal path loses geometric stability."""

    if reference.sample_ids != perturbed.sample_ids:
        raise ValueError("Stability geometry requires identical ordered sample_ids")
    if not 0 <= collapse_threshold <= 1:
        raise ValueError("collapse_threshold must lie in [0, 1]")
    shared = [layer for layer in reference.layers if layer in perturbed.representations]
    if not shared:
        raise ValueError("No shared representation layers were found")
    rows = []
    for layer in shared:
        left = reference.representations[layer]
        right = perturbed.representations[layer]
        if left.shape != right.shape:
            raise ValueError(f"Aligned representation shape changed for {layer}")
        paired_cosines = [cosine_similarity(x, y) for x, y in zip(left, right, strict=True)]
        relative_l2 = np.linalg.norm(left - right, axis=1) / np.maximum(
            np.linalg.norm(left, axis=1), 1e-12
        )
        cka = linear_cka(left, right)
        neighbourhood = neighbourhood_overlap(left, right, neighbours=neighbours)
        angle = principal_angle_distance(left, right)
        stability = (
            0.30 * max(float(np.mean(paired_cosines)), 0.0)
            + 0.30 * cka
            + 0.25 * neighbourhood
            + 0.15 * (1.0 - angle)
        )
        rows.append(
            {
                "layer": layer,
                "mean_paired_cosine": float(np.mean(paired_cosines)),
                "minimum_paired_cosine": float(np.min(paired_cosines)),
                "mean_relative_l2": float(np.mean(relative_l2)),
                "cka_similarity": cka,
                "neighbourhood_overlap": neighbourhood,
                "principal_angle_distance": angle,
                "stability_index": stability,
            }
        )
    collapse = next((row for row in rows if row["stability_index"] < collapse_threshold), None)
    weakest = min(rows, key=lambda row: row["stability_index"])
    return AnalysisResult(
        analysis="stability_geometry",
        metrics={
            "layers": len(rows),
            "collapse_threshold": collapse_threshold,
            "first_collapse_layer": None if collapse is None else collapse["layer"],
            "weakest_layer": weakest["layer"],
            "minimum_stability_index": weakest["stability_index"],
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "paired geometric robustness diagnostic",
            "neighbours": neighbours,
            "stability_index_calibrated": False,
        },
        warnings=[
            "The stability index is a declared composite for ranking layers and must not be read as a probability."
        ],
        limitations=[
            "Different metrics emphasise different geometric changes.",
            "A stable internal geometry can still support an incorrect output.",
        ],
    )
