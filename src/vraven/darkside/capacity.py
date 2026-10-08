"""Unused capacity, dormant sensitivity and overparameterisation signatures."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import effective_rank, participation_ratio, stable_rank


def audit_unused_capacity(
    dataset: TraceDataset,
    *,
    activity_threshold: float = 1e-6,
    low_activity_fraction: float = 0.95,
    high_impact_quantile: float = 0.90,
) -> AnalysisResult:
    """Classify units from dataset-level activity and target influence profiles."""

    if activity_threshold < 0:
        raise ValueError("activity_threshold cannot be negative")
    if not 0 < low_activity_fraction <= 1:
        raise ValueError("low_activity_fraction must lie in (0, 1]")
    if not 0 < high_impact_quantile < 1:
        raise ValueError("high_impact_quantile must lie in (0, 1)")
    unit_rows = []
    layer_rows = []
    for layer in dataset.layers:
        representation = dataset.representations[layer]
        layer_rows.append(
            {
                "layer": layer,
                "effective_rank": effective_rank(representation),
                "stable_rank": stable_rank(representation),
                "participation_ratio": participation_ratio(representation),
                "feature_width": representation.shape[1],
            }
        )
        energy = dataset.unit_energies.get(layer)
        influence = dataset.unit_influences.get(layer)
        if energy is None or influence is None:
            continue
        width = min(energy.shape[1], influence.shape[1])
        energy = energy[:, :width]
        influence = influence[:, :width]
        active_fraction = np.mean(energy > activity_threshold, axis=0)
        mean_energy = np.mean(energy, axis=0)
        mean_impact = np.mean(np.abs(influence), axis=0)
        impact_threshold = float(np.quantile(mean_impact, high_impact_quantile))
        for index in range(width):
            low_activity = active_fraction[index] <= (1.0 - low_activity_fraction)
            high_impact = mean_impact[index] >= impact_threshold and mean_impact[index] > 0
            if low_activity and high_impact:
                category = "dormant_sensitive"
            elif low_activity:
                category = "dead_or_near_dead"
            elif high_impact:
                category = "critical_active"
            else:
                category = "active_low_impact"
            unit_rows.append(
                {
                    "layer": layer,
                    "unit": index,
                    "active_fraction": float(active_fraction[index]),
                    "mean_energy": float(mean_energy[index]),
                    "mean_absolute_influence": float(mean_impact[index]),
                    "category": category,
                }
            )
    categories = {}
    for row in unit_rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    most_compressed = min(
        layer_rows,
        key=lambda row: row["effective_rank"] / max(row["feature_width"], 1),
    )
    return AnalysisResult(
        analysis="unused_capacity",
        metrics={
            "layers": len(layer_rows),
            "units_audited": len(unit_rows),
            "category_counts": categories,
            "lowest_rank_fraction_layer": most_compressed["layer"],
            "lowest_effective_rank_fraction": most_compressed["effective_rank"]
            / max(most_compressed["feature_width"], 1),
        },
        tables={"layers": layer_rows, "units": unit_rows},
        metadata={
            "claim_type": "dataset-level activity and local-influence audit",
            "activity_threshold": activity_threshold,
            "low_activity_fraction": low_activity_fraction,
            "high_impact_quantile": high_impact_quantile,
        },
        limitations=[
            "Near-zero activation on the audit dataset does not prove a unit is globally unused.",
            "First-order influence may miss interaction-dependent units.",
            "Representation sketches can underestimate the rank of the original hidden state.",
        ],
    )
