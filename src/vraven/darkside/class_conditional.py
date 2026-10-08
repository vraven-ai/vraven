"""Class-conditional internal representation and community comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import energy_distance, hedges_g, linear_cka, mmd2_rbf


def compare_class_conditionals(
    dataset: TraceDataset,
    *,
    class_a: Any,
    class_b: Any,
    label_source: str = "true",
    top_units: int = 20,
) -> AnalysisResult:
    """Compare internal states for two declared classes at every shared layer."""

    if label_source not in {"true", "predicted"}:
        raise ValueError("label_source must be 'true' or 'predicted'")
    if class_a == class_b:
        raise ValueError("class_a and class_b must be different")
    if top_units < 1:
        raise ValueError("top_units must be positive")
    labels = dataset.true_labels if label_source == "true" else dataset.predicted_labels
    mask_a = labels == class_a
    mask_b = labels == class_b
    if np.sum(mask_a) < 2 or np.sum(mask_b) < 2:
        raise ValueError("Each class requires at least two samples")

    layer_rows = []
    unit_rows = []
    for layer, representation in dataset.representations.items():
        a = representation[mask_a]
        b = representation[mask_b]
        effects = hedges_g(a, b)
        combined_mask = mask_a | mask_b
        combined = representation[combined_mask]
        binary = (labels[combined_mask] == class_b).astype(np.float64)
        label_matrix = np.column_stack([1.0 - binary, binary])
        layer_rows.append(
            {
                "layer": layer,
                "samples_a": int(np.sum(mask_a)),
                "samples_b": int(np.sum(mask_b)),
                "label_cka": linear_cka(combined, label_matrix),
                "energy_distance": energy_distance(a, b),
                "mmd2_rbf": mmd2_rbf(a, b),
                "mean_absolute_hedges_g": float(np.nanmean(np.abs(effects))),
                "maximum_absolute_hedges_g": float(np.nanmax(np.abs(effects))),
            }
        )

        units = dataset.unit_influences.get(layer)
        if units is None or units.shape[1] < 1:
            continue
        unit_effects = hedges_g(units[mask_a], units[mask_b])
        mean_a = units[mask_a].mean(axis=0)
        mean_b = units[mask_b].mean(axis=0)
        order = np.argsort(np.nan_to_num(np.abs(unit_effects), nan=-1.0))[::-1][:top_units]
        for index in order:
            unit_rows.append(
                {
                    "layer": layer,
                    "unit": int(index),
                    "mean_class_a": float(mean_a[index]),
                    "mean_class_b": float(mean_b[index]),
                    "hedges_g": float(unit_effects[index]),
                    "sign_flip": bool(np.sign(mean_a[index]) != np.sign(mean_b[index])),
                }
            )

    strongest = max(layer_rows, key=lambda row: row["energy_distance"])
    return AnalysisResult(
        analysis="class_conditional_internals",
        metrics={
            "class_a": class_a,
            "class_b": class_b,
            "label_source": label_source,
            "layers_compared": len(layer_rows),
            "largest_distribution_shift_layer": strongest["layer"],
            "largest_energy_distance": strongest["energy_distance"],
            "reported_unit_flips": sum(bool(row["sign_flip"]) for row in unit_rows),
        },
        tables={"layers": layer_rows, "units": unit_rows},
        metadata={"claim_type": "cohort-level observational comparison"},
        limitations=[
            "Class separation does not establish causal use of a representation.",
            "The representation sketches may compress or omit fine-grained structure.",
        ],
    )


def map_class_conditional_unit_field(
    dataset: TraceDataset,
    *,
    class_a: Any,
    class_b: Any,
    label_source: str = "true",
    max_units_per_layer: int | None = 128,
) -> AnalysisResult:
    """Dense unit-field map of class contrast for every shared influence layer.

    Unlike ``compare_class_conditionals``, this keeps a full (or capped) unit
    grid so the figure can show a scientific layer×unit field rather than a
    top-k bar summary.
    """

    if label_source not in {"true", "predicted"}:
        raise ValueError("label_source must be 'true' or 'predicted'")
    if class_a == class_b:
        raise ValueError("class_a and class_b must be different")
    if max_units_per_layer is not None and max_units_per_layer < 1:
        raise ValueError("max_units_per_layer must be positive or None")

    labels = dataset.true_labels if label_source == "true" else dataset.predicted_labels
    mask_a = labels == class_a
    mask_b = labels == class_b
    if np.sum(mask_a) < 2 or np.sum(mask_b) < 2:
        raise ValueError("Each class requires at least two samples")

    field_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for layer, units in dataset.unit_influences.items():
        if units.ndim != 2 or units.shape[1] < 1:
            continue
        effects = hedges_g(units[mask_a], units[mask_b])
        mean_a = units[mask_a].mean(axis=0)
        mean_b = units[mask_b].mean(axis=0)
        width = int(units.shape[1])
        if max_units_per_layer is None or width <= max_units_per_layer:
            indices = np.arange(width)
        else:
            indices = np.argsort(np.nan_to_num(np.abs(effects), nan=-1.0))[::-1][
                :max_units_per_layer
            ]
            indices = np.sort(indices)
        for unit in indices:
            field_rows.append(
                {
                    "layer": layer,
                    "unit": int(unit),
                    "mean_class_a": float(mean_a[unit]),
                    "mean_class_b": float(mean_b[unit]),
                    "hedges_g": float(effects[unit]),
                    "abs_hedges_g": float(abs(effects[unit])),
                    "sign_flip": bool(np.sign(mean_a[unit]) != np.sign(mean_b[unit])),
                }
            )
        selected = effects[indices]
        layer_rows.append(
            {
                "layer": layer,
                "units_reported": int(len(indices)),
                "units_total": width,
                "mean_absolute_hedges_g": float(np.nanmean(np.abs(selected))),
                "maximum_absolute_hedges_g": float(np.nanmax(np.abs(selected))),
                "sign_flips": int(
                    np.sum(np.sign(mean_a[indices]) != np.sign(mean_b[indices]))
                ),
            }
        )

    if not field_rows:
        raise ValueError("No unit_influences were available to map a class-conditional field")

    strongest = max(layer_rows, key=lambda row: row["maximum_absolute_hedges_g"])
    return AnalysisResult(
        analysis="class_conditional_unit_field",
        metrics={
            "class_a": class_a,
            "class_b": class_b,
            "label_source": label_source,
            "layers_mapped": len(layer_rows),
            "units_mapped": len(field_rows),
            "largest_unit_contrast_layer": strongest["layer"],
            "largest_absolute_hedges_g": strongest["maximum_absolute_hedges_g"],
            "sign_flips": sum(int(row["sign_flip"]) for row in field_rows),
        },
        tables={"field": field_rows, "layers": layer_rows},
        metadata={
            "claim_type": "cohort-level observational unit field",
            "max_units_per_layer": max_units_per_layer,
        },
        limitations=[
            "The field is observational class contrast, not a causal circuit map.",
            "Capping units by |Hedges g| can hide weaker but structurally important units.",
        ],
    )
