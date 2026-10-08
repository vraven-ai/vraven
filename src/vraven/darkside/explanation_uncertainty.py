"""Uncertainty over repeated explanations, ranks and top-k membership."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .math_utils import bootstrap_ci


def analyse_explanation_uncertainty(
    repeated_attributions: object,
    *,
    top_k: int = 10,
    confidence: float = 0.95,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
    element_labels: list[str] | None = None,
) -> AnalysisResult:
    """Quantify variability across seeds, perturbations or stochastic passes."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    values = np.asarray(repeated_attributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("repeated_attributions must have shape [runs, elements] with at least two runs")
    if not np.isfinite(values).all():
        raise ValueError("repeated_attributions contains non-finite values")
    runs, width = values.shape
    if width < 1:
        raise ValueError("repeated_attributions must include at least one element")
    if element_labels is not None and len(element_labels) != width:
        raise ValueError("element_labels length must match the explanation width")
    count = min(top_k, width)
    ranks = np.empty_like(values)
    selected = np.zeros_like(values, dtype=bool)
    for run in range(runs):
        order = np.argsort(np.argsort(-np.abs(values[run]), kind="mergesort"), kind="mergesort") + 1
        ranks[run] = order
        selected[run, np.argpartition(np.abs(values[run]), -count)[-count:]] = True
    rows = []
    for index in range(width):
        centre, low, high = bootstrap_ci(
            values[:, index],
            confidence=confidence,
            repeats=bootstrap_repeats,
            seed=seed + index,
        )
        rows.append(
            {
                "index": index,
                "label": element_labels[index] if element_labels else str(index),
                "mean": float(values[:, index].mean()),
                "standard_deviation": float(values[:, index].std(ddof=1)),
                "bootstrap_centre": centre,
                "bootstrap_low": low,
                "bootstrap_high": high,
                "positive_probability": float(np.mean(values[:, index] > 0)),
                "negative_probability": float(np.mean(values[:, index] < 0)),
                "topk_selection_probability": float(np.mean(selected[:, index])),
                "mean_rank": float(np.mean(ranks[:, index])),
                "rank_standard_deviation": float(np.std(ranks[:, index], ddof=1)),
            }
        )
    most_uncertain = max(rows, key=lambda row: row["rank_standard_deviation"])
    return AnalysisResult(
        analysis="explanation_uncertainty",
        metrics={
            "runs": runs,
            "elements": width,
            "top_k": count,
            "confidence": confidence,
            "most_rank_uncertain_element": most_uncertain["label"],
            "maximum_rank_standard_deviation": most_uncertain["rank_standard_deviation"],
        },
        tables={"elements": rows},
        metadata={
            "claim_type": "empirical repeated-explanation uncertainty",
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
        },
        limitations=[
            "The uncertainty estimate covers only the declared source of repetition.",
            "Bootstrap intervals do not correct systematic explanation bias.",
        ],
    )
