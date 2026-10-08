"""Empirical multiplicity and stability of repeatedly discovered circuits.

The same behavioural criterion can admit several sparse internal mechanisms.
This module treats that non-uniqueness as a measurable result rather than
silently selecting one convenient circuit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np

from .contracts import AnalysisResult


def _normalise_masks(
    circuit_masks: Mapping[str, object] | object,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return named binary circuit masks as a two-dimensional array."""

    if isinstance(circuit_masks, Mapping):
        if len(circuit_masks) < 2:
            raise ValueError("At least two discovered circuits are required")
        names = tuple(str(name) for name in circuit_masks)
        rows = [np.asarray(values).reshape(-1) for values in circuit_masks.values()]
        widths = {row.size for row in rows}
        if len(widths) != 1:
            raise ValueError("All circuit masks must use the same component ordering and width")
        masks = np.vstack(rows)
    else:
        masks = np.asarray(circuit_masks)
        if masks.ndim != 2:
            raise ValueError("circuit_masks must be a two-dimensional runs-by-components matrix")
        names = tuple(f"run_{index}" for index in range(masks.shape[0]))

    if masks.shape[0] < 2 or masks.shape[1] < 1:
        raise ValueError("At least two circuits and one component are required")
    if not np.isfinite(masks.astype(np.float64)).all():
        raise ValueError("Circuit masks contain non-finite values")

    numeric = masks.astype(np.float64)
    if not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValueError("Circuit masks must contain only binary inclusion values")
    return names, numeric.astype(bool)


def _wilson_interval(successes: np.ndarray, trials: int, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for Bernoulli component inclusion probability."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (probability + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return np.clip(centre - half_width, 0.0, 1.0), np.clip(centre + half_width, 0.0, 1.0)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def analyse_circuit_multiplicity(
    circuit_masks: Mapping[str, object] | object,
    *,
    component_labels: Sequence[str] | None = None,
    confidence: float = 0.95,
    stable_threshold: float = 0.80,
) -> AnalysisResult:
    """Measure non-uniqueness and empirical stability across discovered circuits.

    Parameters
    ----------
    circuit_masks:
        Binary matrix with one discovered circuit per row and one aligned model
        component per column, or a mapping from run name to one binary mask.
    component_labels:
        Optional labels in exactly the same component order as the masks.
    confidence:
        Confidence level for Wilson intervals over inclusion frequency.
    stable_threshold:
        A component belongs to the empirical stable core only when the lower
        Wilson bound is at least this threshold.

    Notes
    -----
    This is an empirical repeated-discovery audit. It is not a formal
    verification certificate over a continuous input domain or dataset edit
    neighbourhood.
    """

    if not 0 < stable_threshold <= 1:
        raise ValueError("stable_threshold must lie in (0, 1]")
    names, masks = _normalise_masks(circuit_masks)
    runs, width = masks.shape

    labels = (
        tuple(str(value) for value in component_labels)
        if component_labels is not None
        else tuple(f"component_{index}" for index in range(width))
    )
    if len(labels) != width:
        raise ValueError("component_labels length does not match circuit width")
    if len(set(labels)) != len(labels):
        raise ValueError("component_labels must be unique")

    successes = masks.sum(axis=0).astype(np.float64)
    inclusion = successes / runs
    interval_low, interval_high = _wilson_interval(successes, runs, confidence)

    with np.errstate(divide="ignore", invalid="ignore"):
        inclusion_entropy = -(
            np.where(inclusion > 0, inclusion * np.log2(inclusion), 0.0)
            + np.where(inclusion < 1, (1.0 - inclusion) * np.log2(1.0 - inclusion), 0.0)
        )

    pairwise = np.eye(runs, dtype=np.float64)
    off_diagonal: list[float] = []
    for left in range(runs):
        for right in range(left + 1, runs):
            value = _jaccard(masks[left], masks[right])
            pairwise[left, right] = value
            pairwise[right, left] = value
            off_diagonal.append(value)

    co_inclusion = masks.astype(np.float64).T @ masks.astype(np.float64) / runs
    unique_count = int(np.unique(masks.astype(np.uint8), axis=0).shape[0])
    stable_core = interval_low >= stable_threshold
    threshold_ambiguous = (interval_low < stable_threshold) & (interval_high >= stable_threshold)
    circuit_sizes = masks.sum(axis=1).astype(np.float64)
    mean_jaccard = float(np.mean(off_diagonal)) if off_diagonal else 1.0

    component_rows = []
    for index, label in enumerate(labels):
        component_rows.append(
            {
                "component_index": index,
                "component": label,
                "included_runs": int(successes[index]),
                "inclusion_probability": float(inclusion[index]),
                "wilson_low": float(interval_low[index]),
                "wilson_high": float(interval_high[index]),
                "inclusion_entropy_bits": float(inclusion_entropy[index]),
                "stable_core": bool(stable_core[index]),
                "threshold_ambiguous": bool(threshold_ambiguous[index]),
            }
        )

    return AnalysisResult(
        analysis="circuit_multiplicity",
        metrics={
            "runs": runs,
            "components": width,
            "unique_circuits": unique_count,
            "mean_circuit_size": float(circuit_sizes.mean()),
            "circuit_size_standard_deviation": float(circuit_sizes.std(ddof=1)) if runs > 1 else 0.0,
            "mean_pairwise_jaccard": mean_jaccard,
            "multiplicity_index": 1.0 - mean_jaccard,
            "mean_component_entropy_bits": float(inclusion_entropy.mean()),
            "stable_core_size": int(stable_core.sum()),
            "threshold_ambiguous_components": int(threshold_ambiguous.sum()),
        },
        tables={
            "components": component_rows,
            "run_names": list(names),
            "run_sizes": circuit_sizes.astype(int).tolist(),
            "pairwise_jaccard": pairwise.tolist(),
            "co_inclusion": co_inclusion.tolist(),
            "stable_core_indices": np.flatnonzero(stable_core).astype(int).tolist(),
        },
        metadata={
            "claim_type": "empirical repeated-circuit stability and multiplicity",
            "confidence": confidence,
            "interval": "Wilson score interval",
            "stable_threshold": stable_threshold,
            "component_alignment": "caller-declared fixed ordering",
        },
        warnings=[
            "Different valid circuits can reflect genuine mechanism multiplicity, search instability, grouping choices or intervention artefacts."
        ],
        limitations=[
            "The stable core is empirical and depends on the supplied discovery runs.",
            "This analysis is not a formal robustness or minimality certificate.",
            "Component identities must be genuinely aligned across all runs.",
        ],
    )
