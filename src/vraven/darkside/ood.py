"""High-confidence internal mechanism anomaly detection."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .math_utils import empirical_percentile


def _fit_precision(reference: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(reference)
        return estimator.location_, estimator.precision_, "Ledoit-Wolf"
    except ImportError:  # pragma: no cover
        location = reference.mean(axis=0)
        covariance = np.cov(reference, rowvar=False)
        covariance = np.atleast_2d(covariance) + ridge * np.eye(reference.shape[1])
        return location, np.linalg.pinv(covariance), "ridge-pseudoinverse"


def _mahalanobis(values: np.ndarray, location: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centred = values - location
    squared = np.einsum("ni,ij,nj->n", centred, precision, centred)
    return np.sqrt(np.maximum(squared, 0.0))


def analyse_high_confidence_ood(
    reference_fingerprints: object,
    query_fingerprints: object,
    *,
    query_confidence: object | None = None,
    confidence_threshold: float = 0.90,
    anomaly_percentile_threshold: float = 0.99,
    ridge: float = 1e-4,
) -> AnalysisResult:
    """Flag confident predictions whose internal fingerprints are atypical."""

    reference = np.asarray(reference_fingerprints, dtype=np.float64)
    query = np.asarray(query_fingerprints, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Reference and query fingerprints must be 2D with equal width")
    if query.shape[0] < 1 or reference.shape[1] < 1:
        raise ValueError("Fingerprint arrays cannot be empty")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("Fingerprint arrays contain non-finite values")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    if not 0 < anomaly_percentile_threshold < 1:
        raise ValueError("anomaly_percentile_threshold must lie in (0, 1)")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    n_reference = int(reference.shape[0])
    n_query = int(query.shape[0])
    original_width = int(reference.shape[1])
    if n_reference < 3:
        raise ValueError("At least three reference fingerprints are required")

    # When N << D, shrink the fingerprint space before Mahalanobis / Ledoit-Wolf.
    effective_dim = original_width
    pca_components = 0
    if n_reference <= original_width:
        try:
            from sklearn.decomposition import PCA

            pca_components = max(2, min(n_reference - 2, 32, original_width))
            pca = PCA(n_components=pca_components, random_state=0)
            reference = pca.fit_transform(reference)
            query = pca.transform(query)
            effective_dim = int(pca_components)
        except ImportError:  # pragma: no cover
            pca_components = 0

    location, precision, estimator = _fit_precision(reference, ridge)
    reference_scores = _mahalanobis(reference, location, precision)
    query_scores = _mahalanobis(query, location, precision)
    confidence = (
        np.asarray(query_confidence, dtype=np.float64).reshape(-1)
        if query_confidence is not None
        else np.full(query.shape[0], np.nan)
    )
    if confidence.size != query.shape[0]:
        raise ValueError("query_confidence length does not match query fingerprints")
    finite_confidence = confidence[np.isfinite(confidence)]
    if finite_confidence.size and (np.any(finite_confidence < 0) or np.any(finite_confidence > 1)):
        raise ValueError("Finite query confidence values must lie in [0, 1]")
    rows = []
    for index, score in enumerate(query_scores):
        percentile = empirical_percentile(reference_scores, float(score))
        high_confidence = bool(np.isfinite(confidence[index]) and confidence[index] >= confidence_threshold)
        rows.append(
            {
                "query_index": index,
                "mahalanobis_distance": float(score),
                "reference_percentile": percentile,
                "confidence": None if not np.isfinite(confidence[index]) else float(confidence[index]),
                "high_confidence": high_confidence,
                "high_confidence_internal_anomaly": bool(
                    high_confidence and percentile >= anomaly_percentile_threshold
                ),
            }
        )
    flagged = [row for row in rows if row["high_confidence_internal_anomaly"]]
    strongest = max(rows, key=lambda row: row["reference_percentile"])
    return AnalysisResult(
        analysis="high_confidence_ood",
        metrics={
            "reference_samples": n_reference,
            "query_samples": n_query,
            "fingerprint_width_original": original_width,
            "fingerprint_width": effective_dim,
            "pca_components": pca_components,
            "flagged_high_confidence_anomalies": len(flagged),
            "maximum_reference_percentile": strongest["reference_percentile"],
            "most_anomalous_query": strongest["query_index"],
        },
        tables={
            "queries": rows,
            "reference_distances": reference_scores.tolist(),
        },
        metadata={
            "claim_type": "internal fingerprint anomaly screening",
            "display_name": "High-confidence internal anomaly screening",
            "covariance_estimator": estimator,
            "confidence_threshold": confidence_threshold,
            "anomaly_percentile_threshold": anomaly_percentile_threshold,
            "pca_components": pca_components,
        },
        warnings=[
            "An internal anomaly is a review signal, not proof of an adversarial or out-of-distribution input.",
            "When N_reference ≤ fingerprint width, PCA is applied before covariance estimation.",
        ],
        limitations=[
            "Reference coverage determines the meaning of the empirical percentile.",
            "Mahalanobis distance assumes that a covariance summary is informative for the reduced fingerprint space.",
            "Class-separated reference/query splits can confuse class contrast with anomaly.",
        ],
    )
