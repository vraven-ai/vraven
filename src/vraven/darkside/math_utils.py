"""Numerically guarded mathematical utilities for DarkSide analyses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-12


def as_2d(values: Any, *, name: str = "values") -> np.ndarray:
    """Return finite observations as a two-dimensional float64 matrix."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def robust_scale(values: Any, *, axis: int = 0, clip: float | None = 8.0) -> np.ndarray:
    """Median-centre and interquartile-scale values with optional clipping."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must contain finite observations")
    if clip is not None and clip <= 0:
        raise ValueError("clip must be positive when supplied")
    median = np.median(array, axis=axis, keepdims=True)
    q1 = np.quantile(array, 0.25, axis=axis, keepdims=True)
    q3 = np.quantile(array, 0.75, axis=axis, keepdims=True)
    scale = np.where(np.abs(q3 - q1) > _EPS, q3 - q1, 1.0)
    result = (array - median) / scale
    return np.clip(result, -clip, clip) if clip is not None else result


def cosine_similarity(left: Any, right: Any) -> float:
    """Compute guarded cosine similarity for equally shaped vectors."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Cosine similarity requires equal vector shapes")
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= _EPS:
        return 1.0 if np.linalg.norm(x - y) <= _EPS else 0.0
    return float(np.dot(x, y) / denominator)


def _centre_gram(gram: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    if n < 2:
        return np.zeros_like(gram)
    centring = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    return centring @ gram @ centring


def linear_cka(left: Any, right: Any) -> float:
    """Centred kernel alignment using linear kernels.

    Rows must refer to the same ordered samples. Feature dimensions may differ.
    """

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires the same aligned samples")
    k = _centre_gram(x @ x.T)
    second_gram = _centre_gram(y @ y.T)
    denominator = float(np.linalg.norm(k, "fro") * np.linalg.norm(second_gram, "fro"))
    if denominator <= _EPS:
        return 1.0 if np.linalg.norm(k - second_gram, "fro") <= _EPS else 0.0
    return float(np.sum(k * second_gram) / denominator)


def orthogonal_procrustes_distance(left: Any, right: Any) -> float:
    """Normalised residual after centring and optimal orthogonal alignment."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape != y.shape:
        raise ValueError("Procrustes distance requires equal matrix shapes")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm <= _EPS or y_norm <= _EPS:
        return 0.0 if np.linalg.norm(x - y) <= _EPS else 1.0
    x = x / x_norm
    y = y / y_norm
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    aligned = x @ (u @ vt)
    return float(np.linalg.norm(aligned - y, "fro") / math.sqrt(2.0))


def pairwise_distances(values: Any) -> np.ndarray:
    """Return the full Euclidean distance matrix for sample rows."""

    x = as_2d(values)
    squares = np.sum(x * x, axis=1, keepdims=True)
    distances_squared = np.maximum(squares + squares.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(distances_squared)


def energy_distance(left: Any, right: Any) -> float:
    """Multivariate energy distance between empirical sample sets."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Energy distance requires equal feature dimensions")
    cross = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    within_x = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    within_y = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean()
    return float(max(2.0 * cross - within_x - within_y, 0.0))


def median_bandwidth(*arrays: Any) -> float:
    """Estimate an RBF bandwidth from positive pairwise distances."""

    if not arrays:
        raise ValueError("At least one array is required")
    joined = np.vstack([as_2d(value) for value in arrays])
    distances = pairwise_distances(joined)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > _EPS]
    return float(np.median(positive)) if positive.size else 1.0


def mmd2_rbf(left: Any, right: Any, *, bandwidth: float | None = None) -> float:
    """Biased non-negative squared maximum mean discrepancy with an RBF kernel."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD requires equal feature dimensions")
    sigma = median_bandwidth(x, y) if bandwidth is None else float(bandwidth)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("bandwidth must be finite and positive")

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-squared / (2.0 * sigma * sigma))

    value = kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
    return float(max(value, 0.0))


def sliced_wasserstein(
    left: Any,
    right: Any,
    *,
    projections: int = 128,
    seed: int = 42,
) -> float:
    """Monte Carlo sliced 1-Wasserstein distance with deterministic projections."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein requires equal feature dimensions")
    if projections < 1:
        raise ValueError("projections must be positive")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, x.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), _EPS)
    distances = []
    for direction in directions:
        px = np.sort(x @ direction)
        py = np.sort(y @ direction)
        quantiles = np.linspace(0.0, 1.0, max(len(px), len(py)))
        xq = np.interp(quantiles, np.linspace(0.0, 1.0, len(px)), px)
        yq = np.interp(quantiles, np.linspace(0.0, 1.0, len(py)), py)
        distances.append(float(np.mean(np.abs(xq - yq))))
    return float(np.mean(distances))


def effective_rank(values: Any) -> float:
    """Compute entropy-based effective rank of centred representations."""

    x = as_2d(values)
    singular = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
    power = singular * singular
    total = float(power.sum())
    if total <= _EPS:
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, _EPS))))
    return float(np.exp(entropy))


def stable_rank(values: Any) -> float:
    """Compute the Frobenius-to-spectral stable rank of centred data."""

    x = as_2d(values)
    singular = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
    if singular.size == 0 or singular[0] <= _EPS:
        return 0.0
    return float(np.sum(singular * singular) / (singular[0] ** 2))


def participation_ratio(values: Any) -> float:
    """Compute covariance-spectrum participation ratio."""

    x = as_2d(values)
    covariance = np.cov(x, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0)
    denominator = float(np.sum(eigenvalues * eigenvalues))
    if denominator <= _EPS:
        return 0.0
    return float(np.sum(eigenvalues) ** 2 / denominator)


def hedges_g(left: Any, right: Any) -> np.ndarray:
    """Bias-corrected standardised mean difference, feature by feature."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Hedges g requires equal feature dimensions")
    n_x, n_y = x.shape[0], y.shape[0]
    if n_x < 2 or n_y < 2:
        return np.full(x.shape[1], np.nan)
    variance = ((n_x - 1) * x.var(axis=0, ddof=1) + (n_y - 1) * y.var(axis=0, ddof=1))
    variance /= max(n_x + n_y - 2, 1)
    pooled = np.sqrt(np.maximum(variance, _EPS))
    degrees = n_x + n_y - 2
    correction = 1.0 - 3.0 / max(4.0 * degrees - 1.0, 1.0)
    return correction * (x.mean(axis=0) - y.mean(axis=0)) / pooled


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    """Compute Spearman rank correlation with deterministic tie ranks."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Spearman correlation requires equal vectors with at least two values")
    return cosine_similarity(_rankdata(x) - _rankdata(x).mean(), _rankdata(y) - _rankdata(y).mean())


def kendall_tau_b(left: Any, right: Any) -> float:
    """Compute Kendall's tau-b, including ties in each variable."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Kendall tau requires equal vectors with at least two values")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x) - 1):
        dx = np.sign(x[i] - x[i + 1 :])
        dy = np.sign(y[i] - y[i + 1 :])
        concordant += int(np.sum((dx * dy) > 0))
        discordant += int(np.sum((dx * dy) < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dy == 0) & (dx != 0)))
    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    return float((concordant - discordant) / denominator) if denominator > 0 else 0.0


def topk_jaccard(left: Any, right: Any, *, k: int) -> float:
    """Measure overlap between the top-k absolute entries of two vectors."""

    if k < 1:
        raise ValueError("k must be positive")
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Top-k agreement requires equal vector shapes")
    count = min(max(int(k), 1), len(x))
    a = set(np.argpartition(np.abs(x), -count)[-count:].tolist())
    b = set(np.argpartition(np.abs(y), -count)[-count:].tolist())
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def sign_agreement(left: Any, right: Any, *, threshold: float = 0.0) -> float:
    """Measure sign agreement on entries exceeding a shared magnitude threshold."""

    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Sign agreement requires equal vector shapes")
    mask = (np.abs(x) > threshold) | (np.abs(y) > threshold)
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask]))) if np.any(mask) else 1.0


def neighbourhood_overlap(left: Any, right: Any, *, neighbours: int = 10) -> float:
    """Measure average k-nearest-neighbour preservation for aligned samples."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Neighbourhood overlap requires aligned samples")
    if x.shape[0] < 2:
        return 1.0
    k = min(max(neighbours, 1), x.shape[0] - 1)
    dx = pairwise_distances(x)
    dy = pairwise_distances(y)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    overlaps = []
    for index in range(x.shape[0]):
        nx = set(np.argpartition(dx[index], k)[:k].tolist())
        ny = set(np.argpartition(dy[index], k)[:k].tolist())
        overlaps.append(len(nx & ny) / k)
    return float(np.mean(overlaps))


def principal_angle_distance(left: Any, right: Any, *, rank: int | None = None) -> float:
    """Return normalised RMS principal-angle distance between sample subspaces."""

    x = as_2d(left, name="left")
    y = as_2d(right, name="right")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Principal angles require aligned sample rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    ux, _, _ = np.linalg.svd(x, full_matrices=False)
    uy, _, _ = np.linalg.svd(y, full_matrices=False)
    maximum = min(ux.shape[1], uy.shape[1])
    if rank is not None and rank < 1:
        raise ValueError("rank must be positive when supplied")
    used = min(rank or maximum, maximum)
    if used < 1:
        return 0.0
    cosines = np.linalg.svd(ux[:, :used].T @ uy[:, :used], compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    angles = np.arccos(cosines)
    return float(np.sqrt(np.mean(angles * angles)) / (math.pi / 2.0))


def bootstrap_ci(
    values: Any,
    *,
    statistic: str = "mean",
    confidence: float = 0.95,
    repeats: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return an estimate and percentile bootstrap confidence interval."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 1 or not np.isfinite(vector).all():
        raise ValueError("Bootstrap input must contain finite values")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    function = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        estimates[index] = function(rng.choice(vector, size=vector.size, replace=True))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(function(vector)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def pareto_frontier(cost: Any, benefit: Any) -> np.ndarray:
    """Return a Boolean mask for points not dominated by lower cost and higher benefit."""

    c = np.asarray(cost, dtype=np.float64).reshape(-1)
    b = np.asarray(benefit, dtype=np.float64).reshape(-1)
    if c.shape != b.shape:
        raise ValueError("Cost and benefit must have equal lengths")
    if not np.isfinite(c).all() or not np.isfinite(b).all():
        raise ValueError("Cost and benefit must be finite")
    frontier = np.ones(len(c), dtype=bool)
    for i in range(len(c)):
        dominated = (c <= c[i]) & (b >= b[i]) & ((c < c[i]) | (b > b[i]))
        if np.any(dominated):
            frontier[i] = False
    return frontier


def empirical_percentile(reference: Any, value: float) -> float:
    """Return the mid-rank empirical percentile of a finite query value."""

    vector = np.asarray(reference, dtype=np.float64).reshape(-1)
    if vector.size < 1 or not np.isfinite(vector).all() or not np.isfinite(value):
        raise ValueError("Empirical percentile requires finite reference values and query")
    return float((np.sum(vector < value) + 0.5 * np.sum(vector == value)) / vector.size)
