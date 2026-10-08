"""Shared projection for layer-by-layer representation evolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace

TSNE_CAVEAT = (
    "Per-layer t-SNE embeddings are stochastic neighbourhood embeddings. "
    "Panel-to-panel distances are not literal Euclidean trajectories unless "
    "Procrustes alignment is applied; always read seed and alignment metadata."
)


@dataclass(frozen=True, slots=True)
class SharedProjection:
    layer_names: tuple[str, ...]
    coordinates: tuple[np.ndarray, ...]
    explained_variance_ratio: tuple[float, ...]
    sample_count: int
    projection: str = "shared_pca"
    random_seed: int | None = None
    metadata: dict[str, object] | None = None


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(count, length))
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def _try_sklearn_tsne():
    try:
        from sklearn.manifold import TSNE

        return TSNE
    except Exception:
        return None


def _pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    grams = matrix @ matrix.T
    squared = np.diag(grams)
    return np.maximum(squared[:, None] + squared[None, :] - 2.0 * grams, 0.0)


def _joint_probabilities(distances: np.ndarray, perplexity: float) -> np.ndarray:
    count = distances.shape[0]
    target = np.log(max(perplexity, 2.0))
    probabilities = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        row = distances[index].copy()
        row[index] = np.inf
        for _ in range(50):
            density = np.exp(-row * beta)
            sum_density = float(np.sum(density))
            if sum_density <= 1e-12:
                sum_density = 1e-12
            conditional = density / sum_density
            entropy = -np.sum(conditional * np.log(np.maximum(conditional, 1e-12)))
            error = entropy - target
            if abs(error) < 1e-5:
                break
            if error > 0:
                beta_min = beta
                beta = beta * 2.0 if not np.isfinite(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if not np.isfinite(beta_min) else (beta + beta_min) / 2.0
        probabilities[index] = conditional
    joint = (probabilities + probabilities.T) / (2.0 * count)
    return np.maximum(joint, 1e-12)


def _exact_tsne(
    matrix: np.ndarray,
    *,
    components: int,
    perplexity: float,
    n_iter: int,
    random_seed: int,
) -> np.ndarray:
    """Exact t-SNE for small batches when sklearn/scipy are unavailable."""

    values = np.asarray(matrix, dtype=np.float64)
    count = values.shape[0]
    if count < 3:
        raise CapabilityError("t-SNE requires a batch with at least three samples")
    effective_perplexity = min(float(perplexity), max(2.0, (count - 1) / 3.0))
    distances = _pairwise_squared_distances(values)
    joint = _joint_probabilities(distances, effective_perplexity)

    rng = np.random.default_rng(random_seed)
    # PCA init for stability.
    centred = values - values.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    retained = min(components, vectors.shape[0])
    embedding = centred @ vectors[:retained].T
    if retained < components:
        embedding = np.pad(embedding, ((0, 0), (0, components - retained)))
    embedding += rng.normal(scale=1e-4, size=embedding.shape)

    learning_rate = max(count, 50.0)
    gains = np.ones_like(embedding)
    update = np.zeros_like(embedding)
    for iteration in range(n_iter):
        # Student-t affinities in low dimensions.
        low_dist = _pairwise_squared_distances(embedding)
        affinity = 1.0 / (1.0 + low_dist)
        np.fill_diagonal(affinity, 0.0)
        affinity_sum = float(np.sum(affinity))
        if affinity_sum <= 1e-12:
            affinity_sum = 1e-12
        low_prob = np.maximum(affinity / affinity_sum, 1e-12)
        early = 4.0 if iteration < 100 else 1.0
        grad_factor = early * joint - low_prob
        gradient = np.zeros_like(embedding)
        for axis in range(components):
            delta = embedding[:, axis][:, None] - embedding[:, axis][None, :]
            gradient[:, axis] = 4.0 * np.sum(grad_factor * affinity * delta, axis=1)
        gains = (gains + 0.2) * (np.sign(gradient) != np.sign(update)) + gains * 0.8 * (
            np.sign(gradient) == np.sign(update)
        )
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if iteration < 20 else 0.8
        update = momentum * update - learning_rate * gains * gradient
        embedding = embedding + update
        embedding -= embedding.mean(axis=0, keepdims=True)
    return embedding


def _fit_tsne(
    matrix: np.ndarray,
    *,
    components: int,
    perplexity: float,
    n_iter: int,
    random_seed: int,
) -> tuple[np.ndarray, str]:
    TSNE = _try_sklearn_tsne()
    if TSNE is not None:
        try:
            embedder = TSNE(
                n_components=components,
                perplexity=perplexity,
                max_iter=max(250, int(n_iter)),
                random_state=random_seed,
                init="pca",
                learning_rate="auto",
            )
        except TypeError:
            embedder = TSNE(
                n_components=components,
                perplexity=perplexity,
                n_iter=max(250, int(n_iter)),
                random_state=random_seed,
                init="pca",
                learning_rate="auto",
            )
        return np.asarray(embedder.fit_transform(matrix), dtype=np.float64), "sklearn"
    return (
        _exact_tsne(
            matrix,
            components=components,
            perplexity=perplexity,
            n_iter=max(250, int(n_iter)),
            random_seed=random_seed,
        ),
        "numpy_exact",
    )


def _procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target coordinates to reference with translation, rotation and scaling."""

    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape or ref.ndim != 2 or ref.shape[0] < 2:
        return tgt
    ref_mean = ref.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    ref_c = ref - ref_mean
    tgt_c = tgt - tgt_mean
    ref_norm = np.linalg.norm(ref_c)
    tgt_norm = np.linalg.norm(tgt_c)
    if ref_norm <= 1e-12 or tgt_norm <= 1e-12:
        return tgt - tgt_mean + ref_mean
    matrix = tgt_c.T @ ref_c
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(np.trace(rotation.T @ matrix) / (tgt_norm**2))
    return (scale * (tgt_c @ rotation)) + ref_mean


def shared_pca(
    trace: NeuralTrace,
    *,
    layer_names: list[str] | tuple[str, ...] | None = None,
    max_layers: int = 4,
    components: int = 2,
) -> SharedProjection:
    """Fit one PCA basis across all selected layer representations."""

    available = [layer for layer in trace.layers if layer.representation]
    if layer_names is not None:
        requested = set(layer_names)
        available = [layer for layer in available if layer.name in requested]
    else:
        available = [available[index] for index in evenly_spaced_indices(len(available), max_layers)]
    if not available:
        raise CapabilityError("No compressed layer representations were captured")
    matrices = [np.asarray(layer.representation, dtype=np.float64) for layer in available]
    sample_count = matrices[0].shape[0]
    feature_count = matrices[0].shape[1]
    if sample_count < 2:
        raise CapabilityError("Representation evolution requires a batch with at least two samples")
    if any(matrix.shape != (sample_count, feature_count) for matrix in matrices):
        raise CapabilityError("Selected representations do not share a common batch and feature shape")
    stacked = np.vstack(matrices)
    centre = stacked.mean(axis=0, keepdims=True)
    centred = stacked - centre
    _, singular_values, vectors = np.linalg.svd(centred, full_matrices=False)
    retained = min(components, vectors.shape[0])
    basis = vectors[:retained].T
    projected = [(matrix - centre) @ basis for matrix in matrices]
    if retained < components:
        projected = [
            np.pad(matrix, ((0, 0), (0, components - retained)), mode="constant")
            for matrix in projected
        ]
    total_variance = np.square(singular_values).sum()
    ratios = (
        np.square(singular_values[:retained]) / total_variance
        if total_variance > 0
        else np.zeros(retained)
    )
    ratios = np.pad(ratios, (0, components - retained), mode="constant")
    return SharedProjection(
        layer_names=tuple(layer.name for layer in available),
        coordinates=tuple(projected),
        explained_variance_ratio=tuple(float(value) for value in ratios),
        sample_count=sample_count,
        projection="shared_pca",
    )


def layer_tsne(
    trace: NeuralTrace,
    *,
    layer_names: list[str] | tuple[str, ...] | None = None,
    max_layers: int = 4,
    components: int = 2,
    random_seed: int = 17,
    perplexity: float = 30.0,
    n_iter: int = 400,
    align: bool = True,
) -> SharedProjection:
    """Fit t-SNE independently per selected layer, optionally Procrustes-aligned."""

    available = [layer for layer in trace.layers if layer.representation]
    if layer_names is not None:
        requested = set(layer_names)
        available = [layer for layer in available if layer.name in requested]
    else:
        available = [available[index] for index in evenly_spaced_indices(len(available), max_layers)]
    if not available:
        raise CapabilityError("No compressed layer representations were captured")
    matrices = [np.asarray(layer.representation, dtype=np.float64) for layer in available]
    sample_count = matrices[0].shape[0]
    if sample_count < 3:
        raise CapabilityError("t-SNE requires a batch with at least three samples")
    if any(matrix.shape[0] != sample_count for matrix in matrices):
        raise CapabilityError("Selected representations do not share a common batch size")

    effective_perplexity = min(float(perplexity), max(5.0, (sample_count - 1) / 3.0))
    coordinates: list[np.ndarray] = []
    backends: list[str] = []
    for matrix in matrices:
        coords, backend = _fit_tsne(
            matrix,
            components=components,
            perplexity=effective_perplexity,
            n_iter=max(250, int(n_iter)),
            random_seed=random_seed,
        )
        coordinates.append(coords)
        backends.append(backend)

    method = "tsne_independent"
    if align and len(coordinates) > 1:
        reference = coordinates[0]
        coordinates = [reference] + [
            _procrustes_align(reference, panel) for panel in coordinates[1:]
        ]
        method = "tsne_per_layer_procrustes"

    return SharedProjection(
        layer_names=tuple(layer.name for layer in available),
        coordinates=tuple(coordinates),
        explained_variance_ratio=tuple(0.0 for _ in range(components)),
        sample_count=sample_count,
        projection=method,
        random_seed=random_seed,
        metadata={
            "perplexity": effective_perplexity,
            "n_iter": n_iter,
            "align": align,
            "backend": backends[0] if backends else "unknown",
            "caveat": TSNE_CAVEAT,
        },
    )


def dataset_tsne(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    components: int = 2,
    random_seed: int = 17,
    perplexity: float = 30.0,
    n_iter: int = 400,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project a sample×feature matrix with a single t-SNE embedding."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise CapabilityError("dataset t-SNE requires a [samples, dimensions] matrix with >= 3 rows")
    effective_perplexity = min(float(perplexity), max(5.0, (values.shape[0] - 1) / 3.0))
    coords, backend = _fit_tsne(
        values,
        components=components,
        perplexity=effective_perplexity,
        n_iter=n_iter,
        random_seed=random_seed,
    )
    return coords, {
        "projection": "dataset_tsne",
        "random_seed": random_seed,
        "perplexity": effective_perplexity,
        "n_iter": n_iter,
        "backend": backend,
        "caveat": TSNE_CAVEAT,
    }
