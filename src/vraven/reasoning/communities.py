"""Deterministic discovery of latent activation communities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vraven.core.errors import CapabilityError
from vraven.core.trace import LayerTrace


@dataclass(frozen=True, slots=True)
class ActivationCommunity:
    community_id: int
    unit_indices: tuple[int, ...]
    centroid: tuple[float, ...]
    mean_activation: float
    mean_contribution: float | None


def _initial_centres(data: np.ndarray, clusters: int) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1)
    first = int(np.argmax(norms))
    centres = [data[first]]
    distances = np.linalg.norm(data - centres[0], axis=1)
    for _ in range(1, clusters):
        next_index = int(np.argmax(distances))
        centres.append(data[next_index])
        candidate_distance = np.linalg.norm(data - centres[-1], axis=1)
        distances = np.minimum(distances, candidate_distance)
    return np.asarray(centres)


def discover_activation_communities(
    layer: LayerTrace,
    *,
    n_clusters: int = 4,
    max_iterations: int = 100,
) -> tuple[ActivationCommunity, ...]:
    """Cluster units by their activation pattern across the captured batch.

    The result is a latent activation community. VRAVEN does not label it as a
    human concept without separate concept validation.
    """

    if not layer.unit_matrix:
        raise CapabilityError("The selected layer does not contain a batch unit matrix")
    matrix = np.asarray(layer.unit_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise CapabilityError("Community discovery requires at least two samples and two units")
    data = matrix.T
    data = data - data.mean(axis=1, keepdims=True)
    scale = data.std(axis=1, keepdims=True)
    data = data / np.where(scale > 1e-12, scale, 1.0)
    clusters = max(1, min(n_clusters, data.shape[0]))
    centres = _initial_centres(data, clusters)
    assignments = np.zeros(data.shape[0], dtype=int)
    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, None, :] - centres[None, :, :], axis=2)
        new_assignments = distances.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for index in range(clusters):
            members = data[assignments == index]
            if members.size:
                centres[index] = members.mean(axis=0)
    contributions = (
        np.asarray(layer.unit_contributions, dtype=np.float64)
        if layer.unit_contributions is not None
        else None
    )
    activation_mean = np.asarray(layer.unit_activations, dtype=np.float64)
    communities: list[ActivationCommunity] = []
    for index in range(clusters):
        members = np.flatnonzero(assignments == index)
        communities.append(
            ActivationCommunity(
                community_id=index,
                unit_indices=tuple(int(item) for item in members),
                centroid=tuple(float(item) for item in centres[index]),
                mean_activation=float(activation_mean[members].mean()) if members.size else 0.0,
                mean_contribution=(
                    float(contributions[members].mean())
                    if contributions is not None and members.size
                    else None
                ),
            )
        )
    return tuple(communities)
