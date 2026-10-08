"""Deterministic reasoning fingerprints derived from reduced traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vraven.core.trace import NeuralTrace

METRIC_NAMES = (
    "activation_mean",
    "activation_variance",
    "mean_energy",
    "sparsity",
    "activation_dispersion_entropy",
    "gradient_norm",
    "activation_gradient_signed",
    "activation_gradient_influence",
)


@dataclass(frozen=True, slots=True)
class ReasoningDNA:
    layer_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    raw_matrix: np.ndarray
    normalised_matrix: np.ndarray


def _safe(value: float | None) -> float:
    return 0.0 if value is None or not np.isfinite(value) else float(value)


def reasoning_dna(trace: NeuralTrace) -> ReasoningDNA:
    rows: list[list[float]] = []
    for layer in trace.layers:
        rows.append(
            [
                _safe(layer.activation_mean),
                _safe(layer.activation_variance),
                _safe(layer.mean_energy),
                _safe(layer.sparsity),
                _safe(layer.activation_dispersion_entropy),
                _safe(layer.gradient_norm),
                _safe(layer.activation_gradient_signed),
                _safe(layer.activation_gradient_influence),
            ]
        )
    raw = np.asarray(rows, dtype=np.float64)
    if raw.size == 0:
        normalised = raw.copy()
    else:
        median = np.median(raw, axis=0, keepdims=True)
        q1 = np.quantile(raw, 0.25, axis=0, keepdims=True)
        q3 = np.quantile(raw, 0.75, axis=0, keepdims=True)
        spread = q3 - q1
        standard = np.std(raw, axis=0, keepdims=True)
        spread = np.where(np.abs(spread) > 1e-12, spread, standard)
        spread = np.where(np.abs(spread) > 1e-12, spread, 1.0)
        normalised = np.clip((raw - median) / spread, -4.0, 4.0)
    return ReasoningDNA(
        layer_names=tuple(layer.name for layer in trace.layers),
        metric_names=METRIC_NAMES,
        raw_matrix=raw,
        normalised_matrix=normalised,
    )


def _resize_vector(values: np.ndarray, dimensions: int) -> np.ndarray:
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than zero")
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.zeros(dimensions, dtype=np.float32)
    if flat.size == dimensions:
        return flat.astype(np.float32)
    source = np.linspace(0.0, 1.0, num=flat.size)
    target = np.linspace(0.0, 1.0, num=dimensions)
    return np.interp(target, source, flat).astype(np.float32)


def reasoning_fingerprint(trace: NeuralTrace, dimensions: int = 128) -> np.ndarray:
    """Return a fixed-width deterministic vector for trace comparison."""

    dna = reasoning_dna(trace)
    return _resize_vector(dna.normalised_matrix, dimensions)


def fingerprint_similarity(left: NeuralTrace, right: NeuralTrace, dimensions: int = 128) -> float:
    first = reasoning_fingerprint(left, dimensions=dimensions).astype(np.float64)
    second = reasoning_fingerprint(right, dimensions=dimensions).astype(np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def shared_reasoning_fingerprints(
    traces: list[NeuralTrace] | tuple[NeuralTrace, ...],
    *,
    dimensions: int = 128,
    normalise: bool = True,
) -> np.ndarray:
    """Stack per-trace fingerprints for DarkSide cohort analyses."""

    if not traces:
        return np.zeros((0, dimensions), dtype=np.float64)
    matrix = np.vstack(
        [reasoning_fingerprint(trace, dimensions=dimensions) for trace in traces]
    ).astype(np.float64)
    if normalise and matrix.size:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        matrix = matrix / norms
    return matrix
