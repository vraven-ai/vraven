"""Explanation stability measurements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Fingerprint shapes differ: {a.shape} and {b.shape}")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= epsilon:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def sign_flip_rate(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("The compared explanations must have equal shapes")
    active = (np.abs(a) > epsilon) | (np.abs(b) > epsilon)
    if not active.any():
        return 0.0
    return float(np.mean(np.sign(a[active]) != np.sign(b[active])))


def perturbation_consistency(
    reference: Sequence[float],
    perturbations: Sequence[Sequence[float]],
) -> float:
    if not perturbations:
        raise ValueError("At least one perturbed explanation is required")
    scores = [cosine_similarity(reference, candidate) for candidate in perturbations]
    return float(np.mean(scores))
