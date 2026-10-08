"""Agreement metrics for comparing independent explanation vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AgreementMatrices:
    labels: tuple[str, ...]
    rank_correlation: np.ndarray
    sign_agreement: np.ndarray
    topk_overlap: np.ndarray


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Explanation vectors must have the same shape")
    if a.size < 2:
        return 1.0
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return 1.0 if np.allclose(rank_a, rank_b) else 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def sign_agreement(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Explanation vectors must have the same shape")
    active = (np.abs(a) > epsilon) | (np.abs(b) > epsilon)
    if not active.any():
        return 1.0
    return float(np.mean(np.sign(a[active]) == np.sign(b[active])))


def topk_overlap(left: Sequence[float], right: Sequence[float], k: int | None = None) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Explanation vectors must have the same shape")
    if a.size == 0:
        return 1.0
    resolved_k = k or max(1, int(np.ceil(a.size * 0.1)))
    resolved_k = min(resolved_k, a.size)
    top_a = set(np.argpartition(np.abs(a), -resolved_k)[-resolved_k:].tolist())
    top_b = set(np.argpartition(np.abs(b), -resolved_k)[-resolved_k:].tolist())
    return len(top_a & top_b) / resolved_k


def agreement_matrices(
    explanations: Mapping[str, Sequence[float]],
    *,
    top_k: int | None = None,
) -> AgreementMatrices:
    if len(explanations) < 2:
        raise ValueError("At least two explanation methods are required")
    labels = tuple(explanations)
    vectors = [np.asarray(explanations[label], dtype=np.float64).reshape(-1) for label in labels]
    shape = vectors[0].shape
    if any(vector.shape != shape for vector in vectors):
        raise ValueError("All explanation vectors must have the same shape")
    count = len(labels)
    rank = np.eye(count, dtype=np.float64)
    sign = np.eye(count, dtype=np.float64)
    overlap = np.eye(count, dtype=np.float64)
    for row in range(count):
        for column in range(row + 1, count):
            rank_value = spearman_correlation(vectors[row], vectors[column])
            sign_value = sign_agreement(vectors[row], vectors[column])
            overlap_value = topk_overlap(vectors[row], vectors[column], k=top_k)
            rank[row, column] = rank[column, row] = rank_value
            sign[row, column] = sign[column, row] = sign_value
            overlap[row, column] = overlap[column, row] = overlap_value
    return AgreementMatrices(labels, rank, sign, overlap)
