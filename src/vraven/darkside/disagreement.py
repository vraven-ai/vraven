"""Cross-method explanation disagreement as a first-class diagnostic."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contracts import AnalysisResult
from .math_utils import (
    kendall_tau_b,
    sign_agreement,
    spearman_correlation,
    topk_jaccard,
)


def analyse_method_disagreement(
    attributions: Mapping[str, object],
    *,
    top_k: int = 10,
    sign_threshold: float = 0.0,
) -> AnalysisResult:
    """Compare methods on the same indexed explanatory surface."""

    if len(attributions) < 2:
        raise ValueError("At least two explanation methods are required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if sign_threshold < 0:
        raise ValueError("sign_threshold cannot be negative")
    names = tuple(attributions)
    vectors = {name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in attributions.items()}
    widths = {vector.size for vector in vectors.values()}
    if len(widths) != 1:
        raise ValueError("All explanation vectors must use the same indexed surface")
    if not all(np.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("Explanation vectors contain non-finite values")
    width = widths.pop()
    if width < 2:
        raise ValueError("Explanation vectors need at least two elements")

    matrices = {
        "spearman": np.eye(len(names)),
        "kendall_tau_b": np.eye(len(names)),
        "topk_jaccard": np.eye(len(names)),
        "sign_agreement": np.eye(len(names)),
    }
    pair_rows = []
    for i, left_name in enumerate(names):
        for j in range(i + 1, len(names)):
            right_name = names[j]
            left, right = vectors[left_name], vectors[right_name]
            values = {
                "spearman": spearman_correlation(left, right),
                "kendall_tau_b": kendall_tau_b(left, right),
                "topk_jaccard": topk_jaccard(left, right, k=top_k),
                "sign_agreement": sign_agreement(left, right, threshold=sign_threshold),
            }
            for metric, value in values.items():
                matrices[metric][i, j] = matrices[metric][j, i] = value
            pair_rows.append({"method_a": left_name, "method_b": right_name, **values})

    absolute_ranks = []
    signs = []
    for name in names:
        vector = vectors[name]
        order = np.argsort(np.argsort(-np.abs(vector), kind="mergesort"), kind="mergesort") + 1
        absolute_ranks.append(order)
        signs.append(np.sign(vector))
    rank_matrix = np.vstack(absolute_ranks)
    sign_matrix = np.vstack(signs)
    rank_dispersion = rank_matrix.std(axis=0, ddof=0)
    positive_fraction = np.mean(sign_matrix > 0, axis=0)
    negative_fraction = np.mean(sign_matrix < 0, axis=0)
    zero_fraction = np.mean(sign_matrix == 0, axis=0)
    probabilities = np.stack([positive_fraction, negative_fraction, zero_fraction], axis=1)
    sign_entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    ) / np.log(3.0)
    consensus_rank = rank_matrix.mean(axis=0)
    element_rows = [
        {
            "index": index,
            "mean_rank": float(consensus_rank[index]),
            "rank_std": float(rank_dispersion[index]),
            "sign_entropy": float(sign_entropy[index]),
            "positive_method_fraction": float(positive_fraction[index]),
            "negative_method_fraction": float(negative_fraction[index]),
        }
        for index in range(width)
    ]
    most_disputed = max(element_rows, key=lambda row: (row["rank_std"], row["sign_entropy"]))
    mean_agreement = float(
        np.mean([row["topk_jaccard"] for row in pair_rows]) if pair_rows else 1.0
    )
    return AnalysisResult(
        analysis="cross_method_disagreement",
        metrics={
            "methods": len(names),
            "elements": width,
            "top_k": min(top_k, width),
            "mean_topk_jaccard": mean_agreement,
            "most_disputed_index": most_disputed["index"],
            "maximum_rank_std": most_disputed["rank_std"],
        },
        tables={
            "method_names": list(names),
            "matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
            "pairs": pair_rows,
            "elements": element_rows,
        },
        metadata={
            "claim_type": "method agreement diagnostic",
            "surface_alignment_required": True,
        },
        warnings=[
            "Disagreement is itself evidence about method uncertainty; agreement does not prove correctness."
        ],
        limitations=[
            "Methods may answer different formal questions even when their vectors share a shape.",
            "Rank measures discard some magnitude information.",
        ],
    )
