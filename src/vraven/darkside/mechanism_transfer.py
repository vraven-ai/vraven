"""Mechanism overlap and circuit distance across conditions or checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contracts import AnalysisResult
from .math_utils import cosine_similarity, spearman_correlation, topk_jaccard


def analyse_mechanism_transfer(
    effect_vectors: Mapping[str, object],
    *,
    top_k: int = 10,
) -> AnalysisResult:
    """Compare aligned intervention-effect maps across models or conditions."""

    if len(effect_vectors) < 2:
        raise ValueError("At least two effect vectors are required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    names = tuple(effect_vectors)
    vectors = {name: np.asarray(value, dtype=np.float64).reshape(-1) for name, value in effect_vectors.items()}
    widths = {len(value) for value in vectors.values()}
    if len(widths) != 1:
        raise ValueError("Effect vectors must share an aligned component index")
    width = widths.pop()
    if width < 2:
        raise ValueError("Effect vectors need at least two aligned components")
    if not all(np.isfinite(value).all() for value in vectors.values()):
        raise ValueError("Effect vectors contain non-finite values")
    matrices = {
        "signed_cosine": np.eye(len(names)),
        "spearman": np.eye(len(names)),
        "topk_jaccard": np.eye(len(names)),
        "normalised_l1_distance": np.zeros((len(names), len(names))),
    }
    rows = []
    for i, left_name in enumerate(names):
        for j in range(i + 1, len(names)):
            right_name = names[j]
            left, right = vectors[left_name], vectors[right_name]
            scale = np.sum(np.abs(left)) + np.sum(np.abs(right))
            values = {
                "signed_cosine": cosine_similarity(left, right),
                "spearman": spearman_correlation(left, right),
                "topk_jaccard": topk_jaccard(left, right, k=top_k),
                "normalised_l1_distance": float(np.sum(np.abs(left - right)) / max(scale, 1e-12)),
            }
            for metric, value in values.items():
                matrices[metric][i, j] = matrices[metric][j, i] = value
            rows.append({"condition_a": left_name, "condition_b": right_name, **values})
    top_sets = []
    count = min(top_k, width)
    for name in names:
        top_sets.append(set(np.argpartition(np.abs(vectors[name]), -count)[-count:].tolist()))
    stable_core = sorted(set.intersection(*top_sets)) if top_sets else []
    most_distant = max(rows, key=lambda row: row["normalised_l1_distance"])
    return AnalysisResult(
        analysis="mechanism_transfer",
        metrics={
            "conditions": len(names),
            "components": width,
            "stable_core_size": len(stable_core),
            "most_distant_pair": [most_distant["condition_a"], most_distant["condition_b"]],
            "maximum_normalised_l1_distance": most_distant["normalised_l1_distance"],
        },
        tables={
            "condition_names": list(names),
            "matrices": {key: value.tolist() for key, value in matrices.items()},
            "pairs": rows,
            "stable_core_indices": stable_core,
        },
        metadata={
            "claim_type": "aligned mechanism-effect comparison",
            "top_k": count,
        },
        limitations=[
            "Component identities must be genuinely aligned across compared conditions.",
            "Circuit overlap does not by itself establish functional equivalence.",
        ],
    )
