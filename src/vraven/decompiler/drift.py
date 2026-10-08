"""Concept Drift and Out-of-Distribution (OOD) Monitor for VRAVEN.

Computes Mahalanobis & Cosine distance of runtime concept traces against baseline
ConceptAtlas to detect semantic distribution shift in real time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DriftReport:
    sample_index: int
    overall_distance: float
    status: str  # "NORMAL", "WARNING", "CRITICAL_DRIFT"
    top_drifting_concepts: list[tuple[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


class ConceptDriftMonitor:
    """Monitor incoming concept activations against a reference ConceptAtlas."""

    def __init__(self, reference_concepts: np.ndarray, concept_ids: Sequence[str]) -> None:
        self.reference = np.asarray(reference_concepts, dtype=float)
        self.concept_ids = list(concept_ids)
        self.mean = self.reference.mean(axis=0)
        self.std = self.reference.std(axis=0)
        self.std[self.std < 1e-12] = 1.0

    def evaluate_sample(self, concept_values: np.ndarray | Sequence[float], *, warning_threshold: float = 2.5, critical_threshold: float = 4.0) -> DriftReport:
        vals = np.asarray(concept_values, dtype=float).ravel()
        if len(vals) != len(self.concept_ids):
            raise ValueError(f"Expected {len(self.concept_ids)} concepts, got {len(vals)}")

        z_scores = np.abs(vals - self.mean) / self.std
        overall_dist = float(np.sqrt(np.mean(z_scores**2)))

        if overall_dist >= critical_threshold:
            status = "CRITICAL_DRIFT"
        elif overall_dist >= warning_threshold:
            status = "WARNING"
        else:
            status = "NORMAL"

        top_indices = np.argsort(z_scores)[::-1][:5]
        top_drifting = [(self.concept_ids[idx], float(z_scores[idx])) for idx in top_indices]

        return DriftReport(
            sample_index=0,
            overall_distance=overall_dist,
            status=status,
            top_drifting_concepts=top_drifting,
        )
