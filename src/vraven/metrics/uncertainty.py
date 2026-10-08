"""Predictive and explanation uncertainty helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vraven.core.trace import NeuralTrace
from vraven.metrics.influence import layer_influence, pathway_concentration


@dataclass(frozen=True, slots=True)
class ConflictProfile:
    predictive_uncertainty: float | None
    target_margin_uncertainty: float | None
    internal_opposition: float | None
    pathway_concentration: float | None
    causal_sensitivity: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "predictive_uncertainty": self.predictive_uncertainty,
            "target_margin_uncertainty": self.target_margin_uncertainty,
            "internal_opposition": self.internal_opposition,
            "pathway_concentration": self.pathway_concentration,
            "causal_sensitivity": self.causal_sensitivity,
        }


def internal_opposition_ratio(trace: NeuralTrace) -> float | None:
    influence = layer_influence(trace, normalise=False)
    if influence.signed.size == 0:
        return None
    support = influence.support.sum()
    opposition = influence.opposition.sum()
    total = support + opposition
    if total <= 0:
        return 0.0
    return float(opposition / total)


def causal_sensitivity(trace: NeuralTrace) -> float | None:
    if not trace.causal_interventions:
        return None
    deltas = [abs(item.score_delta) for item in trace.causal_interventions]
    baseline = [abs(item.baseline_target_score) for item in trace.causal_interventions]
    normalised = [delta / max(base, 1e-8) for delta, base in zip(deltas, baseline, strict=True)]
    return float(np.clip(np.mean(normalised), 0.0, 1.0))


def conflict_profile(trace: NeuralTrace, sample_index: int = 0) -> ConflictProfile:
    prediction = trace.prediction
    predictive = None
    margin_uncertainty = None
    if prediction is not None:
        if prediction.predictive_entropy and sample_index < len(prediction.predictive_entropy):
            predictive = float(np.clip(prediction.predictive_entropy[sample_index], 0.0, 1.0))
        if prediction.margin and sample_index < len(prediction.margin):
            margin_uncertainty = float(np.clip(1.0 - prediction.margin[sample_index], 0.0, 1.0))
    return ConflictProfile(
        predictive_uncertainty=predictive,
        target_margin_uncertainty=margin_uncertainty,
        internal_opposition=internal_opposition_ratio(trace),
        pathway_concentration=pathway_concentration(trace),
        causal_sensitivity=causal_sensitivity(trace),
    )
