"""Layer-level pathway summaries derived from target-specific influence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vraven.core.trace import NeuralTrace
from vraven.metrics.influence import layer_influence


@dataclass(frozen=True, slots=True)
class RankedLayer:
    name: str
    execution_index: int
    signed_influence: float
    absolute_influence: float


@dataclass(frozen=True, slots=True)
class PathwaySummary:
    ranked_layers: tuple[RankedLayer, ...]
    transition_layer: str | None
    net_support: float
    total_opposition: float


def rank_layers(trace: NeuralTrace, limit: int | None = None) -> tuple[RankedLayer, ...]:
    ranked = [
        RankedLayer(
            name=layer.name,
            execution_index=layer.execution_index,
            signed_influence=float(layer.activation_gradient_signed or 0.0),
            absolute_influence=float(layer.activation_gradient_influence or 0.0),
        )
        for layer in trace.layers
        if layer.activation_gradient_influence is not None
    ]
    ranked.sort(key=lambda item: item.absolute_influence, reverse=True)
    return tuple(ranked[:limit] if limit is not None else ranked)


def decision_transition_layer(trace: NeuralTrace) -> str | None:
    """First layer after which cumulative first-order evidence remains non-negative."""

    influence = layer_influence(trace, normalise=True)
    if influence.signed.size == 0:
        return None
    cumulative = np.cumsum(influence.signed)
    for index, value in enumerate(cumulative):
        if value >= 0 and np.all(cumulative[index:] >= 0):
            return influence.names[index]
    return None


def pathway_summary(trace: NeuralTrace) -> PathwaySummary:
    influence = layer_influence(trace, normalise=False)
    return PathwaySummary(
        ranked_layers=rank_layers(trace),
        transition_layer=decision_transition_layer(trace),
        net_support=float(influence.signed.sum()) if influence.signed.size else 0.0,
        total_opposition=float(influence.opposition.sum()) if influence.opposition.size else 0.0,
    )
