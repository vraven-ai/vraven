"""Target-specific first-order influence metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vraven.core.trace import NeuralTrace


@dataclass(frozen=True, slots=True)
class LayerInfluence:
    names: tuple[str, ...]
    signed: np.ndarray
    absolute: np.ndarray
    support: np.ndarray
    opposition: np.ndarray


def activation_gradient_totals(
    activation: torch.Tensor,
    gradient: torch.Tensor,
) -> dict[str, float]:
    """Calculate first-order activation-gradient interaction.

    ``sum(a * dscore/da)`` is a local linear approximation. It is not causal
    evidence and should be described as target-specific first-order influence.
    """

    product = activation.detach().float() * gradient.detach().float()
    reduced = torch.stack((product.sum(), product.abs().sum())).cpu()
    return {"signed": float(reduced[0]), "absolute": float(reduced[1])}


def layer_influence(trace: NeuralTrace, *, normalise: bool = True) -> LayerInfluence:
    names: list[str] = []
    signed_values: list[float] = []
    absolute_values: list[float] = []
    for layer in trace.layers:
        if layer.activation_gradient_signed is None:
            continue
        names.append(layer.name)
        signed_values.append(float(layer.activation_gradient_signed))
        absolute_values.append(float(layer.activation_gradient_influence or 0.0))
    signed = np.asarray(signed_values, dtype=np.float64)
    absolute = np.asarray(absolute_values, dtype=np.float64)
    if normalise and absolute.size:
        denominator = absolute.sum()
        if denominator > 0:
            signed = signed / denominator
            absolute = absolute / denominator
    return LayerInfluence(
        names=tuple(names),
        signed=signed,
        absolute=absolute,
        support=np.clip(signed, 0.0, None),
        opposition=np.clip(-signed, 0.0, None),
    )


def cumulative_evidence(trace: NeuralTrace) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    influence = layer_influence(trace, normalise=True)
    support = np.cumsum(influence.support)
    opposition = np.cumsum(influence.opposition)
    return influence.names, support, opposition


def pathway_concentration(trace: NeuralTrace) -> float | None:
    """Herfindahl concentration of absolute first-order influence in ``[0, 1]``."""

    influence = layer_influence(trace, normalise=True)
    if influence.absolute.size == 0:
        return None
    total = influence.absolute.sum()
    if total <= 0:
        return 0.0
    proportions = influence.absolute / total
    return float(np.square(proportions).sum())
