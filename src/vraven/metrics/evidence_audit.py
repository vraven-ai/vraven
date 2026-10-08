"""Evidence concentration and observational--causal concordance audits.

The quantities in this module deliberately remain separate.  They describe
different failure modes and are not collapsed into a synthetic trust score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import factorial
from typing import Any

import numpy as np

from vraven.core.trace import NeuralTrace
from vraven.metrics.agreement import sign_agreement, spearman_correlation
from vraven.metrics.influence import layer_influence


@dataclass(frozen=True, slots=True)
class EvidenceConcentrationAudit:
    """Scale-invariant description of where first-order evidence is located."""

    layer_names: tuple[str, ...]
    signed_influence: tuple[float, ...]
    absolute_influence: tuple[float, ...]
    mass_share: tuple[float, ...]
    cumulative_mass: tuple[float, ...]
    shannon_entropy: float | None
    normalised_entropy: float | None
    effective_layers: float | None
    participation_ratio: float | None
    herfindahl_concentration: float | None
    normalised_gini: float | None
    signed_cancellation: float | None
    half_mass_layer: str | None
    half_mass_depth: float | None
    eighty_mass_layer: str | None
    eighty_mass_depth: float | None


@dataclass(frozen=True, slots=True)
class CausalConcordanceAudit:
    """Agreement between local activation-gradient evidence and zero ablation."""

    layer_names: tuple[str, ...]
    observational_raw: tuple[float, ...]
    causal_raw: tuple[float, ...]
    observational_l1: tuple[float, ...]
    causal_l1: tuple[float, ...]
    spearman_rho: float | None
    sign_agreement: float | None
    cosine_similarity: float | None
    normalised_l1_discrepancy: float | None
    permutation_p_value: float | None
    permutation_method: str | None
    excluded_nonzero_interventions: tuple[str, ...]
    excluded_ambiguous_layers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScientificEvidenceAudit:
    """Report-level evidence audit with no opaque composite score."""

    evidence_tier: str
    concentration: EvidenceConcentrationAudit
    concordance: CausalConcordanceAudit
    gradient_layer_count: int
    intervened_layer_count: int
    zero_ablation_layer_count: int
    causal_coverage: float | None
    median_predictive_entropy: float | None
    median_prediction_margin: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_vector(values: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(vector), vector, 0.0)


def _l1_normalise(values: np.ndarray) -> np.ndarray:
    denominator = float(np.abs(values).sum())
    if denominator <= 0.0:
        return np.zeros_like(values)
    return values / denominator


def _mass_crossing(
    names: tuple[str, ...], cumulative: np.ndarray, threshold: float
) -> tuple[str | None, float | None]:
    if not names or cumulative.size == 0 or cumulative[-1] <= 0.0:
        return None, None
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(names) - 1)
    return names[index], float((index + 1) / len(names))


def _normalised_gini(probabilities: np.ndarray) -> float | None:
    """Return the finite-sample-normalised Gini coefficient in ``[0, 1]``."""

    count = probabilities.size
    if count == 0 or float(probabilities.sum()) <= 0.0:
        return None
    if count == 1:
        return 0.0
    ordered = np.sort(probabilities)
    weights = np.arange(1, count + 1, dtype=np.float64)
    raw = float((2.0 * np.dot(weights, ordered) / ordered.sum() - (count + 1)) / count)
    return float(np.clip(raw * count / (count - 1), 0.0, 1.0))


def evidence_concentration(trace: NeuralTrace) -> EvidenceConcentrationAudit:
    """Audit the depth distribution of absolute first-order influence.

    With ``p_l = |a_l * grad_l|_1 / sum_j |a_j * grad_j|_1``, the effective
    number of layers is ``exp(-sum_l p_l log p_l)`` and the participation ratio
    is ``1 / sum_l p_l^2``.  Both equal the number of layers for uniform mass and
    one for a point mass.  Signed cancellation is calculated separately from the
    signed layer interactions.
    """

    influence = layer_influence(trace, normalise=False)
    names = influence.names
    signed = _finite_vector(influence.signed)
    absolute = np.clip(_finite_vector(influence.absolute), 0.0, None)
    total = float(absolute.sum())
    shares = absolute / total if total > 0.0 else np.zeros_like(absolute)
    cumulative = np.cumsum(shares)
    positive = shares[shares > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else None
    if entropy is None:
        normalised_entropy = None
        effective = None
        participation = None
        herfindahl = None
    else:
        normalised_entropy = (
            float(entropy / np.log(len(shares))) if len(shares) > 1 else 0.0
        )
        effective = float(np.exp(entropy))
        herfindahl = float(np.square(shares).sum())
        participation = float(1.0 / herfindahl) if herfindahl > 0.0 else None
    signed_mass = float(np.abs(signed).sum())
    cancellation = (
        float(np.clip(1.0 - abs(float(signed.sum())) / signed_mass, 0.0, 1.0))
        if signed_mass > 0.0
        else None
    )
    half_layer, half_depth = _mass_crossing(names, cumulative, 0.5)
    eighty_layer, eighty_depth = _mass_crossing(names, cumulative, 0.8)
    return EvidenceConcentrationAudit(
        layer_names=names,
        signed_influence=tuple(float(value) for value in signed),
        absolute_influence=tuple(float(value) for value in absolute),
        mass_share=tuple(float(value) for value in shares),
        cumulative_mass=tuple(float(value) for value in cumulative),
        shannon_entropy=entropy,
        normalised_entropy=normalised_entropy,
        effective_layers=effective,
        participation_ratio=participation,
        herfindahl_concentration=herfindahl,
        normalised_gini=_normalised_gini(shares),
        signed_cancellation=cancellation,
        half_mass_layer=half_layer,
        half_mass_depth=half_depth,
        eighty_mass_layer=eighty_layer,
        eighty_mass_depth=eighty_depth,
    )


def _permutation_significance(
    observed: np.ndarray,
    causal: np.ndarray,
    *,
    random_seed: int,
    max_permutations: int,
) -> tuple[float | None, str | None]:
    count = observed.size
    if count < 3 or np.std(observed) == 0.0 or np.std(causal) == 0.0:
        return None, None
    observed_values = observed.tolist()
    statistic = abs(spearman_correlation(observed_values, causal.tolist()))
    if count <= 8 and factorial(count) <= max_permutations:
        candidates = (np.asarray(item, dtype=np.float64) for item in permutations(causal.tolist()))
        scores = [
            abs(spearman_correlation(observed_values, candidate.tolist()))
            for candidate in candidates
        ]
        exceedances = sum(score >= statistic - 1e-12 for score in scores)
        return float(exceedances / len(scores)), f"exact ({len(scores)} permutations)"
    rng = np.random.default_rng(random_seed)
    exceedances = 0
    for _ in range(max_permutations):
        permuted = rng.permutation(causal)
        exceedances += (
            abs(spearman_correlation(observed_values, permuted.tolist())) >= statistic - 1e-12
        )
    return float((exceedances + 1) / (max_permutations + 1)), (
        f"Monte Carlo ({max_permutations} permutations; seed={random_seed})"
    )


def observational_causal_concordance(
    trace: NeuralTrace,
    *,
    random_seed: int = 17,
    max_permutations: int = 4096,
) -> CausalConcordanceAudit:
    """Compare first-order layer interactions with matched zero-ablation effects.

    Zeroing a layer suggests the local first-order score-drop approximation
    ``a_l * d s_t / d a_l``.  The comparison is therefore restricted to explicit
    ``strategy='zero'`` interventions.  Reused modules with multiple execution
    records are excluded because one module intervention cannot be assigned to a
    unique execution without additional capture semantics.
    """

    if max_permutations < 1:
        raise ValueError("max_permutations must be at least 1")
    by_base: dict[str, list[tuple[str, float]]] = {}
    for layer in trace.layers:
        if layer.activation_gradient_signed is None:
            continue
        base = str(layer.metadata.get("base_module_name", layer.name.split("#", 1)[0]))
        by_base.setdefault(base, []).append((layer.name, float(layer.activation_gradient_signed)))

    names: list[str] = []
    observed: list[float] = []
    causal: list[float] = []
    excluded_nonzero: list[str] = []
    excluded_ambiguous: list[str] = []
    seen: set[str] = set()
    for intervention in trace.causal_interventions:
        base = intervention.layer_name.split("#", 1)[0]
        if base in seen:
            continue
        seen.add(base)
        if intervention.strategy != "zero":
            excluded_nonzero.append(intervention.layer_name)
            continue
        candidates = by_base.get(base, [])
        if len(candidates) != 1:
            if len(candidates) > 1:
                excluded_ambiguous.append(intervention.layer_name)
            continue
        layer_name, value = candidates[0]
        names.append(layer_name)
        observed.append(value)
        causal.append(float(intervention.score_delta))

    observed_array = _finite_vector(observed)
    causal_array = _finite_vector(causal)
    observed_l1 = _l1_normalise(observed_array)
    causal_l1 = _l1_normalise(causal_array)
    count = observed_array.size
    rank = (
        spearman_correlation(observed_array.tolist(), causal_array.tolist())
        if count >= 2
        else None
    )
    signs = (
        sign_agreement(observed_array.tolist(), causal_array.tolist()) if count else None
    )
    norms = float(np.linalg.norm(observed_array) * np.linalg.norm(causal_array))
    cosine = (
        float(np.dot(observed_array, causal_array) / norms) if count and norms > 0.0 else None
    )
    discrepancy = (
        float(np.clip(0.5 * np.abs(observed_l1 - causal_l1).sum(), 0.0, 1.0))
        if count and (np.abs(observed_array).sum() > 0 or np.abs(causal_array).sum() > 0)
        else None
    )
    p_value, permutation_method = _permutation_significance(
        observed_array,
        causal_array,
        random_seed=random_seed,
        max_permutations=max_permutations,
    )
    return CausalConcordanceAudit(
        layer_names=tuple(names),
        observational_raw=tuple(float(value) for value in observed_array),
        causal_raw=tuple(float(value) for value in causal_array),
        observational_l1=tuple(float(value) for value in observed_l1),
        causal_l1=tuple(float(value) for value in causal_l1),
        spearman_rho=rank,
        sign_agreement=signs,
        cosine_similarity=cosine,
        normalised_l1_discrepancy=discrepancy,
        permutation_p_value=p_value,
        permutation_method=permutation_method,
        excluded_nonzero_interventions=tuple(excluded_nonzero),
        excluded_ambiguous_layers=tuple(excluded_ambiguous),
    )


def scientific_evidence_audit(trace: NeuralTrace) -> ScientificEvidenceAudit:
    """Build a transparent report audit from quantities supported by the trace."""

    concentration = evidence_concentration(trace)
    concordance = observational_causal_concordance(trace)
    gradient_count = len(concentration.layer_names)
    intervened = len({item.layer_name.split("#", 1)[0] for item in trace.causal_interventions})
    zero_count = len(
        {
            item.layer_name.split("#", 1)[0]
            for item in trace.causal_interventions
            if item.strategy == "zero"
        }
    )
    coverage = (
        float(len(concordance.layer_names) / gradient_count) if gradient_count else None
    )
    if trace.causal_interventions:
        tier = "targeted_interventional"
    elif gradient_count:
        tier = "observational_first_order"
    else:
        tier = "descriptive_activation"
    prediction = trace.prediction
    entropy = (
        float(np.median(_finite_vector(prediction.predictive_entropy)))
        if prediction and prediction.predictive_entropy
        else None
    )
    margin = (
        float(np.median(_finite_vector(prediction.margin)))
        if prediction and prediction.margin
        else None
    )
    return ScientificEvidenceAudit(
        evidence_tier=tier,
        concentration=concentration,
        concordance=concordance,
        gradient_layer_count=gradient_count,
        intervened_layer_count=intervened,
        zero_ablation_layer_count=zero_count,
        causal_coverage=coverage,
        median_predictive_entropy=entropy,
        median_prediction_margin=margin,
    )
