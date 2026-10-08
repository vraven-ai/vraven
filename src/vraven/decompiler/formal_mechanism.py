"""Formal survival certificates for internal and causal mechanism vectors.

The decision property, representation property and intervention-effect property are
kept separate.  A causal coordinate is defined only through an executable internal
intervention; activation magnitude alone is never treated as causal evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from html import escape
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .formal_robustness import (
    Bound,
    FormalVerificationPoint,
    IntervalBoundVerifier,
    UnsupportedVerificationOperator,
    _normalise_logit_bounds,
)
from .verification_protocol import (
    CoordinateSelectionProvenance,
    InputDomainSpec,
    PublicationReadinessLedger,
    build_formal_mechanism_publication_ledger,
    resolve_input_domain,
)

CertificateStatus = Literal["certified", "inconclusive", "unsupported"]
CausalCoordinateStatus = Literal["eligible", "inconclusive"]
ApplicabilityStatus = Literal["applicable", "inconclusive"]
Reduction = Literal["flatten", "channel_mean"]


@dataclass(slots=True, frozen=True)
class MechanismPropertySpec:
    """Pre-registered internal representation-survival property."""

    node_id: str
    reduction: Reduction = "flatten"
    coordinate_indices: tuple[int, ...] = ()
    max_relative_l2_drift: float = 0.25
    min_cosine: float = 0.95
    top_k: int = 8
    require_top_k_stability: bool = True


@dataclass(slots=True, frozen=True)
class CausalEffectPropertySpec:
    """Pre-registered survival property for a vector of intervention effects."""

    node_id: str
    coordinate_indices: tuple[int, ...] = ()
    baseline: float = 0.0
    auto_top_k: int = 4
    max_screened_coordinates: int = 256
    max_relative_l2_drift: float = 0.50
    min_cosine: float = 0.80
    min_sign_stable_fraction: float = 1.0
    minimum_clean_effect: float = 1e-8


@dataclass(slots=True)
class VectorSurvivalPoint:
    epsilon: float
    status: CertificateStatus
    certified: bool
    absolute_l2_drift_upper_bound: float | None
    clean_norm_lower_bound: float | None
    relative_l2_drift_upper_bound: float | None
    cosine_lower_bound: float | None
    sign_stable_fraction: float | None
    top_k_stable: bool | None
    checks: dict[str, bool] = field(default_factory=dict)
    coordinate_lower_bounds: list[float] = field(default_factory=list)
    coordinate_upper_bounds: list[float] = field(default_factory=list)
    unsupported_reason: str | None = None
    inconclusive_reason: str | None = None


@dataclass(slots=True)
class CausalEffectSurvivalPoint(VectorSurvivalPoint):
    effect_lower_bounds: list[float] = field(default_factory=list)
    effect_upper_bounds: list[float] = field(default_factory=list)
    minimum_clean_effect: float | None = None
    eligible_coordinate_count: int = 0
    required_coordinate_count: int = 0


@dataclass(slots=True)
class JointMechanismPoint:
    epsilon: float
    decision: FormalVerificationPoint
    representation: VectorSurvivalPoint
    causal_effect: CausalEffectSurvivalPoint
    jointly_certified: bool


@dataclass(slots=True, frozen=True)
class FormalMechanismApplicabilityRecord:
    """Zero-radius applicability ledger for one frozen-basis cohort sample."""

    selected_cohort_index: int
    sample_id: str | None
    label: int
    basis_epsilon: float
    representation_applicable: bool
    causal_applicable: bool
    mechanism_applicable: bool
    representation_applicability_status: ApplicabilityStatus
    causal_applicability_status: ApplicabilityStatus
    mechanism_applicability_status: ApplicabilityStatus
    representation_clean_norm_lower_bound: float
    representation_top_k_well_defined: bool | None
    causal_eligible_coordinate_count: int
    causal_required_coordinate_count: int
    minimum_clean_effect: float
    representation_inapplicability_reason: str | None = None
    causal_inapplicability_reason: str | None = None
    mechanism_inapplicability_reason: str | None = None


@dataclass(slots=True)
class FormalMechanismCohortPoint:
    epsilon: float
    n_samples: int
    n_clean_correct: int
    n_decision_certified: int
    n_representation_certified: int
    n_causal_effect_certified: int
    n_mechanism_certified: int
    n_jointly_certified: int
    n_decision_and_mechanism: int
    n_decision_only: int
    n_mechanism_only: int
    n_neither: int
    n_representation_applicable: int
    n_causal_applicable: int
    n_mechanism_applicable: int
    n_decision_and_representation_applicable: int
    n_decision_and_causal_applicable: int
    n_decision_and_mechanism_applicable: int
    n_applicable_decision_and_mechanism: int
    n_applicable_decision_only: int
    n_applicable_mechanism_only: int
    n_applicable_neither: int
    decision_certified_accuracy: float
    representation_certified_fraction: float
    causal_effect_certified_fraction: float
    joint_certified_accuracy: float
    representation_applicable_fraction: float
    causal_applicable_fraction: float
    mechanism_applicable_fraction: float
    applicable_decision_certified_fraction: float | None
    applicable_mechanism_certified_fraction: float | None
    applicable_joint_certified_fraction: float | None
    representation_inapplicability_reasons: dict[str, int] = field(default_factory=dict)
    causal_inapplicability_reasons: dict[str, int] = field(default_factory=dict)
    mechanism_inapplicability_reasons: dict[str, int] = field(default_factory=dict)
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class FormalCausalMechanismCertificate:
    """Proof ledger for decision, representation and causal-effect survival."""

    model_name: str
    property: str
    norm: str
    input_domain: tuple[float | None, float | None]
    target: int
    competitor: int
    mechanism_spec: MechanismPropertySpec
    causal_spec: CausalEffectPropertySpec
    selected_causal_coordinates: list[int]
    selected_coordinate_clean_effects: list[float]
    points: list[JointMechanismPoint]
    decision_radius_lower_bound: float
    representation_geometry_radius_lower_bound: float
    representation_top_k_radius_lower_bound: float
    representation_radius_lower_bound: float
    causal_geometry_radius_lower_bound: float
    causal_sign_radius_lower_bound: float
    causal_effect_radius_lower_bound: float
    joint_radius_lower_bound: float
    selected_coordinate_clean_effect_lower_bounds: list[float] = field(default_factory=list)
    causal_coordinate_status: CausalCoordinateStatus = "inconclusive"
    causal_coordinate_reason: str | None = (
        "Clean-effect eligibility was not recorded for this certificate."
    )
    causal_coordinate_eligible_count: int = 0
    causal_coordinate_required_count: int = 0
    cohort_points: list[FormalMechanismCohortPoint] = field(default_factory=list)
    cohort_applicability_records: list[FormalMechanismApplicabilityRecord] = field(
        default_factory=list
    )
    selected_cohort_indices: list[int] = field(default_factory=list)
    input_spec: InputDomainSpec | None = None
    coordinate_selection: CoordinateSelectionProvenance | None = None
    evaluation_sample_ids: list[str] = field(default_factory=list)
    publication_readiness: PublicationReadinessLedger | None = None
    verifier: str = "vraven_formal_causal_mechanism_relational_ibp_v3"
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    def write_html(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_certificate_html(self), encoding="utf-8")
        return destination


@dataclass(slots=True)
class _CausalCoordinateResolution:
    """Internal result of resolving a frozen intervention-coordinate basis."""

    spec: CausalEffectPropertySpec
    clean_effect_bounds: Bound
    provenance: CoordinateSelectionProvenance
    required_count: int
    selection_reason: str | None = None


class FormalCausalMechanismVerifier:
    """Certify internal-vector and intervention-effect survival under L∞ input sets."""

    def __init__(self, model: nn.Module, *, model_name: str | None = None) -> None:
        self.model = model
        self.model_name = model_name or type(model).__name__
        self.decision_verifier = IntervalBoundVerifier(model, model_name=self.model_name)

    def certify(
        self,
        x: Tensor,
        *,
        mechanism: MechanismPropertySpec,
        causal: CausalEffectPropertySpec,
        target: int | None = None,
        epsilon_grid: tuple[float, ...] = (
            0.0,
            1e-5,
            5e-5,
            1e-4,
            2.5e-4,
            5e-4,
            1e-3,
        ),
        clip_min: float | None = None,
        clip_max: float | None = None,
        radius_search_upper: float | None = None,
        radius_search_steps: int = 20,
        cohort_x: Tensor | None = None,
        cohort_y: Tensor | list[int] | None = None,
        selected_cohort_indices: list[int] | None = None,
        cohort_epsilon_grid: tuple[float, ...] = (0.0, 1e-5),
        input_spec: InputDomainSpec | None = None,
        calibration_x: Tensor | None = None,
        calibration_indices: list[int] | None = None,
        calibration_sample_ids: list[str] | None = None,
        calibration_labels: Tensor | list[int] | None = None,
        calibration_disjoint_from_evaluation: bool | None = None,
        evaluation_sample_ids: list[str] | None = None,
    ) -> FormalCausalMechanismCertificate:
        """Construct a proof ledger with independent and conjunctive radii."""

        clip_min, clip_max = resolve_input_domain(input_spec, clip_min=clip_min, clip_max=clip_max)
        epsilons = sorted({float(value) for value in epsilon_grid})
        if not epsilons or epsilons[0] < 0:
            raise ValueError("epsilon_grid must contain non-negative values")
        if radius_search_steps < 1:
            raise ValueError("radius_search_steps must be positive")
        _validate_property_spec(mechanism, causal)

        original_zero, _records, captured_zero = self.decision_verifier.bound_graph(
            x,
            epsilon=0.0,
            clip_min=clip_min,
            clip_max=clip_max,
            capture_nodes=(mechanism.node_id, causal.node_id),
        )
        zero_logits = _normalise_logit_bounds(original_zero)
        midpoint_logits = (zero_logits[0] + zero_logits[1]) / 2.0
        decision_target = int(midpoint_logits[0].argmax().item()) if target is None else int(target)
        competitor = _strongest_competitor(midpoint_logits[0], decision_target)
        if mechanism.node_id not in captured_zero:
            raise ValueError(f"mechanism node {mechanism.node_id!r} was not captured")
        if causal.node_id not in captured_zero:
            raise ValueError(f"causal node {causal.node_id!r} was not captured")

        mechanism_zero = _select_bound(
            _reduce_bound(captured_zero[mechanism.node_id], mechanism.reduction),
            mechanism.coordinate_indices,
        )
        resolution = self._resolve_causal_coordinates(
            x,
            causal=causal,
            original_output=original_zero,
            target=decision_target,
            competitor=competitor,
            clip_min=clip_min,
            clip_max=clip_max,
            calibration_x=calibration_x,
            calibration_indices=calibration_indices,
            calibration_sample_ids=calibration_sample_ids,
            calibration_labels=calibration_labels,
            calibration_disjoint_from_evaluation=(calibration_disjoint_from_evaluation),
        )
        resolved_causal = resolution.spec
        clean_effect_bounds = resolution.clean_effect_bounds
        selection_provenance = resolution.provenance
        clean_effect_midpoints = [
            float(((lower + upper) / 2.0).item())
            for lower, upper in zip(clean_effect_bounds[0], clean_effect_bounds[1], strict=True)
        ]
        clean_effect_lower_magnitudes = [
            float(value) for value in _interval_absolute_lower_bounds(clean_effect_bounds).tolist()
        ]
        cache: dict[float, JointMechanismPoint] = {}

        def evaluate(epsilon: float) -> JointMechanismPoint:
            key = float(epsilon)
            if key not in cache:
                cache[key] = self._verify_point(
                    x,
                    epsilon=key,
                    mechanism=mechanism,
                    mechanism_zero=mechanism_zero,
                    causal=resolved_causal,
                    causal_zero=clean_effect_bounds,
                    target=decision_target,
                    competitor=competitor,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    required_causal_coordinates=resolution.required_count,
                    causal_selection_reason=resolution.selection_reason,
                )
            return cache[key]

        points = [evaluate(epsilon) for epsilon in epsilons]
        upper = float(radius_search_upper) if radius_search_upper is not None else max(epsilons)
        radii: dict[str, float] = {}
        traces: dict[str, list[dict[str, Any]]] = {}
        predicates = {
            "decision": lambda point: point.decision.certified,
            "representation_geometry": lambda point: bool(
                point.representation.checks.get("relative_l2_drift")
                and point.representation.checks.get("cosine")
            ),
            "representation_top_k": lambda point: bool(
                point.representation.checks.get("top_k_stability")
            ),
            "representation": lambda point: point.representation.certified,
            "causal_geometry": lambda point: bool(
                point.causal_effect.checks.get("minimum_clean_effect")
                and point.causal_effect.checks.get("required_coordinate_count")
                and point.causal_effect.checks.get("relative_l2_drift")
                and point.causal_effect.checks.get("cosine")
            ),
            "causal_sign": lambda point: bool(
                point.causal_effect.checks.get("minimum_clean_effect")
                and point.causal_effect.checks.get("required_coordinate_count")
                and point.causal_effect.checks.get("sign_stability")
            ),
            "causal_effect": lambda point: point.causal_effect.certified,
            "joint": lambda point: point.jointly_certified,
        }
        for name, predicate in predicates.items():
            radius, trace = _search_radius(
                evaluate,
                predicate,
                upper=upper,
                steps=int(radius_search_steps),
            )
            radii[name] = radius
            traces[name] = trace

        evaluation_ids = list(evaluation_sample_ids or [])
        cohort_points: list[FormalMechanismCohortPoint] = []
        cohort_applicability_records: list[FormalMechanismApplicabilityRecord] = []
        cohort_indices: list[int] = []
        if cohort_x is not None:
            if cohort_y is None:
                raise ValueError("cohort_y is required for cohort mechanism certificates")
            labels = (
                [int(value) for value in cohort_y.detach().reshape(-1).tolist()]
                if isinstance(cohort_y, Tensor)
                else [int(value) for value in cohort_y]
            )
            if len(labels) != int(cohort_x.shape[0]):
                raise ValueError("cohort_y must contain one label per sample")
            cohort_indices = (
                list(range(int(cohort_x.shape[0])))
                if selected_cohort_indices is None
                else [int(value) for value in selected_cohort_indices]
            )
            if (
                not cohort_indices
                or min(cohort_indices) < 0
                or max(cohort_indices) >= int(cohort_x.shape[0])
            ):
                raise ValueError("selected_cohort_indices is empty or out of range")
            cohort_points, cohort_applicability_records = self._certify_cohort(
                cohort_x,
                labels=labels,
                indices=cohort_indices,
                sample_ids=evaluation_ids,
                mechanism=mechanism,
                causal=resolved_causal,
                required_causal_coordinates=resolution.required_count,
                causal_selection_reason=resolution.selection_reason,
                epsilon_grid=cohort_epsilon_grid,
                clip_min=clip_min,
                clip_max=clip_max,
            )

        readiness = build_formal_mechanism_publication_ledger(
            input_spec=input_spec,
            selection=selection_provenance,
            evaluation_sample_ids=evaluation_ids,
        )
        clean_eligible_count = _eligible_effect_count(
            clean_effect_bounds,
            resolved_causal.minimum_clean_effect,
        )
        causal_coordinate_reason = resolution.selection_reason
        if causal_coordinate_reason is None and clean_eligible_count < len(
            resolved_causal.coordinate_indices
        ):
            causal_coordinate_reason = _clean_effect_inconclusive_reason(
                eligible=clean_eligible_count,
                required=len(resolved_causal.coordinate_indices),
                minimum_clean_effect=resolved_causal.minimum_clean_effect,
            )
        causal_coordinate_status: CausalCoordinateStatus = (
            "eligible" if causal_coordinate_reason is None else "inconclusive"
        )
        return FormalCausalMechanismCertificate(
            model_name=self.model_name,
            property=(
                "joint local decision, internal representation and intervention-effect survival"
            ),
            norm="linf",
            input_domain=(clip_min, clip_max),
            target=decision_target,
            competitor=competitor,
            mechanism_spec=mechanism,
            causal_spec=resolved_causal,
            selected_causal_coordinates=list(resolved_causal.coordinate_indices),
            selected_coordinate_clean_effects=clean_effect_midpoints,
            points=points,
            decision_radius_lower_bound=radii["decision"],
            representation_geometry_radius_lower_bound=radii["representation_geometry"],
            representation_top_k_radius_lower_bound=radii["representation_top_k"],
            representation_radius_lower_bound=radii["representation"],
            causal_geometry_radius_lower_bound=radii["causal_geometry"],
            causal_sign_radius_lower_bound=radii["causal_sign"],
            causal_effect_radius_lower_bound=radii["causal_effect"],
            joint_radius_lower_bound=radii["joint"],
            selected_coordinate_clean_effect_lower_bounds=(clean_effect_lower_magnitudes),
            causal_coordinate_status=causal_coordinate_status,
            causal_coordinate_reason=causal_coordinate_reason,
            causal_coordinate_eligible_count=clean_eligible_count,
            causal_coordinate_required_count=resolution.required_count,
            cohort_points=cohort_points,
            cohort_applicability_records=cohort_applicability_records,
            selected_cohort_indices=cohort_indices,
            input_spec=input_spec,
            coordinate_selection=selection_provenance,
            evaluation_sample_ids=evaluation_ids,
            publication_readiness=readiness,
            assumptions=[
                "The mechanism node and its coordinate basis are frozen before radius evaluation.",
                "Each causal coordinate is defined by an executable constant-value do intervention.",
                "Original and intervened executions are paired on the same uncertain input.",
                "The target and competing class are fixed from the clean focus sample.",
                "All certificates quantify over the same declared L-infinity input set.",
            ],
            limitations=[
                "IBP is sound for supported transformers but incomplete; inconclusive is not a counterexample.",
                "Relational secant and interval enclosures can remain conservative after nonlinear or branching operators.",
                "The causal effect is specific to the declared node, coordinates and intervention baseline.",
                "Causal-coordinate eligibility depends on the pre-registered minimum clean-effect threshold.",
                "Internal interventions may leave the natural activation manifold.",
                (
                    "Coordinate selection is focus-sample adaptive and requires an independent calibration split."
                    if selection_provenance.mode == "focus_adaptive"
                    else "The frozen coordinate basis is specific to the declared selection protocol."
                ),
                "The proof concerns the real-valued parameterised network, not bit-level hardware execution.",
                "Independent verifier replay is required before treating this as a high-assurance proof object.",
            ],
            metadata={
                "effect_definition": ("e_c(x') = [z_t-z_j](x') - [z_t-z_j](x'; do(m_c=baseline))"),
                "effect_bound_method": (
                    "shared-input relational interval propagation of original minus "
                    "intervened execution; exact affine bias cancellation, sound global "
                    "secant-slope nonlinear enclosures, and terminal linear margin fusion"
                ),
                "clean_effect_eligibility_definition": (
                    "every frozen coordinate c requires inf |e_c(x)| >= "
                    f"{resolved_causal.minimum_clean_effect:.17g} at the clean sample; "
                    "the infimum is lower-bounded from its zero-radius effect interval"
                ),
                "cohort_applicability_definition": (
                    "representation and causal-effect applicability are evaluated once at "
                    "epsilon=0 on each sample under the frozen coordinate basis; mechanism "
                    "applicability is their conjunction"
                ),
                "cohort_conditional_denominator": (
                    "n_mechanism_applicable; non-applicable samples are applicability-"
                    "inconclusive and are excluded from the conditional D/M table"
                ),
                "representation_definition": (
                    "coordinate-aligned node vector after the declared reduction"
                ),
                "cosine_bound": (
                    "sqrt(1-(d/r)^2) when the certified drift d is below the clean "
                    "norm lower bound r; otherwise -1"
                ),
                "selection_rule": (
                    selection_provenance.statistic
                    if not causal.coordinate_indices
                    else "pre-declared coordinates"
                ),
                "selection_provenance": asdict(selection_provenance),
                "model_mode": "evaluation",
                "radius_search_traces": traces,
                "evidence_tier": "formal_internal_intervention_property_certificate",
            },
        )

    def _certify_cohort(
        self,
        x: Tensor,
        *,
        labels: list[int],
        indices: list[int],
        sample_ids: list[str],
        mechanism: MechanismPropertySpec,
        causal: CausalEffectPropertySpec,
        required_causal_coordinates: int,
        causal_selection_reason: str | None,
        epsilon_grid: tuple[float, ...],
        clip_min: float | None,
        clip_max: float | None,
    ) -> tuple[list[FormalMechanismCohortPoint], list[FormalMechanismApplicabilityRecord]]:
        if sample_ids and len(sample_ids) == int(x.shape[0]):
            selected_sample_ids: list[str | None] = [sample_ids[index] for index in indices]
        elif sample_ids and len(sample_ids) == len(indices):
            selected_sample_ids = list(sample_ids)
        elif sample_ids:
            raise ValueError(
                "evaluation_sample_ids must identify either the full cohort or every selected "
                "cohort sample"
            )
        else:
            selected_sample_ids = [None] * len(indices)

        contexts: list[tuple[int, int, Bound, Bound, FormalMechanismApplicabilityRecord]] = []
        for index, sample_id in zip(indices, selected_sample_ids, strict=True):
            sample = x[index : index + 1]
            target = labels[index]
            original_zero, _records, captured = self.decision_verifier.bound_graph(
                sample,
                epsilon=0.0,
                clip_min=clip_min,
                clip_max=clip_max,
                capture_nodes=(mechanism.node_id,),
            )
            logits = _normalise_logit_bounds(original_zero)
            if target < 0 or target >= int(logits[0].shape[-1]):
                raise ValueError(f"cohort label {target} is outside the class range")
            competitor = _strongest_competitor((logits[0][0] + logits[1][0]) / 2.0, target)
            mechanism_zero = _select_bound(
                _reduce_bound(captured[mechanism.node_id], mechanism.reduction),
                mechanism.coordinate_indices,
            )
            causal_zero = (
                _empty_effect_bound(original_zero)
                if not causal.coordinate_indices
                else self._effect_bounds_for_coordinates(
                    sample,
                    original_output=original_zero,
                    node_id=causal.node_id,
                    coordinates=causal.coordinate_indices,
                    baseline=causal.baseline,
                    epsilon=0.0,
                    target=target,
                    competitor=competitor,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            )
            contexts.append(
                (
                    target,
                    competitor,
                    mechanism_zero,
                    causal_zero,
                    _applicability_record(
                        selected_cohort_index=index,
                        sample_id=sample_id,
                        label=target,
                        mechanism=mechanism,
                        mechanism_zero=mechanism_zero,
                        causal=causal,
                        causal_zero=causal_zero,
                        required_causal_coordinates=required_causal_coordinates,
                        causal_selection_reason=causal_selection_reason,
                    ),
                )
            )

        output: list[FormalMechanismCohortPoint] = []
        for epsilon in sorted({float(value) for value in epsilon_grid}):
            results = [
                self._verify_point(
                    x[index : index + 1],
                    epsilon=epsilon,
                    mechanism=mechanism,
                    mechanism_zero=context[2],
                    causal=causal,
                    causal_zero=context[3],
                    target=context[0],
                    competitor=context[1],
                    clip_min=clip_min,
                    clip_max=clip_max,
                    required_causal_coordinates=required_causal_coordinates,
                    causal_selection_reason=causal_selection_reason,
                )
                for index, context in zip(indices, contexts, strict=True)
            ]
            output.append(
                _cohort_point(
                    epsilon,
                    results,
                    [labels[index] for index in indices],
                    [context[4] for context in contexts],
                )
            )
        return output, [context[4] for context in contexts]

    def _resolve_causal_coordinates(
        self,
        x: Tensor,
        *,
        causal: CausalEffectPropertySpec,
        original_output: Bound,
        target: int,
        competitor: int,
        clip_min: float | None,
        clip_max: float | None,
        calibration_x: Tensor | None,
        calibration_indices: list[int] | None,
        calibration_sample_ids: list[str] | None,
        calibration_labels: Tensor | list[int] | None,
        calibration_disjoint_from_evaluation: bool | None,
    ) -> _CausalCoordinateResolution:
        captured_output, _records, captured = self.decision_verifier.bound_graph(
            x,
            epsilon=0.0,
            clip_min=clip_min,
            clip_max=clip_max,
            capture_nodes=(causal.node_id,),
        )
        del captured_output
        width = int(captured[causal.node_id][0].reshape(1, -1).shape[-1])
        if causal.coordinate_indices:
            coordinates = _validate_indices(causal.coordinate_indices, width)
            effects = self._effect_bounds_for_coordinates(
                x,
                original_output=original_output,
                node_id=causal.node_id,
                coordinates=coordinates,
                baseline=causal.baseline,
                epsilon=0.0,
                target=target,
                competitor=competitor,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            return _CausalCoordinateResolution(
                spec=replace(causal, coordinate_indices=coordinates),
                clean_effect_bounds=effects,
                required_count=len(coordinates),
                provenance=CoordinateSelectionProvenance(
                    mode="predeclared",
                    statistic="coordinates fixed directly in CausalEffectPropertySpec",
                    data_role="property specification",
                    n_samples=0,
                    disjoint_from_evaluation=True,
                    interpretation=(
                        "The verifier did not use evaluation activations or effects to choose "
                        "the coordinate basis."
                    ),
                ),
            )

        if calibration_x is not None:
            (
                coordinates,
                scores,
                selected_indices,
                required_count,
            ) = self._calibration_coordinates(
                calibration_x,
                causal=causal,
                width=width,
                target=target,
                competitor=competitor,
                clip_min=clip_min,
                clip_max=clip_max,
                indices=calibration_indices,
            )
            effects = (
                _empty_effect_bound(original_output)
                if not coordinates
                else self._effect_bounds_for_coordinates(
                    x,
                    original_output=original_output,
                    node_id=causal.node_id,
                    coordinates=coordinates,
                    baseline=causal.baseline,
                    epsilon=0.0,
                    target=target,
                    competitor=competitor,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            )
            supplied_ids = list(calibration_sample_ids or [])
            if supplied_ids and len(supplied_ids) != len(selected_indices):
                raise ValueError(
                    "calibration_sample_ids must match the selected calibration samples"
                )
            sample_ids = supplied_ids or [f"calibration:{index}" for index in selected_indices]
            strata: dict[str, int] = {}
            if calibration_labels is not None:
                labels = (
                    [int(value) for value in calibration_labels.detach().reshape(-1).tolist()]
                    if isinstance(calibration_labels, Tensor)
                    else [int(value) for value in calibration_labels]
                )
                if len(labels) != int(calibration_x.shape[0]):
                    raise ValueError(
                        "calibration_labels must contain one label per calibration sample"
                    )
                for index in selected_indices:
                    key = str(labels[index])
                    strata[key] = strata.get(key, 0) + 1
            selection_reason = _selection_cardinality_reason(
                eligible=len(coordinates),
                required=required_count,
                minimum_clean_effect=causal.minimum_clean_effect,
                source="calibration",
            )
            return _CausalCoordinateResolution(
                spec=replace(causal, coordinate_indices=coordinates),
                clean_effect_bounds=effects,
                required_count=required_count,
                selection_reason=selection_reason,
                provenance=CoordinateSelectionProvenance(
                    mode="calibration_split",
                    statistic=(
                        "largest mean certified absolute lower bound on the zero-"
                        "intervention margin effect after screening by mean absolute "
                        "calibration activation"
                    ),
                    data_role="coordinate-selection calibration",
                    n_samples=len(selected_indices),
                    sample_ids=sample_ids,
                    strata=strata,
                    disjoint_from_evaluation=calibration_disjoint_from_evaluation,
                    selected_scores=scores,
                    interpretation=(
                        "Coordinates satisfying the minimum-effect rule were frozen using "
                        "aggregate intervention effects on the declared calibration samples "
                        "before evaluation certification. An incomplete requested basis is "
                        "reported as inconclusive rather than silently reduced."
                    ),
                ),
            )

        midpoint = (captured[causal.node_id][0] + captured[causal.node_id][1]).reshape(-1) / 2.0
        screen_n = min(width, int(causal.max_screened_coordinates))
        candidates = tuple(
            int(value) for value in torch.topk(midpoint.abs(), k=screen_n).indices.tolist()
        )
        candidate_effects = self._effect_bounds_for_coordinates(
            x,
            original_output=original_output,
            node_id=causal.node_id,
            coordinates=candidates,
            baseline=causal.baseline,
            epsilon=0.0,
            target=target,
            competitor=competitor,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        effect_lower_magnitudes = _interval_absolute_lower_bounds(candidate_effects)
        eligible = [
            index
            for index, value in enumerate(effect_lower_magnitudes)
            if float(value.item()) >= causal.minimum_clean_effect
        ]
        required_count = min(int(causal.auto_top_k), screen_n)
        ranked = sorted(
            eligible,
            key=lambda index: float(effect_lower_magnitudes[index].item()),
            reverse=True,
        )[:required_count]
        coordinates = tuple(candidates[index] for index in ranked)
        selected = (
            candidate_effects[0][ranked],
            candidate_effects[1][ranked],
        )
        selection_reason = _selection_cardinality_reason(
            eligible=len(coordinates),
            required=required_count,
            minimum_clean_effect=causal.minimum_clean_effect,
            source="focus-sample",
        )
        return _CausalCoordinateResolution(
            spec=replace(causal, coordinate_indices=coordinates),
            clean_effect_bounds=selected,
            required_count=required_count,
            selection_reason=selection_reason,
            provenance=CoordinateSelectionProvenance(
                mode="focus_adaptive",
                statistic=(
                    "largest certified absolute lower bounds on focus-sample zero-"
                    "intervention margin effects within the capped largest-magnitude "
                    "activation screen"
                ),
                data_role="evaluation focus sample",
                n_samples=1,
                sample_ids=["focus"],
                disjoint_from_evaluation=False,
                selected_scores=[float(effect_lower_magnitudes[index].item()) for index in ranked],
                interpretation=(
                    "Eligible coordinates meet the declared clean-effect lower bound. "
                    "This basis is selection-adaptive and must not support a population "
                    "claim; an incomplete requested basis is explicitly inconclusive."
                ),
            ),
        )

    def _calibration_coordinates(
        self,
        x: Tensor,
        *,
        causal: CausalEffectPropertySpec,
        width: int,
        target: int,
        competitor: int,
        clip_min: float | None,
        clip_max: float | None,
        indices: list[int] | None,
    ) -> tuple[tuple[int, ...], list[float], list[int], int]:
        """Freeze coordinates using aggregate effects on a calibration split."""

        if x.ndim < 1 or int(x.shape[0]) < 1:
            raise ValueError("calibration_x must contain at least one sample")
        selected_indices = (
            list(range(int(x.shape[0]))) if indices is None else [int(value) for value in indices]
        )
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or min(selected_indices) < 0
            or max(selected_indices) >= int(x.shape[0])
        ):
            raise ValueError("calibration_indices is empty, duplicated or out of range")

        activation_sum: Tensor | None = None
        outputs: list[tuple[Tensor, Bound]] = []
        for index in selected_indices:
            sample = x[index : index + 1]
            output, _records, captured = self.decision_verifier.bound_graph(
                sample,
                epsilon=0.0,
                clip_min=clip_min,
                clip_max=clip_max,
                capture_nodes=(causal.node_id,),
            )
            midpoint = ((captured[causal.node_id][0] + captured[causal.node_id][1]) / 2.0).reshape(
                -1
            )
            if int(midpoint.numel()) != width:
                raise ValueError("calibration and focus mechanism widths differ")
            activation_sum = (
                midpoint.abs() if activation_sum is None else activation_sum + midpoint.abs()
            )
            outputs.append((sample, output))
        assert activation_sum is not None
        mean_activation = activation_sum / len(selected_indices)
        screen_n = min(width, int(causal.max_screened_coordinates))
        candidates = tuple(
            int(value) for value in torch.topk(mean_activation, k=screen_n).indices.tolist()
        )

        effect_lower_bound_sum: Tensor | None = None
        for sample, output in outputs:
            effects = self._effect_bounds_for_coordinates(
                sample,
                original_output=output,
                node_id=causal.node_id,
                coordinates=candidates,
                baseline=causal.baseline,
                epsilon=0.0,
                target=target,
                competitor=competitor,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            lower_magnitude = _interval_absolute_lower_bounds(effects)
            effect_lower_bound_sum = (
                lower_magnitude
                if effect_lower_bound_sum is None
                else effect_lower_bound_sum + lower_magnitude
            )
        assert effect_lower_bound_sum is not None
        mean_effect = effect_lower_bound_sum / len(selected_indices)
        eligible = [
            index
            for index, value in enumerate(mean_effect)
            if float(value.item()) >= causal.minimum_clean_effect
        ]
        required_count = min(int(causal.auto_top_k), screen_n)
        ranked = sorted(
            eligible,
            key=lambda index: float(mean_effect[index].item()),
            reverse=True,
        )[:required_count]
        return (
            tuple(candidates[index] for index in ranked),
            [float(mean_effect[index].item()) for index in ranked],
            selected_indices,
            required_count,
        )

    def _verify_point(
        self,
        x: Tensor,
        *,
        epsilon: float,
        mechanism: MechanismPropertySpec,
        mechanism_zero: Bound,
        causal: CausalEffectPropertySpec,
        causal_zero: Bound,
        target: int,
        competitor: int,
        clip_min: float | None,
        clip_max: float | None,
        required_causal_coordinates: int,
        causal_selection_reason: str | None,
    ) -> JointMechanismPoint:
        decision = self.decision_verifier.verify_sample(
            x,
            epsilon=epsilon,
            target=target,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        if decision.status == "unsupported":
            unsupported = VectorSurvivalPoint(
                epsilon=epsilon,
                status="unsupported",
                certified=False,
                absolute_l2_drift_upper_bound=None,
                clean_norm_lower_bound=None,
                relative_l2_drift_upper_bound=None,
                cosine_lower_bound=None,
                sign_stable_fraction=None,
                top_k_stable=None,
                unsupported_reason=decision.unsupported_reason,
            )
            causal_unsupported = CausalEffectSurvivalPoint(**asdict(unsupported))
            return JointMechanismPoint(
                epsilon=epsilon,
                decision=decision,
                representation=unsupported,
                causal_effect=causal_unsupported,
                jointly_certified=False,
            )
        try:
            original_output, _records, captured = self.decision_verifier.bound_graph(
                x,
                epsilon=epsilon,
                clip_min=clip_min,
                clip_max=clip_max,
                capture_nodes=(mechanism.node_id,),
            )
            mechanism_bound = _select_bound(
                _reduce_bound(captured[mechanism.node_id], mechanism.reduction),
                mechanism.coordinate_indices,
            )
            representation = _vector_survival_point(
                epsilon,
                clean=mechanism_zero,
                perturbed=mechanism_bound,
                max_relative_l2_drift=mechanism.max_relative_l2_drift,
                min_cosine=mechanism.min_cosine,
                top_k=mechanism.top_k,
                require_top_k=mechanism.require_top_k_stability,
            )
            (
                representation_applicable,
                _representation_norm,
                _top_k_well_defined,
                representation_applicability_reason,
            ) = _representation_applicability(mechanism, mechanism_zero)
            representation.checks["zero_radius_applicability"] = representation_applicable
            if not representation_applicable:
                representation.certified = False
                representation.status = "inconclusive"
                representation.inconclusive_reason = (
                    "inconclusive representation applicability: "
                    f"{representation_applicability_reason}"
                )
            if causal_selection_reason is not None:
                causal_point = _inconclusive_causal_point(
                    epsilon,
                    reason=causal_selection_reason,
                    minimum_clean_effect=causal.minimum_clean_effect,
                    eligible_coordinates=_eligible_effect_count(
                        causal_zero,
                        causal.minimum_clean_effect,
                    ),
                    selected_coordinates=len(causal.coordinate_indices),
                    required_coordinates=required_causal_coordinates,
                )
                return JointMechanismPoint(
                    epsilon=epsilon,
                    decision=decision,
                    representation=representation,
                    causal_effect=causal_point,
                    jointly_certified=False,
                )
            effects = self._effect_bounds_for_coordinates(
                x,
                original_output=original_output,
                node_id=causal.node_id,
                coordinates=causal.coordinate_indices,
                baseline=causal.baseline,
                epsilon=epsilon,
                target=target,
                competitor=competitor,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            causal_base = _vector_survival_point(
                epsilon,
                clean=causal_zero,
                perturbed=effects,
                max_relative_l2_drift=causal.max_relative_l2_drift,
                min_cosine=causal.min_cosine,
                top_k=0,
                require_top_k=False,
                min_sign_stable_fraction=causal.min_sign_stable_fraction,
                minimum_clean_magnitude=causal.minimum_clean_effect,
                required_coordinate_count=required_causal_coordinates,
            )
            eligible_coordinate_count = _eligible_effect_count(
                causal_zero,
                causal.minimum_clean_effect,
            )
            causal_point = CausalEffectSurvivalPoint(
                **asdict(causal_base),
                effect_lower_bounds=[float(value) for value in effects[0].tolist()],
                effect_upper_bounds=[float(value) for value in effects[1].tolist()],
                minimum_clean_effect=causal.minimum_clean_effect,
                eligible_coordinate_count=eligible_coordinate_count,
                required_coordinate_count=required_causal_coordinates,
            )
        except (UnsupportedVerificationOperator, KeyError, ValueError) as error:
            representation = VectorSurvivalPoint(
                epsilon=epsilon,
                status="unsupported",
                certified=False,
                absolute_l2_drift_upper_bound=None,
                clean_norm_lower_bound=None,
                relative_l2_drift_upper_bound=None,
                cosine_lower_bound=None,
                sign_stable_fraction=None,
                top_k_stable=None,
                unsupported_reason=str(error),
            )
            causal_point = CausalEffectSurvivalPoint(**asdict(representation))
        joint = bool(decision.certified and representation.certified and causal_point.certified)
        return JointMechanismPoint(
            epsilon=epsilon,
            decision=decision,
            representation=representation,
            causal_effect=causal_point,
            jointly_certified=joint,
        )

    def _effect_bounds_for_coordinates(
        self,
        x: Tensor,
        *,
        original_output: Bound,
        node_id: str,
        coordinates: tuple[int, ...],
        baseline: float,
        epsilon: float,
        target: int,
        competitor: int,
        clip_min: float | None,
        clip_max: float | None,
    ) -> Bound:
        margin_coefficients = _fixed_margin_coefficients(original_output, target, competitor)
        lowers: list[Tensor] = []
        uppers: list[Tensor] = []
        for coordinate in coordinates:
            transform = _coordinate_intervention_pair(coordinate, baseline)
            _original_output, _intervened_output, effect = (
                self.decision_verifier.bound_graph_difference(
                    x,
                    epsilon=epsilon,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    paired_bound_transforms={node_id: transform},
                    output_linear_coefficients=margin_coefficients,
                )
            )
            lowers.append(effect[0].reshape(-1)[0])
            uppers.append(effect[1].reshape(-1)[0])
        return torch.stack(lowers), torch.stack(uppers)


def _validate_property_spec(
    mechanism: MechanismPropertySpec,
    causal: CausalEffectPropertySpec,
) -> None:
    if mechanism.max_relative_l2_drift < 0:
        raise ValueError("max_relative_l2_drift must be non-negative")
    if not -1 <= mechanism.min_cosine <= 1:
        raise ValueError("min_cosine must lie in [-1, 1]")
    if mechanism.top_k < 0:
        raise ValueError("top_k must be non-negative")
    if causal.auto_top_k < 1 or causal.max_screened_coordinates < 1:
        raise ValueError("causal coordinate budgets must be positive")
    if not math.isfinite(float(causal.minimum_clean_effect)) or causal.minimum_clean_effect <= 0:
        raise ValueError("minimum_clean_effect must be finite and strictly positive")
    if not 0 <= causal.min_sign_stable_fraction <= 1:
        raise ValueError("min_sign_stable_fraction must lie in [0, 1]")
    if causal.max_relative_l2_drift < 0 or not -1 <= causal.min_cosine <= 1:
        raise ValueError("invalid causal vector thresholds")


def _reduce_bound(bound: Bound, reduction: Reduction) -> Bound:
    lower, upper = bound
    if lower.shape[0] != 1:
        raise ValueError("mechanism certificates require one sample")
    if reduction == "flatten":
        return lower.reshape(-1), upper.reshape(-1)
    if reduction == "channel_mean":
        if lower.ndim < 3:
            raise ValueError("channel_mean requires [batch, channel, ...] activations")
        dims = tuple(range(2, lower.ndim))
        return lower.mean(dim=dims).reshape(-1), upper.mean(dim=dims).reshape(-1)
    raise ValueError(f"unsupported mechanism reduction {reduction!r}")


def _select_bound(bound: Bound, indices: tuple[int, ...]) -> Bound:
    if not indices:
        return bound
    resolved = _validate_indices(indices, int(bound[0].numel()))
    index = torch.tensor(resolved, device=bound[0].device, dtype=torch.long)
    return bound[0][index], bound[1][index]


def _validate_indices(indices: tuple[int, ...], width: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in indices)
    if not values or len(set(values)) != len(values):
        raise ValueError("coordinate indices must be non-empty and unique")
    if min(values) < 0 or max(values) >= width:
        raise ValueError(f"coordinate index is outside [0, {width})")
    return values


def _coordinate_intervention_pair(coordinate: int, baseline: float):
    """Return a sound paired transform for one constant-value intervention."""

    def transform(original: Bound, intervened: Bound, difference: Bound) -> tuple[Bound, Bound]:
        original_lower, original_upper = original
        intervened_lower, intervened_upper = intervened
        difference_lower, difference_upper = difference
        flat_original_lower = original_lower.reshape(original_lower.shape[0], -1)
        flat_original_upper = original_upper.reshape(original_upper.shape[0], -1)
        flat_lower = intervened_lower.clone().reshape(intervened_lower.shape[0], -1)
        flat_upper = intervened_upper.clone().reshape(intervened_upper.shape[0], -1)
        flat_difference_lower = difference_lower.clone().reshape(difference_lower.shape[0], -1)
        flat_difference_upper = difference_upper.clone().reshape(difference_upper.shape[0], -1)
        if coordinate < 0 or coordinate >= flat_lower.shape[-1]:
            raise ValueError(
                f"intervention coordinate {coordinate} is outside width {flat_lower.shape[-1]}"
            )
        flat_lower[:, coordinate] = float(baseline)
        flat_upper[:, coordinate] = float(baseline)
        # At the intervened coordinate, original - do(baseline) retains the ordinary
        # activation interval.  Every untouched coordinate keeps its incoming paired
        # difference, which is exactly zero before a single intervention.
        flat_difference_lower[:, coordinate] = flat_original_lower[:, coordinate] - float(baseline)
        flat_difference_upper[:, coordinate] = flat_original_upper[:, coordinate] - float(baseline)
        return (
            (flat_lower.reshape_as(intervened_lower), flat_upper.reshape_as(intervened_upper)),
            (
                flat_difference_lower.reshape_as(difference_lower),
                flat_difference_upper.reshape_as(difference_upper),
            ),
        )

    return transform


def _fixed_margin_interval(output: Bound, target: int, competitor: int) -> Bound:
    lower, upper = _normalise_logit_bounds(output)
    return (
        lower[0, target] - upper[0, competitor],
        upper[0, target] - lower[0, competitor],
    )


def _fixed_margin_coefficients(output: Bound, target: int, competitor: int) -> Tensor:
    """Return raw-output coefficients for one fixed normalised-logit margin."""

    lower, _upper = output
    width = int(lower.shape[-1])
    if width == 1:
        normalised_coefficients = (-0.5, 0.5)
        if target not in (0, 1) or competitor not in (0, 1):
            raise ValueError("binary single-logit margin indices must lie in {0, 1}")
        return lower.new_tensor(
            [normalised_coefficients[target] - normalised_coefficients[competitor]]
        )
    if min(target, competitor) < 0 or max(target, competitor) >= width:
        raise ValueError("fixed-margin class index is outside the model output width")
    coefficients = lower.new_zeros((width,))
    coefficients[target] = 1.0
    coefficients[competitor] = -1.0
    return coefficients


def _empty_effect_bound(output: Bound) -> Bound:
    """Create an empty effect vector on the verifier output device and dtype."""

    lower, _upper = _normalise_logit_bounds(output)
    empty = lower.new_empty((0,))
    return empty, empty.clone()


def _interval_absolute_lower_bounds(bound: Bound) -> Tensor:
    """Lower-bound the absolute value of every interval-valued coordinate."""

    lower, upper = bound
    return torch.where(
        lower > 0,
        lower,
        torch.where(upper < 0, -upper, torch.zeros_like(lower)),
    )


def _eligible_effect_count(effect: Bound, minimum_clean_effect: float) -> int:
    magnitudes = _interval_absolute_lower_bounds(effect)
    return int((magnitudes >= float(minimum_clean_effect)).sum().item())


def _selection_cardinality_reason(
    *,
    eligible: int,
    required: int,
    minimum_clean_effect: float,
    source: str,
) -> str | None:
    if eligible >= required:
        return None
    return (
        f"inconclusive causal-coordinate selection: only {eligible}/{required} "
        f"{source} coordinates have a certified absolute clean intervention effect "
        f"of at least {minimum_clean_effect:.6g}; the requested basis was not reduced "
        "silently"
    )


def _clean_effect_inconclusive_reason(
    *,
    eligible: int,
    required: int,
    minimum_clean_effect: float,
) -> str:
    return (
        f"inconclusive causal-effect property: only {eligible}/{required} frozen "
        "coordinates have an absolute clean intervention-effect lower bound of at "
        f"least {minimum_clean_effect:.6g} on this sample"
    )


def _inconclusive_causal_point(
    epsilon: float,
    *,
    reason: str,
    minimum_clean_effect: float,
    eligible_coordinates: int,
    selected_coordinates: int,
    required_coordinates: int,
) -> CausalEffectSurvivalPoint:
    minimum_effect_ok = selected_coordinates > 0 and eligible_coordinates == selected_coordinates
    coordinate_count_ok = selected_coordinates == required_coordinates
    return CausalEffectSurvivalPoint(
        epsilon=float(epsilon),
        status="inconclusive",
        certified=False,
        absolute_l2_drift_upper_bound=None,
        clean_norm_lower_bound=None,
        relative_l2_drift_upper_bound=None,
        cosine_lower_bound=None,
        sign_stable_fraction=None,
        top_k_stable=None,
        checks={
            "minimum_clean_effect": minimum_effect_ok,
            "required_coordinate_count": coordinate_count_ok,
            "relative_l2_drift": False,
            "cosine": False,
            "sign_stability": False,
            "top_k_stability": True,
        },
        inconclusive_reason=reason,
        minimum_clean_effect=float(minimum_clean_effect),
        eligible_coordinate_count=int(eligible_coordinates),
        required_coordinate_count=int(required_coordinates),
    )


def _strongest_competitor(logits: Tensor, target: int) -> int:
    candidates = [index for index in range(int(logits.numel())) if index != target]
    if not candidates:
        raise ValueError("causal mechanism verification requires at least two classes")
    return max(candidates, key=lambda index: float(logits[index].item()))


def _vector_survival_point(
    epsilon: float,
    *,
    clean: Bound,
    perturbed: Bound,
    max_relative_l2_drift: float,
    min_cosine: float,
    top_k: int,
    require_top_k: bool,
    min_sign_stable_fraction: float = 0.0,
    minimum_clean_magnitude: float | None = None,
    required_coordinate_count: int | None = None,
) -> VectorSurvivalPoint:
    clean_lower, clean_upper = clean
    lower, upper = perturbed
    if clean_lower.shape != lower.shape or clean_lower.numel() == 0:
        raise ValueError("clean and perturbed mechanism vectors must share non-zero shape")
    drift_per_coordinate = torch.maximum(
        (upper - clean_lower).abs(),
        (lower - clean_upper).abs(),
    )
    drift = float(torch.linalg.vector_norm(drift_per_coordinate).item())
    clean_abs_lower = _interval_absolute_lower_bounds(clean)
    clean_norm_lower = float(torch.linalg.vector_norm(clean_abs_lower).item())
    relative = float("inf") if clean_norm_lower <= 0 else drift / clean_norm_lower
    cosine = _cosine_lower_bound(drift, clean_norm_lower)
    clean_positive = clean_lower > 0
    clean_negative = clean_upper < 0
    clean_sign_known = clean_positive | clean_negative
    sign_stable = (clean_positive & (lower > 0)) | (clean_negative & (upper < 0))
    known_count = int(clean_sign_known.sum().item())
    sign_fraction = (
        float((sign_stable & clean_sign_known).sum().item()) / known_count if known_count else 0.0
    )
    top_stable: bool | None = None
    if top_k > 0:
        top_stable = _top_k_stable(clean, perturbed, top_k)
    clean_magnitude_ok = True
    coordinate_count_ok = True
    inconclusive_reason: str | None = None
    if minimum_clean_magnitude is not None:
        eligible_count = int((clean_abs_lower >= float(minimum_clean_magnitude)).sum().item())
        selected_count = int(clean_lower.numel())
        required_count = (
            selected_count if required_coordinate_count is None else int(required_coordinate_count)
        )
        clean_magnitude_ok = eligible_count == selected_count
        coordinate_count_ok = selected_count == required_count
        if not clean_magnitude_ok:
            inconclusive_reason = _clean_effect_inconclusive_reason(
                eligible=eligible_count,
                required=selected_count,
                minimum_clean_effect=float(minimum_clean_magnitude),
            )
        elif not coordinate_count_ok:
            inconclusive_reason = _selection_cardinality_reason(
                eligible=selected_count,
                required=required_count,
                minimum_clean_effect=float(minimum_clean_magnitude),
                source="selected",
            )
    checks = {
        "relative_l2_drift": relative <= float(max_relative_l2_drift),
        "cosine": cosine >= float(min_cosine),
        "sign_stability": sign_fraction >= float(min_sign_stable_fraction),
        "top_k_stability": (not require_top_k) or bool(top_stable),
    }
    if minimum_clean_magnitude is not None:
        checks["minimum_clean_effect"] = clean_magnitude_ok
        checks["required_coordinate_count"] = coordinate_count_ok
    certified = all(checks.values())
    return VectorSurvivalPoint(
        epsilon=float(epsilon),
        status="certified" if certified else "inconclusive",
        certified=certified,
        absolute_l2_drift_upper_bound=drift,
        clean_norm_lower_bound=clean_norm_lower,
        relative_l2_drift_upper_bound=relative,
        cosine_lower_bound=cosine,
        sign_stable_fraction=sign_fraction,
        top_k_stable=top_stable,
        checks=checks,
        coordinate_lower_bounds=[float(value) for value in lower.tolist()],
        coordinate_upper_bounds=[float(value) for value in upper.tolist()],
        inconclusive_reason=inconclusive_reason,
    )


def _cosine_lower_bound(drift_upper: float, clean_norm_lower: float) -> float:
    if clean_norm_lower <= 0 or drift_upper >= clean_norm_lower:
        return -1.0
    ratio = min(max(drift_upper / clean_norm_lower, 0.0), 1.0)
    return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))


def _top_k_stable(clean: Bound, perturbed: Bound, top_k: int) -> bool:
    clean_lower, clean_upper = clean
    lower, upper = perturbed
    k = min(int(top_k), int(clean_lower.numel()))
    if k <= 0 or k == int(clean_lower.numel()):
        return True
    clean_midpoint = (clean_lower + clean_upper) / 2.0
    selected = torch.topk(clean_midpoint, k=k).indices
    selected_mask = torch.zeros_like(clean_midpoint, dtype=torch.bool)
    selected_mask[selected] = True
    clean_order_known = bool(clean_lower[selected_mask].min() > clean_upper[~selected_mask].max())
    perturbed_order = bool(lower[selected_mask].min() > upper[~selected_mask].max())
    return clean_order_known and perturbed_order


def _representation_applicability(
    mechanism: MechanismPropertySpec,
    mechanism_zero: Bound,
) -> tuple[bool, float, bool | None, str | None]:
    """Return the zero-radius precondition for the declared representation property."""

    representation_norm = float(
        torch.linalg.vector_norm(_interval_absolute_lower_bounds(mechanism_zero)).item()
    )
    top_k_well_defined: bool | None = None
    reasons: list[str] = []
    if representation_norm <= 0:
        reasons.append(
            "the zero-radius representation has no certified non-zero norm, so relative "
            "geometry and direction are undefined"
        )
    if mechanism.require_top_k_stability:
        top_k_well_defined = bool(
            mechanism.top_k > 0 and _top_k_stable(mechanism_zero, mechanism_zero, mechanism.top_k)
        )
        if not top_k_well_defined:
            reasons.append(
                "the required zero-radius top-k set is not uniquely separated at its boundary"
            )
    reason = "; ".join(reasons) or None
    return reason is None, representation_norm, top_k_well_defined, reason


def _applicability_record(
    *,
    selected_cohort_index: int,
    sample_id: str | None,
    label: int,
    mechanism: MechanismPropertySpec,
    mechanism_zero: Bound,
    causal: CausalEffectPropertySpec,
    causal_zero: Bound,
    required_causal_coordinates: int,
    causal_selection_reason: str | None,
) -> FormalMechanismApplicabilityRecord:
    """Resolve cohort applicability once, solely from zero-radius clean bounds."""

    (
        representation_applicable,
        representation_norm,
        top_k_well_defined,
        representation_reason,
    ) = _representation_applicability(mechanism, mechanism_zero)

    selected_count = len(causal.coordinate_indices)
    eligible_count = _eligible_effect_count(causal_zero, causal.minimum_clean_effect)
    causal_reason = causal_selection_reason
    if causal_reason is None and selected_count != int(required_causal_coordinates):
        causal_reason = (
            "the frozen causal basis contains "
            f"{selected_count}/{int(required_causal_coordinates)} required coordinates"
        )
    if causal_reason is None and eligible_count != int(required_causal_coordinates):
        causal_reason = _clean_effect_inconclusive_reason(
            eligible=eligible_count,
            required=int(required_causal_coordinates),
            minimum_clean_effect=causal.minimum_clean_effect,
        )
    causal_applicable = causal_reason is None

    if representation_applicable and causal_applicable:
        mechanism_reason = None
    elif representation_applicable:
        mechanism_reason = f"causal-effect applicability is inconclusive: {causal_reason}"
    elif causal_applicable:
        mechanism_reason = f"representation applicability is inconclusive: {representation_reason}"
    else:
        mechanism_reason = (
            f"representation applicability is inconclusive: {representation_reason}; "
            f"causal-effect applicability is inconclusive: {causal_reason}"
        )

    return FormalMechanismApplicabilityRecord(
        selected_cohort_index=int(selected_cohort_index),
        sample_id=None if sample_id is None else str(sample_id),
        label=int(label),
        basis_epsilon=0.0,
        representation_applicable=representation_applicable,
        causal_applicable=causal_applicable,
        mechanism_applicable=representation_applicable and causal_applicable,
        representation_applicability_status=(
            "applicable" if representation_applicable else "inconclusive"
        ),
        causal_applicability_status="applicable" if causal_applicable else "inconclusive",
        mechanism_applicability_status=(
            "applicable" if representation_applicable and causal_applicable else "inconclusive"
        ),
        representation_clean_norm_lower_bound=representation_norm,
        representation_top_k_well_defined=top_k_well_defined,
        causal_eligible_coordinate_count=eligible_count,
        causal_required_coordinate_count=int(required_causal_coordinates),
        minimum_clean_effect=float(causal.minimum_clean_effect),
        representation_inapplicability_reason=representation_reason,
        causal_inapplicability_reason=causal_reason,
        mechanism_inapplicability_reason=mechanism_reason,
    )


def _search_radius(
    evaluate,
    predicate,
    *,
    upper: float,
    steps: int,
) -> tuple[float, list[dict[str, Any]]]:
    lower = 0.0
    upper_bound = float(upper)
    trace: list[dict[str, Any]] = []
    endpoint = evaluate(upper_bound)
    endpoint_ok = bool(predicate(endpoint))
    trace.append({"epsilon": upper_bound, "certified": endpoint_ok})
    if endpoint_ok:
        return upper_bound, trace
    zero = evaluate(0.0)
    if not predicate(zero):
        trace.append({"epsilon": 0.0, "certified": False})
        return 0.0, trace
    for _ in range(steps):
        midpoint = (lower + upper_bound) / 2.0
        result = evaluate(midpoint)
        passed = bool(predicate(result))
        trace.append({"epsilon": midpoint, "certified": passed})
        if passed:
            lower = midpoint
        else:
            upper_bound = midpoint
    return lower, trace


def _cohort_point(
    epsilon: float,
    results: list[JointMechanismPoint],
    labels: list[int],
    applicability_records: list[FormalMechanismApplicabilityRecord],
) -> FormalMechanismCohortPoint:
    n = len(results)
    if n == 0 or len(labels) != n or len(applicability_records) != n:
        raise ValueError(
            "results, labels and applicability_records must have the same non-zero length"
        )
    for label, record in zip(labels, applicability_records, strict=True):
        if record.label != label:
            raise ValueError("applicability-record labels must match cohort labels")
        if record.basis_epsilon != 0.0:
            raise ValueError("cohort applicability must be resolved at epsilon=0")
        if record.mechanism_applicable != (
            record.representation_applicable and record.causal_applicable
        ):
            raise ValueError("mechanism applicability must equal representation∩causal")
        for applicable, status in (
            (record.representation_applicable, record.representation_applicability_status),
            (record.causal_applicable, record.causal_applicability_status),
            (record.mechanism_applicable, record.mechanism_applicability_status),
        ):
            expected = "applicable" if applicable else "inconclusive"
            if status != expected:
                raise ValueError("applicability status must agree with its applicability indicator")
    clean = [
        point.decision.clean_prediction == label
        for point, label in zip(results, labels, strict=True)
    ]
    decision = [point.decision.certified and clean[index] for index, point in enumerate(results)]
    representation = [point.representation.certified for point in results]
    causal = [point.causal_effect.certified for point in results]
    mechanism = [representation[index] and causal[index] for index in range(len(results))]
    decision_and_mechanism = [decision[index] and mechanism[index] for index in range(len(results))]
    decision_only = [decision[index] and not mechanism[index] for index in range(len(results))]
    mechanism_only = [mechanism[index] and not decision[index] for index in range(len(results))]
    neither = [not decision[index] and not mechanism[index] for index in range(len(results))]
    joint = [point.jointly_certified and clean[index] for index, point in enumerate(results)]
    if joint != decision_and_mechanism:
        raise RuntimeError("joint certificate accounting disagrees with D∩(R∩C)")
    representation_applicable = [
        record.representation_applicable for record in applicability_records
    ]
    causal_applicable = [record.causal_applicable for record in applicability_records]
    mechanism_applicable = [record.mechanism_applicable for record in applicability_records]
    if any(mechanism[index] and not mechanism_applicable[index] for index in range(n)):
        raise RuntimeError("mechanism survival cannot be certified outside its applicability set")
    decision_and_representation_applicable = [
        decision[index] and representation_applicable[index] for index in range(n)
    ]
    decision_and_causal_applicable = [
        decision[index] and causal_applicable[index] for index in range(n)
    ]
    decision_and_mechanism_applicable = [
        decision[index] and mechanism_applicable[index] for index in range(n)
    ]
    applicable_decision_and_mechanism = [
        mechanism_applicable[index] and decision[index] and mechanism[index] for index in range(n)
    ]
    applicable_decision_only = [
        mechanism_applicable[index] and decision[index] and not mechanism[index]
        for index in range(n)
    ]
    applicable_mechanism_only = [
        mechanism_applicable[index] and mechanism[index] and not decision[index]
        for index in range(n)
    ]
    applicable_neither = [
        mechanism_applicable[index] and not decision[index] and not mechanism[index]
        for index in range(n)
    ]

    def reason_counts(attribute: str, positions: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index in positions:
            reason = getattr(applicability_records[index], attribute)
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(set(labels)):
        positions = [index for index, value in enumerate(labels) if value == label]
        n_class_applicable = sum(mechanism_applicable[index] for index in positions)
        n_class_applicable_decision = sum(
            decision_and_mechanism_applicable[index] for index in positions
        )
        n_class_applicable_mechanism = sum(
            mechanism[index] and mechanism_applicable[index] for index in positions
        )
        n_class_applicable_joint = sum(
            applicable_decision_and_mechanism[index] for index in positions
        )
        per_class[str(label)] = {
            "n": len(positions),
            "clean_correct": sum(clean[index] for index in positions),
            "decision_certified": sum(decision[index] for index in positions),
            "representation_certified": sum(representation[index] for index in positions),
            "causal_effect_certified": sum(causal[index] for index in positions),
            "mechanism_certified": sum(mechanism[index] for index in positions),
            "jointly_certified": sum(joint[index] for index in positions),
            "decision_and_mechanism": sum(decision_and_mechanism[index] for index in positions),
            "decision_only": sum(decision_only[index] for index in positions),
            "mechanism_only": sum(mechanism_only[index] for index in positions),
            "neither": sum(neither[index] for index in positions),
            "n_representation_applicable": sum(
                representation_applicable[index] for index in positions
            ),
            "n_causal_applicable": sum(causal_applicable[index] for index in positions),
            "n_mechanism_applicable": n_class_applicable,
            "n_decision_and_representation_applicable": sum(
                decision_and_representation_applicable[index] for index in positions
            ),
            "n_decision_and_causal_applicable": sum(
                decision_and_causal_applicable[index] for index in positions
            ),
            "n_decision_and_mechanism_applicable": n_class_applicable_decision,
            "n_applicable_decision_and_mechanism": n_class_applicable_joint,
            "n_applicable_decision_only": sum(
                applicable_decision_only[index] for index in positions
            ),
            "n_applicable_mechanism_only": sum(
                applicable_mechanism_only[index] for index in positions
            ),
            "n_applicable_neither": sum(applicable_neither[index] for index in positions),
            "applicable_decision_certified_fraction": (
                n_class_applicable_decision / n_class_applicable if n_class_applicable else None
            ),
            "applicable_mechanism_certified_fraction": (
                n_class_applicable_mechanism / n_class_applicable if n_class_applicable else None
            ),
            "applicable_joint_certified_fraction": (
                n_class_applicable_joint / n_class_applicable if n_class_applicable else None
            ),
            "representation_inapplicability_reasons": reason_counts(
                "representation_inapplicability_reason", positions
            ),
            "causal_inapplicability_reasons": reason_counts(
                "causal_inapplicability_reason", positions
            ),
            "mechanism_inapplicability_reasons": reason_counts(
                "mechanism_inapplicability_reason", positions
            ),
        }
        if (
            n_class_applicable_joint
            + per_class[str(label)]["n_applicable_decision_only"]
            + per_class[str(label)]["n_applicable_mechanism_only"]
            + per_class[str(label)]["n_applicable_neither"]
            != n_class_applicable
        ):
            raise RuntimeError(
                "class-conditional applicable D/M counts do not sum to the applicable class size"
            )
    n_clean = sum(clean)
    n_decision = sum(decision)
    n_representation = sum(representation)
    n_causal = sum(causal)
    n_mechanism = sum(mechanism)
    n_joint = sum(joint)
    n_decision_only = sum(decision_only)
    n_mechanism_only = sum(mechanism_only)
    n_neither = sum(neither)
    n_representation_applicable = sum(representation_applicable)
    n_causal_applicable = sum(causal_applicable)
    n_mechanism_applicable = sum(mechanism_applicable)
    n_decision_and_representation_applicable = sum(decision_and_representation_applicable)
    n_decision_and_causal_applicable = sum(decision_and_causal_applicable)
    n_decision_and_mechanism_applicable = sum(decision_and_mechanism_applicable)
    n_applicable_decision_and_mechanism = sum(applicable_decision_and_mechanism)
    n_applicable_decision_only = sum(applicable_decision_only)
    n_applicable_mechanism_only = sum(applicable_mechanism_only)
    n_applicable_neither = sum(applicable_neither)
    if n_joint + n_decision_only + n_mechanism_only + n_neither != n:
        raise RuntimeError("decision/mechanism contingency counts do not sum to cohort size")
    if (
        n_applicable_decision_and_mechanism
        + n_applicable_decision_only
        + n_applicable_mechanism_only
        + n_applicable_neither
        != n_mechanism_applicable
    ):
        raise RuntimeError(
            "applicable decision/mechanism contingency counts do not sum to applicability size"
        )
    if n_applicable_decision_and_mechanism != n_joint:
        raise RuntimeError("joint certification must be contained in mechanism applicability")
    return FormalMechanismCohortPoint(
        epsilon=float(epsilon),
        n_samples=n,
        n_clean_correct=n_clean,
        n_decision_certified=n_decision,
        n_representation_certified=n_representation,
        n_causal_effect_certified=n_causal,
        n_mechanism_certified=n_mechanism,
        n_jointly_certified=n_joint,
        n_decision_and_mechanism=n_joint,
        n_decision_only=n_decision_only,
        n_mechanism_only=n_mechanism_only,
        n_neither=n_neither,
        n_representation_applicable=n_representation_applicable,
        n_causal_applicable=n_causal_applicable,
        n_mechanism_applicable=n_mechanism_applicable,
        n_decision_and_representation_applicable=(n_decision_and_representation_applicable),
        n_decision_and_causal_applicable=n_decision_and_causal_applicable,
        n_decision_and_mechanism_applicable=n_decision_and_mechanism_applicable,
        n_applicable_decision_and_mechanism=n_applicable_decision_and_mechanism,
        n_applicable_decision_only=n_applicable_decision_only,
        n_applicable_mechanism_only=n_applicable_mechanism_only,
        n_applicable_neither=n_applicable_neither,
        decision_certified_accuracy=n_decision / n,
        representation_certified_fraction=n_representation / n,
        causal_effect_certified_fraction=n_causal / n,
        joint_certified_accuracy=n_joint / n,
        representation_applicable_fraction=n_representation_applicable / n,
        causal_applicable_fraction=n_causal_applicable / n,
        mechanism_applicable_fraction=n_mechanism_applicable / n,
        applicable_decision_certified_fraction=(
            n_decision_and_mechanism_applicable / n_mechanism_applicable
            if n_mechanism_applicable
            else None
        ),
        applicable_mechanism_certified_fraction=(
            n_mechanism / n_mechanism_applicable if n_mechanism_applicable else None
        ),
        applicable_joint_certified_fraction=(
            n_applicable_decision_and_mechanism / n_mechanism_applicable
            if n_mechanism_applicable
            else None
        ),
        representation_inapplicability_reasons=reason_counts(
            "representation_inapplicability_reason", list(range(n))
        ),
        causal_inapplicability_reasons=reason_counts(
            "causal_inapplicability_reason", list(range(n))
        ),
        mechanism_inapplicability_reasons=reason_counts(
            "mechanism_inapplicability_reason", list(range(n))
        ),
        per_class=per_class,
    )


def _certificate_html(report: FormalCausalMechanismCertificate) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:.6g}</td>"
        f"<td class='{point.decision.status}'>{escape(point.decision.status)}</td>"
        f"<td class='{point.representation.status}'>{escape(point.representation.status)}</td>"
        f"<td>{_number(point.representation.relative_l2_drift_upper_bound)}</td>"
        f"<td>{_number(point.representation.cosine_lower_bound)}</td>"
        f"<td class='{point.causal_effect.status}'>{escape(point.causal_effect.status)}</td>"
        f"<td>{point.causal_effect.eligible_coordinate_count}/"
        f"{point.causal_effect.required_coordinate_count}</td>"
        f"<td>{_number(point.causal_effect.relative_l2_drift_upper_bound)}</td>"
        f"<td>{_number(point.causal_effect.cosine_lower_bound)}</td>"
        f"<td class='{'certified' if point.jointly_certified else 'inconclusive'}'>"
        f"{'certified' if point.jointly_certified else 'inconclusive'}</td>"
        "</tr>"
        for point in report.points
    )
    clean_effect_lower_bounds: list[float | None] = (
        list(report.selected_coordinate_clean_effect_lower_bounds)
        if len(report.selected_coordinate_clean_effect_lower_bounds)
        == len(report.selected_causal_coordinates)
        else [None] * len(report.selected_causal_coordinates)
    )
    coordinates = "".join(
        f"<tr><td>{coordinate}</td><td>{effect:.6g}</td><td>{_number(lower_bound)}</td></tr>"
        for coordinate, effect, lower_bound in zip(
            report.selected_causal_coordinates,
            report.selected_coordinate_clean_effects,
            clean_effect_lower_bounds,
            strict=True,
        )
    )
    cohort_rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:.6g}</td>"
        f"<td>{point.n_decision_certified}/{point.n_samples}</td>"
        f"<td>{point.n_representation_certified}/{point.n_samples}</td>"
        f"<td>{point.n_causal_effect_certified}/{point.n_samples}</td>"
        f"<td>{point.n_mechanism_certified}/{point.n_samples}</td>"
        f"<td>{point.n_jointly_certified}/{point.n_samples}</td>"
        f"<td>{point.n_decision_only}/{point.n_samples}</td>"
        f"<td>{point.n_mechanism_only}/{point.n_samples}</td>"
        f"<td>{point.n_neither}/{point.n_samples}</td>"
        "</tr>"
        for point in report.cohort_points
    )
    applicability_rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:.6g}</td>"
        f"<td>{point.n_representation_applicable}/{point.n_samples}</td>"
        f"<td>{point.n_causal_applicable}/{point.n_samples}</td>"
        f"<td>{point.n_mechanism_applicable}/{point.n_samples}</td>"
        f"<td>{_count_over_denominator(point.n_decision_and_mechanism_applicable, point.n_mechanism_applicable)}</td>"
        f"<td>{_count_over_denominator(point.n_applicable_decision_and_mechanism, point.n_mechanism_applicable)}</td>"
        f"<td>{_count_over_denominator(point.n_applicable_decision_only, point.n_mechanism_applicable)}</td>"
        f"<td>{_count_over_denominator(point.n_applicable_mechanism_only, point.n_mechanism_applicable)}</td>"
        f"<td>{_count_over_denominator(point.n_applicable_neither, point.n_mechanism_applicable)}</td>"
        "</tr>"
        for point in report.cohort_points
    )
    applicability_record_rows = "".join(
        "<tr>"
        f"<td>{record.selected_cohort_index}</td>"
        f"<td>{escape(record.sample_id) if record.sample_id is not None else '—'}</td>"
        f"<td>{record.label}</td>"
        f"<td class='{'certified' if record.representation_applicable else 'inconclusive'}'>"
        f"{'applicable' if record.representation_applicable else 'inconclusive'}</td>"
        f"<td class='{'certified' if record.causal_applicable else 'inconclusive'}'>"
        f"{'applicable' if record.causal_applicable else 'inconclusive'} · "
        f"{record.causal_eligible_coordinate_count}/{record.causal_required_coordinate_count}</td>"
        f"<td class='{'certified' if record.mechanism_applicable else 'inconclusive'}'>"
        f"{'applicable' if record.mechanism_applicable else 'inconclusive'}</td>"
        f"<td>{escape(record.mechanism_inapplicability_reason or '—')}</td>"
        "</tr>"
        for record in report.cohort_applicability_records
    )
    first_cohort_point = report.cohort_points[0] if report.cohort_points else None
    reason_rows = ""
    if first_cohort_point is not None:
        for component, reasons in (
            ("Representation", first_cohort_point.representation_inapplicability_reasons),
            ("Causal effect", first_cohort_point.causal_inapplicability_reasons),
            ("Mechanism conjunction", first_cohort_point.mechanism_inapplicability_reasons),
        ):
            reason_rows += "".join(
                f"<tr><td>{component}</td><td>{count}</td><td>{escape(reason)}</td></tr>"
                for reason, count in sorted(reasons.items())
            )
    assumptions = "".join(f"<li>{escape(value)}</li>" for value in report.assumptions)
    limitations = "".join(f"<li>{escape(value)}</li>" for value in report.limitations)
    selection = report.coordinate_selection
    selection_text = (
        "Legacy certificate: coordinate-selection provenance was not recorded."
        if selection is None
        else (
            f"Mode: {selection.mode}; data role: {selection.data_role}; "
            f"n={selection.n_samples}; disjoint from evaluation: "
            f"{selection.disjoint_from_evaluation}. {selection.interpretation}"
        )
    )
    readiness = report.publication_readiness
    gate_rows = (
        ""
        if readiness is None
        else "".join(
            "<tr>"
            f"<td>{escape(check.name)}</td>"
            f"<td class='{'certified' if check.verdict == 'pass' else 'inconclusive'}'>"
            f"{escape(check.verdict)}</td><td>{escape(check.evidence)}</td></tr>"
            for check in readiness.checks
        )
    )
    readiness_status = "not assessed" if readiness is None else readiness.status
    coordinate_reason = (
        "All coordinates meet the declared local clean-effect threshold."
        if report.causal_coordinate_reason is None
        else report.causal_coordinate_reason
    )
    input_contract = (
        {"bounds": report.input_domain, "status": "legacy / semantics undeclared"}
        if report.input_spec is None
        else asdict(report.input_spec)
    )
    return f"""<!doctype html><html lang='en' data-theme='dark'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>VRAVEN formal causal mechanism certificate</title>
<style>:root{{--bg:#030A12;--panel:#07111F;--raised:#0B1728;--line:#1C344D;--text:#F1F5F9;--muted:#94A3B8;--ok:#32D74B;--warn:#FFD60A;--accent:#00D4FF}}html[data-theme='light']{{--bg:#fff;--panel:#fff;--raised:#f8fafc;--line:#cbd5e1;--text:#111827;--muted:#526173;--ok:#047857;--warn:#92400e;--accent:#0369a1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}header,.panel,.hero{{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:20px;margin-bottom:15px}}header{{display:flex;justify-content:space-between;gap:16px}}h1{{margin:3px 0;font-size:25px}}h2{{font-size:15px;color:var(--accent)}}p,li{{color:var(--muted)}}.k{{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.radii{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:9px}}.metric{{padding:12px;border:1px solid var(--line);border-radius:9px;background:var(--raised)}}.metric span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}.metric strong{{font:700 19px ui-monospace,monospace}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);text-transform:uppercase;font-size:10px}}td.certified{{color:var(--ok);font-weight:800}}td.inconclusive,td.unsupported{{color:var(--warn);font-weight:800}}button{{padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--raised);color:var(--text)}}code,pre{{font-family:ui-monospace,monospace;white-space:pre-wrap}}</style><script>function theme(t){{document.documentElement.dataset.theme=t}}</script></head><body><main>
<header><div><div class='k'>Formal causal mechanism verification · {escape(report.verifier)}</div><h1>Decision, representation and intervention-effect survival</h1><p>{escape(report.model_name)} · node {escape(report.mechanism_spec.node_id)} · L∞ domain {escape(str(report.input_domain))}</p></div><div><button onclick="theme('dark')">Dark</button> <button onclick="theme('light')">Publication light</button></div></header>
<section class='hero'><div class='k'>Separately certified radii</div><div class='radii'><div class='metric'><span>Decision radius ≥</span><strong>{report.decision_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Representation geometry ≥</span><strong>{report.representation_geometry_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Representation top-k ≥</span><strong>{report.representation_top_k_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Full representation ≥</span><strong>{report.representation_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Causal geometry ≥</span><strong>{report.causal_geometry_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Causal sign ≥</span><strong>{report.causal_sign_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Full causal effect ≥</span><strong>{report.causal_effect_radius_lower_bound:.6g}</strong></div><div class='metric'><span>Joint radius ≥</span><strong>{report.joint_radius_lower_bound:.6g}</strong></div></div><p>The joint certificate is the conjunction of three declared properties. Component radii remain separate so a top-k ordering bottleneck cannot be misreported as geometric collapse. An inconclusive component is not a counterexample and does not prove mechanism change.</p></section>
<section class='panel'><h2>Proof curve</h2><table><thead><tr><th>ε</th><th>Decision</th><th>Representation</th><th>Rep. rel. L2 ≤</th><th>Rep. cosine ≥</th><th>Causal effect</th><th>Eligible coordinates</th><th>Effect rel. L2 ≤</th><th>Effect cosine ≥</th><th>Joint</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class='panel'><h2>Frozen-basis cohort proof coverage · all selected samples</h2><p>These are unconditional certificate-coverage counts over all selected cohort samples. Mechanism is the exact conjunction R∩C of representation and causal-effect survival, and joint is D∩R∩C. A non-certificate can be inapplicable or merely unresolved; it is not labelled as a property failure.</p><table><thead><tr><th>ε</th><th>D</th><th>R</th><th>C</th><th>R∩C</th><th>D∩R∩C</th><th>D only</th><th>R∩C only</th><th>Neither</th></tr></thead><tbody>{cohort_rows or '<tr><td colspan="9">No cohort was supplied.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Applicability-conditioned survival ledger</h2><p>Applicability is frozen once at ε=0. Representation applicability requires non-zero clean geometry and, when requested, a uniquely separated clean top-k boundary. Causal applicability requires every required frozen coordinate to meet the declared minimum clean-effect rule. A = representation-applicable ∩ causal-applicable. The final five columns use A—not the full cohort—as their denominator. Samples outside A are applicability-inconclusive and excluded, never counted as mechanism-property failures.</p><table><thead><tr><th>ε</th><th>R applicable / N</th><th>C applicable / N</th><th>A / N</th><th>D∩A / A</th><th>D∩M / A</th><th>D only / A</th><th>M only / A</th><th>Neither / A</th></tr></thead><tbody>{applicability_rows or '<tr><td colspan="9">No cohort was supplied.</td></tr>'}</tbody></table><h2>Zero-radius inapplicability reasons</h2><table><thead><tr><th>Component</th><th>Samples</th><th>Reason</th></tr></thead><tbody>{reason_rows or '<tr><td colspan="3">No inapplicable cohort sample was recorded.</td></tr>'}</tbody></table><details><summary>Per-sample zero-radius applicability records</summary><table><thead><tr><th>Cohort index</th><th>Sample ID</th><th>Label</th><th>R applicability</th><th>C applicability · eligible/required</th><th>A</th><th>Reason</th></tr></thead><tbody>{applicability_record_rows or '<tr><td colspan="7">No cohort applicability records were supplied.</td></tr>'}</tbody></table></details></section>
<section class='panel'><h2>Executable intervention basis · <span class='{report.causal_coordinate_status}'>{report.causal_coordinate_status}</span></h2><p>Each coordinate was independently set to baseline {report.causal_spec.baseline:g}. The measured clean quantity is the change in the fixed target-versus-competitor logit margin. Eligibility requires every coordinate to have a certified absolute clean-effect lower bound of at least {report.causal_spec.minimum_clean_effect:.6g}; {report.causal_coordinate_eligible_count}/{report.causal_coordinate_required_count} coordinates are eligible. {escape(coordinate_reason)}</p><table><thead><tr><th>Flattened coordinate</th><th>Clean intervention effect midpoint</th><th>Certified |effect| lower bound</th></tr></thead><tbody>{coordinates or '<tr><td colspan="3">No eligible coordinate was selected.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Selection protocol</h2><p>{escape(selection_text)}</p><h2>Input contract</h2><pre>{escape(json.dumps(input_contract, indent=2))}</pre></section>
<section class='panel'><h2>Publication-evidence gate · {escape(readiness_status)}</h2><p>This is a non-composite methodological ledger. It does not alter the local mathematical verdict.</p><table><thead><tr><th>Requirement</th><th>Verdict</th><th>Evidence</th></tr></thead><tbody>{gate_rows or '<tr><td colspan="3">Not assessed for this legacy certificate.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Property contract</h2><pre>{escape(json.dumps({"mechanism": asdict(report.mechanism_spec), "causal": asdict(report.causal_spec)}, indent=2))}</pre><h2>Assumptions</h2><ul>{assumptions}</ul><h2>Limitations</h2><ul>{limitations}</ul></section></main></body></html>"""


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isinf(value):
        return "∞"
    return f"{value:.6g}"


def _count_over_denominator(count: int, denominator: int) -> str:
    if denominator == 0:
        return "— (A=0)"
    return f"{count}/{denominator}"
