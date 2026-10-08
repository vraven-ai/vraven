"""Declared verification semantics and publication-evidence gates.

The objects in this module do not turn a certificate into a publication claim.
They make the assumptions needed to interpret a certificate machine-readable and
retain an explicit ledger of validation work that has and has not been completed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GateVerdict = Literal["pass", "fail", "not_assessed"]
SelectionMode = Literal["predeclared", "calibration_split", "focus_adaptive"]


@dataclass(slots=True, frozen=True)
class InputDomainSpec:
    """Semantic contract for the model input on which a proof is interpreted.

    ``None`` bounds denote a deliberately unbounded side of the mathematical
    domain.  This is different from omitting the specification: ``declared`` and
    the textual fields record whether the unbounded choice was intentional.
    """

    lower: float | None
    upper: float | None
    units: str
    preprocessing: str
    semantic_domain: str
    source: str
    declared: bool = True

    def validate(self) -> None:
        if self.lower is not None and not math.isfinite(float(self.lower)):
            raise ValueError("input-domain lower bound must be finite or None")
        if self.upper is not None and not math.isfinite(float(self.upper)):
            raise ValueError("input-domain upper bound must be finite or None")
        if (
            self.lower is not None
            and self.upper is not None
            and float(self.lower) >= float(self.upper)
        ):
            raise ValueError("input-domain lower bound must be smaller than upper")
        if self.declared and not all(
            value.strip()
            for value in (
                self.units,
                self.preprocessing,
                self.semantic_domain,
                self.source,
            )
        ):
            raise ValueError(
                "a declared input domain requires units, preprocessing, semantic_domain and source"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CoordinateSelectionProvenance:
    """Provenance for the frozen intervention-coordinate basis."""

    mode: SelectionMode
    statistic: str
    data_role: str
    n_samples: int
    sample_ids: list[str] = field(default_factory=list)
    strata: dict[str, int] = field(default_factory=dict)
    disjoint_from_evaluation: bool | None = None
    selected_scores: list[float] = field(default_factory=list)
    interpretation: str = ""


@dataclass(slots=True)
class PublicationGateCheck:
    """One independently inspectable methodological requirement."""

    name: str
    verdict: GateVerdict
    evidence: str
    required_for_population_claim: bool = True


@dataclass(slots=True)
class PublicationReadinessLedger:
    """Non-composite ledger of publication-evidence requirements."""

    scope: str
    status: str
    checks: list[PublicationGateCheck]
    blocking_requirements: list[str]
    interpretation: str


def build_formal_mechanism_publication_ledger(
    *,
    input_spec: InputDomainSpec | None,
    selection: CoordinateSelectionProvenance,
    evaluation_sample_ids: list[str],
    external_verifier_replay: bool = False,
    complete_verifier_evaluation: bool = False,
    multi_model_validation: bool = False,
) -> PublicationReadinessLedger:
    """Build explicit gates without collapsing them into a trust score."""

    input_declared = bool(input_spec is not None and input_spec.declared)
    independent_selection = bool(
        selection.mode == "predeclared"
        or (
            selection.mode == "calibration_split"
            and selection.disjoint_from_evaluation is True
        )
    )
    checks = [
        PublicationGateCheck(
            name="declared_input_semantics",
            verdict="pass" if input_declared else "fail",
            evidence=(
                "Units, preprocessing, semantic domain and domain source are recorded."
                if input_declared
                else "The mathematical bounds may be executable, but their input semantics are not declared."
            ),
        ),
        PublicationGateCheck(
            name="independent_coordinate_selection",
            verdict="pass" if independent_selection else "fail",
            evidence=(
                f"Coordinate basis is frozen using {selection.mode}."
                if independent_selection
                else "Coordinates were selected on the evaluated focus sample or split disjointness is unverified."
            ),
        ),
        PublicationGateCheck(
            name="evaluation_sample_identity",
            verdict="pass" if evaluation_sample_ids else "fail",
            evidence=(
                f"{len(evaluation_sample_ids)} evaluation sample identifiers are recorded."
                if evaluation_sample_ids
                else "Evaluation sample identifiers are absent."
            ),
        ),
        PublicationGateCheck(
            name="independent_verifier_replay",
            verdict="pass" if external_verifier_replay else "fail",
            evidence=(
                "A separately implemented verifier reproduced the declared properties."
                if external_verifier_replay
                else "No independently implemented verifier replay is recorded."
            ),
        ),
        PublicationGateCheck(
            name="complete_verifier_separation_test",
            verdict="pass" if complete_verifier_evaluation else "fail",
            evidence=(
                "Complete verification or verified counterexamples test property separation."
                if complete_verifier_evaluation
                else "IBP inconclusiveness cannot establish an existential mechanism change."
            ),
        ),
        PublicationGateCheck(
            name="multi_model_validation",
            verdict="pass" if multi_model_validation else "fail",
            evidence=(
                "The protocol has been evaluated across the declared model study."
                if multi_model_validation
                else "The current certificate is not a multi-model population study."
            ),
        ),
    ]
    blockers = [
        check.name
        for check in checks
        if check.required_for_population_claim and check.verdict != "pass"
    ]
    status = (
        "population_claim_protocol_complete"
        if not blockers
        else "research_evidence_external_validation_required"
    )
    return PublicationReadinessLedger(
        scope="formal causal-mechanism population claim",
        status=status,
        checks=checks,
        blocking_requirements=blockers,
        interpretation=(
            "The ledger is a checklist, not a numerical quality score. A formal local "
            "certificate remains valid within its declared semantics even when population-"
            "claim requirements are incomplete."
        ),
    )


def resolve_input_domain(
    input_spec: InputDomainSpec | None,
    *,
    clip_min: float | None,
    clip_max: float | None,
) -> tuple[float | None, float | None]:
    """Validate and align legacy clip arguments with a semantic domain spec."""

    if input_spec is None:
        return clip_min, clip_max
    input_spec.validate()
    if clip_min is not None and input_spec.lower != float(clip_min):
        raise ValueError("clip_min conflicts with input_spec.lower")
    if clip_max is not None and input_spec.upper != float(clip_max):
        raise ValueError("clip_max conflicts with input_spec.upper")
    return input_spec.lower, input_spec.upper
