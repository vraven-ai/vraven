"""Validity envelope and evidence certificate for VRAVEN decompilation results.

VRAVEN deliberately avoids collapsing heterogeneous evidence into a single trust score.
The certificate reports computational capture, semantic grounding, interventional access,
programme fidelity, counterfactual verification and unresolved responsibility separately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .capture import CapturedModel
from .causal_graph import CausalMechanismGraph
from .concepts import ConceptAtlas
from .counterfactual import CounterfactualProof
from .programme import MechanismProgram


@dataclass(slots=True)
class ValidityEnvelope:
    """Declared conditions under which the recovered explanation was evaluated."""

    capture_route: str
    observed_input_shapes: dict[str, list[int | str]]
    shape_constraints: list[dict[str, Any]]
    covered_execution_paths: list[str]
    unresolved_operations: list[dict[str, Any]]
    supported_interventions: list[str]
    model_state: str
    framework_version: str
    numerical_equivalence_passed: bool | None
    numerical_equivalence_max_abs: float | None
    restrictions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CoverageCertificate:
    operator_coverage: float
    parameter_coverage: float
    gradient_coverage: float
    intervention_coverage: float
    semantic_coverage: float
    semantic_association_coverage: float
    causal_semantic_coverage: float
    causal_semantic_association_coverage: float
    unresolved_causal_responsibility: float
    concepts_total: int
    concepts_semantically_grounded: int
    concepts_semantically_associated: int
    causal_concepts_total: int
    causal_concepts_grounded: int
    causal_concepts_associated: int
    unsupported_nodes: int


@dataclass(slots=True)
class FidelityCertificate:
    observational_validation_r2: float
    observational_validation_mae: float
    observational_decision_agreement: float
    interventional_correlation: float | None
    interventional_mae: float | None
    interventional_sign_agreement: float | None
    counterfactual_programme_flip: bool | None
    counterfactual_model_flip: bool | None
    counterfactual_domain_valid: bool | None


@dataclass(slots=True)
class ClaimRecord:
    claim: str
    evidence_tier: str
    supported: bool
    evidence: dict[str, Any]
    limitations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MechanismCertificate:
    schema_version: str
    model_class: str
    target: int
    validity_envelope: ValidityEnvelope
    coverage: CoverageCertificate
    fidelity: FidelityCertificate
    claims: list[ClaimRecord]
    dark_matter: dict[str, float]
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanismCertificate:
        coverage_payload = dict(payload["coverage"])
        coverage_payload.setdefault("semantic_association_coverage", 0.0)
        coverage_payload.setdefault("causal_semantic_association_coverage", 0.0)
        coverage_payload.setdefault("concepts_semantically_associated", 0)
        coverage_payload.setdefault("causal_concepts_associated", 0)
        return cls(
            schema_version=str(payload["schema_version"]),
            model_class=str(payload["model_class"]),
            target=int(payload["target"]),
            validity_envelope=ValidityEnvelope(**payload["validity_envelope"]),
            coverage=CoverageCertificate(**coverage_payload),
            fidelity=FidelityCertificate(**payload["fidelity"]),
            claims=[ClaimRecord(**item) for item in payload.get("claims", ())],
            dark_matter=dict(payload.get("dark_matter", {})),
            warnings=list(payload.get("warnings", ())),
            metadata=dict(payload.get("metadata", {})),
        )


class CertificateBuilder:
    def build(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        causal_graph: CausalMechanismGraph,
        program: MechanismProgram,
        counterfactual: CounterfactualProof | None,
        *,
        target: int,
    ) -> MechanismCertificate:
        validity = self._validity(captured)
        coverage = self._coverage(captured, atlas, causal_graph)
        fidelity = self._fidelity(program, counterfactual)
        claims = self._claims(captured, atlas, causal_graph, program, counterfactual)
        dark_matter = {
            "semantically_grounded_and_causal": coverage.causal_semantic_coverage,
            "semantically_associated_and_causal": coverage.causal_semantic_association_coverage,
            "causally_detected_but_semantically_unresolved": coverage.unresolved_causal_responsibility,
            "semantically_grounded_fraction": coverage.semantic_coverage,
            "semantically_associated_fraction": coverage.semantic_association_coverage,
            "unsupported_operator_fraction": max(0.0, 1.0 - coverage.operator_coverage),
            "intervention_inaccessible_fraction": max(0.0, 1.0 - coverage.intervention_coverage),
        }
        warnings = self._warnings(captured, atlas, program, counterfactual, coverage)
        return MechanismCertificate(
            schema_version="0.1",
            model_class=captured.ir.model_class,
            target=int(target),
            validity_envelope=validity,
            coverage=coverage,
            fidelity=fidelity,
            claims=claims,
            dark_matter=dark_matter,
            warnings=warnings,
            metadata={
                "ir_hash": captured.ir.content_hash(),
                "capture_attempts": [asdict(attempt) for attempt in captured.ir.attempts],
                "concept_method": atlas.metadata.get("method"),
                "programme_rules": len(program.rules),
                "certificate_principle": (
                    "No single trust score is emitted. Independent evidence dimensions must "
                    "be interpreted with their declared limitations."
                ),
            },
        )

    def _validity(self, captured: CapturedModel) -> ValidityEnvelope:
        observed_shapes: dict[str, list[int | str]] = {}
        for node_id in captured.ir.inputs:
            try:
                node = captured.ir.get_node(node_id)
            except KeyError:
                continue
            if node.tensor is not None:
                observed_shapes[node.name] = list(node.tensor.shape)
        unresolved = [
            {
                "id": node.id,
                "name": node.name,
                "target": node.target,
                "reason": node.unresolved_reason,
            }
            for node in captured.ir.unresolved()
        ]
        restrictions = [
            "The certificate applies to the captured checkpoint and observed execution routes only.",
            "Unseen data-dependent branches are not certified.",
            "Concept semantics are post-hoc and may not be identifiable uniquely.",
            "Internal interventions may leave the natural activation manifold.",
        ]
        if captured.ir.capture_route.value != "torch_export":
            restrictions.append(
                "The selected capture route is a fallback route and may omit functional or operator-level computation."
            )
        if unresolved:
            restrictions.append("Unresolved operations remain visible and are excluded from strong semantic claims.")
        return ValidityEnvelope(
            capture_route=captured.ir.capture_route.value,
            observed_input_shapes=observed_shapes,
            shape_constraints=[asdict(item) for item in captured.ir.constraints],
            covered_execution_paths=list(captured.ir.metadata.get("covered_execution_paths", ["observed_path_0"])),
            unresolved_operations=unresolved,
            supported_interventions=[
                "zero",
                "replace",
                "scale",
                "erase_direction",
                "set_concept",
                "set_concept_mean",
                "clamp",
            ] if captured.supports_graph_interventions else [],
            model_state="evaluation",
            framework_version=captured.ir.framework_version,
            numerical_equivalence_passed=captured.ir.coverage.numeric_equivalence_passed,
            numerical_equivalence_max_abs=captured.ir.coverage.numeric_equivalence_max_abs,
            restrictions=restrictions,
        )

    def _coverage(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        causal_graph: CausalMechanismGraph,
    ) -> CoverageCertificate:
        concepts = atlas.concepts
        grounded = [concept for concept in concepts if concept.semantically_grounded]
        associated = [concept for concept in concepts if concept.semantically_associated]
        causal_nodes = list(causal_graph.nodes)
        causal_map = atlas.concept_map()
        grounded_causal = [
            node for node in causal_nodes
            if node.concept_id in causal_map and causal_map[node.concept_id].semantically_grounded
        ]
        associated_causal = [
            node for node in causal_nodes
            if node.concept_id in causal_map and causal_map[node.concept_id].semantically_associated
        ]
        responsibility = np.asarray([abs(node.output_effect) for node in causal_nodes], dtype=float)
        total = float(responsibility.sum())
        if total <= 1e-12:
            weights = np.full(len(causal_nodes), 1.0 / max(1, len(causal_nodes)), dtype=float)
        else:
            weights = responsibility / total
        grounded_mask = np.asarray(
            [
                1.0
                if node.concept_id in causal_map and causal_map[node.concept_id].semantically_grounded
                else 0.0
                for node in causal_nodes
            ],
            dtype=float,
        )
        associated_mask = np.asarray(
            [
                1.0
                if node.concept_id in causal_map and causal_map[node.concept_id].semantically_associated
                else 0.0
                for node in causal_nodes
            ],
            dtype=float,
        )
        causal_semantic = float(np.sum(weights * grounded_mask)) if len(weights) else 0.0
        causal_association = float(np.sum(weights * associated_mask)) if len(weights) else 0.0
        explained_mask = np.maximum(grounded_mask, associated_mask)
        unresolved_responsibility = float(np.sum(weights * (1.0 - explained_mask))) if len(weights) else 0.0
        ir_coverage = captured.ir.coverage
        return CoverageCertificate(
            operator_coverage=ir_coverage.operator_coverage,
            parameter_coverage=ir_coverage.parameter_coverage,
            gradient_coverage=ir_coverage.gradient_coverage,
            intervention_coverage=ir_coverage.intervention_coverage,
            semantic_coverage=float(len(grounded) / len(concepts)) if concepts else 0.0,
            semantic_association_coverage=float(len(associated) / len(concepts)) if concepts else 0.0,
            causal_semantic_coverage=causal_semantic,
            causal_semantic_association_coverage=causal_association,
            unresolved_causal_responsibility=unresolved_responsibility,
            concepts_total=len(concepts),
            concepts_semantically_grounded=len(grounded),
            concepts_semantically_associated=len(associated),
            causal_concepts_total=len(causal_nodes),
            causal_concepts_grounded=len(grounded_causal),
            causal_concepts_associated=len(associated_causal),
            unsupported_nodes=len(captured.ir.unresolved()),
        )

    def _fidelity(
        self,
        program: MechanismProgram,
        counterfactual: CounterfactualProof | None,
    ) -> FidelityCertificate:
        selected = counterfactual.selected if counterfactual is not None else None
        return FidelityCertificate(
            observational_validation_r2=program.fidelity.validation_r2,
            observational_validation_mae=program.fidelity.validation_mae,
            observational_decision_agreement=program.fidelity.validation_decision_agreement,
            interventional_correlation=program.fidelity.interventional_correlation,
            interventional_mae=program.fidelity.interventional_mae,
            interventional_sign_agreement=program.fidelity.interventional_sign_agreement,
            counterfactual_programme_flip=(selected.programme_flip if selected else None),
            counterfactual_model_flip=(selected.model_flip if selected else None),
            counterfactual_domain_valid=(selected.domain_valid if selected else None),
        )

    def _claims(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        causal_graph: CausalMechanismGraph,
        program: MechanismProgram,
        counterfactual: CounterfactualProof | None,
    ) -> list[ClaimRecord]:
        claims = [
            ClaimRecord(
                claim="The executed computation was captured with numerical equivalence.",
                evidence_tier="observed",
                supported=bool(captured.ir.coverage.numeric_equivalence_passed),
                evidence={
                    "route": captured.ir.capture_route.value,
                    "maximum_absolute_error": captured.ir.coverage.numeric_equivalence_max_abs,
                },
                limitations=["Only observed inputs and covered paths were tested."],
            ),
            ClaimRecord(
                claim="Candidate internal concepts were recovered from intermediate representations.",
                evidence_tier="observed",
                supported=bool(atlas.concepts),
                evidence={
                    "concepts": len(atlas.concepts),
                    "nodes": len(atlas.concept_sets),
                    "method": atlas.metadata.get("method"),
                },
                limitations=["Candidate latent factors are not automatically human concepts."],
            ),
            ClaimRecord(
                claim="At least one internal concept has intervention-supported output relevance.",
                evidence_tier="intervention_supported",
                supported=any(node.evidence_tier != "observed_only" for node in causal_graph.nodes),
                evidence={
                    "intervention_supported_nodes": sum(
                        node.evidence_tier != "observed_only" for node in causal_graph.nodes
                    )
                },
                limitations=["Effects depend on intervention baseline and activation-manifold validity."],
            ),
            ClaimRecord(
                claim="The extracted programme reproduces held-out target margins.",
                evidence_tier="replicated",
                supported=(
                    program.fidelity.validation_decision_agreement >= 0.80
                    and program.fidelity.validation_r2 > 0.0
                ),
                evidence={
                    "validation_r2": program.fidelity.validation_r2,
                    "validation_mae": program.fidelity.validation_mae,
                    "decision_agreement": program.fidelity.validation_decision_agreement,
                },
                limitations=["Observational fidelity alone does not establish mechanism recovery."],
            ),
            ClaimRecord(
                claim="The extracted programme predicts the direction of tested internal interventions.",
                evidence_tier="intervention_supported",
                supported=(
                    program.fidelity.interventional_sign_agreement is not None
                    and program.fidelity.interventional_sign_agreement >= 0.70
                ),
                evidence={
                    "interventional_sign_agreement": program.fidelity.interventional_sign_agreement,
                    "interventional_correlation": program.fidelity.interventional_correlation,
                },
                limitations=["Only the tested interventions are covered."],
            ),
        ]
        selected = counterfactual.selected if counterfactual is not None else None
        claims.append(
            ClaimRecord(
                claim="A programme-predicted concept counterfactual changed the original model decision.",
                evidence_tier="intervention_supported",
                supported=bool(selected and selected.model_flip),
                evidence=(selected.to_dict() if selected is not None else {}),
                limitations=[
                    "Internal concept changes are not automatically realisable as valid raw-input changes.",
                    "Domain validation is reported separately.",
                ],
            )
        )
        return claims

    def _warnings(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        program: MechanismProgram,
        counterfactual: CounterfactualProof | None,
        coverage: CoverageCertificate,
    ) -> list[str]:
        warnings: list[str] = []
        if not captured.ir.coverage.numeric_equivalence_passed:
            warnings.append("Captured graph did not pass numerical equivalence validation.")
        if coverage.operator_coverage < 0.95:
            warnings.append("Operator semantics coverage is below 95 per cent.")
        if coverage.causal_semantic_coverage < 0.50:
            warnings.append(
                "Most intervention-tested responsibility lacks independently validated semantic grounding."
            )
        if program.fidelity.validation_r2 < 0.50:
            warnings.append("The executable programme has weak held-out observational fidelity.")
        if (
            program.fidelity.interventional_sign_agreement is None
            or program.fidelity.interventional_sign_agreement < 0.70
        ):
            warnings.append("Interventional programme fidelity is not yet strong enough for a mechanism claim.")
        if counterfactual is None or counterfactual.selected is None:
            warnings.append("No counterfactual proof was verified within the configured search budget.")
        elif counterfactual.selected.domain_valid is not True:
            warnings.append("The counterfactual was not independently validated as domain-valid.")
        if not atlas.concepts:
            warnings.append("No internal concepts were discovered.")
        return warnings
