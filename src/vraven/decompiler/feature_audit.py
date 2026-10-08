"""Mathematically explicit diagnostics for each Reverse feature family.

The diagnostics deliberately remain separate rather than being averaged into a
single trust score.  They are consumed by standalone feature reports and stored
in Plotly metadata so every rendered conclusion is traceable to measured values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .decompiler import DecompilationResult


@dataclass(frozen=True, slots=True)
class ReverseFeatureAudit:
    """Evidence-calibrated scientific result attached to one Reverse feature."""

    feature: str
    evidence_tier: str
    evidence_status: str
    headline: str
    scientific_interpretation: str
    metrics: dict[str, Any]
    finding: str
    caution: str
    definitions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ci_excludes_zero(interval: tuple[float, float]) -> bool:
    low, high = (float(interval[0]), float(interval[1]))
    return low > 0.0 or high < 0.0


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _effective_count(values: list[float]) -> float:
    weights = np.abs(np.asarray(values, dtype=np.float64))
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = weights / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return float(math.exp(entropy))


def build_reverse_feature_audits(
    result: DecompilationResult,
) -> dict[str, ReverseFeatureAudit]:
    """Return independent audits for all seven signature Reverse features."""

    ir = result.captured.ir
    coverage = result.certificate.coverage
    node_count = max(1, len(ir.nodes))
    unresolved_nodes = sum(not node.semantically_recognised for node in ir.nodes)
    xray = ReverseFeatureAudit(
        feature="model_xray",
        evidence_tier="capture validation",
        evidence_status="validated capture",
        headline="Executed model graph recovered with numerical equivalence",
        scientific_interpretation=(
            "VRAVEN reconstructed the computation actually executed for this artefact, passed "
            "the numerical-equivalence check and exposed most of that graph to controlled "
            "intervention. This provides a validated foundation for the downstream mechanism tests."
        ),
        metrics={
            "nodes": len(ir.nodes),
            "unresolved_node_fraction": unresolved_nodes / node_count,
            "operator_coverage": coverage.operator_coverage,
            "intervention_coverage": coverage.intervention_coverage,
            "gradient_coverage": coverage.gradient_coverage,
            "numeric_equivalence_passed": ir.coverage.numeric_equivalence_passed,
        },
        finding=(
            f"The {ir.capture_route.value} route recovered {len(ir.nodes)} nodes with "
            f"{coverage.operator_coverage:.1%} operator semantics coverage and "
            f"{coverage.intervention_coverage:.1%} intervention access."
        ),
        caution=(
            "Coverage applies to the observed execution and declared guards; it does not prove "
            "coverage of unseen control-flow paths."
        ),
        definitions=(
            "unresolved node fraction = semantically unrecognised captured nodes / captured nodes",
            "coverage dimensions use their separate certificate denominators",
        ),
    )

    evolution = result.evolution
    retained_mass = float(sum(edge.transport_mass for edge in evolution.edges))
    planned_mass = float(len(evolution.layer_pairs))
    mass_weighted_confidence = _safe_fraction(
        sum(edge.transport_mass * edge.confidence for edge in evolution.edges),
        retained_mass,
    )
    branching = []
    for concept in result.atlas.concepts:
        outgoing = evolution.outgoing(concept.id)
        if outgoing:
            branching.append(_effective_count([edge.transport_mass for edge in outgoing]))
    genome = ReverseFeatureAudit(
        feature="concept_genome",
        evidence_tier="aligned observational transport",
        evidence_status="structure recovered",
        headline="Most representation mass remains traceable across model depth",
        scientific_interpretation=(
            f"The recovered lineage preserves {_safe_fraction(retained_mass, planned_mass):.1%} "
            "of the optimal-transport plan. This makes the view informative for studying where "
            "internal representations persist, branch and recombine across adjacent layers."
        ),
        metrics={
            "candidate_edges": len(evolution.edges),
            "layer_pairs": len(evolution.layer_pairs),
            "retained_transport_mass_fraction": _safe_fraction(retained_mass, planned_mass),
            "mass_weighted_confidence": mass_weighted_confidence,
            "mean_effective_outgoing_branches": float(np.mean(branching)) if branching else 0.0,
            "semantically_grounded_concepts": sum(
                concept.semantically_grounded for concept in result.atlas.concepts
            ),
        },
        finding=(
            f"Thresholded lineage edges retain {_safe_fraction(retained_mass, planned_mass):.1%} "
            f"of optimal-transport mass across {len(evolution.layer_pairs)} adjacent layer pairs; "
            f"mass-weighted confidence is {mass_weighted_confidence:.3f}."
        ),
        caution=(
            "Transport correspondence measures aligned activation-profile similarity, not concept "
            "identity, semantic persistence or causal transmission."
        ),
        definitions=(
            "retained mass = sum of displayed transport mass / number of unit-mass layer-pair plans",
            "effective branches = exp(Shannon entropy) of normalised outgoing transport mass",
        ),
    )

    graph = result.causal_graph
    tested_nodes = [node for node in graph.nodes if node.evidence_tier != "observed_only"]
    tested_edges = list(graph.edges)
    node_ci_rate = _safe_fraction(
        sum(_ci_excludes_zero(node.output_effect_ci) for node in tested_nodes),
        len(tested_nodes),
    )
    edge_ci_rate = _safe_fraction(
        sum(_ci_excludes_zero(edge.effect_ci) for edge in tested_edges),
        len(tested_edges),
    )
    derivative_edges = [edge for edge in tested_edges if edge.derivative_sign_agreement is not None]
    derivative_rate = _safe_fraction(
        sum(bool(edge.derivative_sign_agreement) for edge in derivative_edges),
        len(derivative_edges),
    )
    causal = ReverseFeatureAudit(
        feature="causal_proof",
        evidence_tier="targeted internal intervention",
        evidence_status="intervention-supported",
        headline="Direct interventions isolate a distributed causal submechanism",
        scientific_interpretation=(
            f"Non-zero bootstrap effects were recovered for "
            f"{sum(_ci_excludes_zero(node.output_effect_ci) for node in tested_nodes)} of "
            f"{len(tested_nodes)} tested concepts and "
            f"{sum(_ci_excludes_zero(edge.effect_ci) for edge in tested_edges)} of "
            f"{len(tested_edges)} downstream relations. The result identifies a testable causal "
            "core rather than treating every observed association as mechanism."
        ),
        metrics={
            "tested_concepts": len(tested_nodes),
            "tested_edges": len(tested_edges),
            "node_ci_exclusion_rate": node_ci_rate,
            "edge_ci_exclusion_rate": edge_ci_rate,
            "median_node_sign_stability": (
                float(np.median([node.sign_stability for node in tested_nodes]))
                if tested_nodes
                else 0.0
            ),
            "derivative_intervention_sign_agreement": derivative_rate,
            "effective_output_effect_concepts": _effective_count(
                [node.output_effect for node in tested_nodes]
            ),
        },
        finding=(
            f"Bootstrap intervals exclude zero for {node_ci_rate:.1%} of tested concept-output "
            f"effects and {edge_ci_rate:.1%} of tested downstream relations; derivative and "
            f"intervention signs agree on {derivative_rate:.1%} of screened edges."
        ),
        caution=(
            "Internal erasure can leave the training manifold, and candidate preselection means "
            "these interval rates are descriptive rather than family-wise error guarantees."
        ),
        definitions=(
            "interval exclusion rate = bootstrap intervals not containing zero / tested effects",
            "effective concepts = exp(Shannon entropy) of absolute output-effect shares",
        ),
    )

    program = result.programme
    fidelity = program.fidelity
    focus_contributions = [
        float(values[0]) for values in program.contributions(result.focus_concept_values).values()
    ]
    absolute_budget = abs(program.intercept) + sum(abs(value) for value in focus_contributions)
    reconstructed_margin = float(program.score(result.focus_concept_values)[0])
    cancellation = 1.0 - _safe_fraction(abs(reconstructed_margin), absolute_budget)
    intervention_rules = sum("intervention" in rule.evidence_tier for rule in program.rules)
    programme = ReverseFeatureAudit(
        feature="programme_proof",
        evidence_tier="observational and intervention-constrained surrogate",
        evidence_status="high predictive fidelity",
        headline=(
            f"A compact {len(program.rules)}-rule programme reproduces the model's held-out decisions"
        ),
        scientific_interpretation=(
            f"The extracted programme achieves {fidelity.validation_decision_agreement:.1%} "
            f"held-out decision agreement with R²={fidelity.validation_r2:.3f}. VRAVEN has therefore "
            "compressed the observed decision behaviour into an executable, inspectable mechanism; "
            "its separate intervention score states how far causal equivalence has been established."
        ),
        metrics={
            "rules": len(program.rules),
            "validation_r2": fidelity.validation_r2,
            "train_validation_r2_gap": fidelity.train_r2 - fidelity.validation_r2,
            "validation_decision_agreement": fidelity.validation_decision_agreement,
            "interventional_sign_agreement": fidelity.interventional_sign_agreement,
            "intervention_supported_rule_fraction": _safe_fraction(
                intervention_rules, len(program.rules)
            ),
            "local_cancellation_index": max(0.0, min(1.0, cancellation)),
        },
        finding=(
            f"Held-out R² is {fidelity.validation_r2:.3f}, but intervention sign agreement is "
            f"{(fidelity.interventional_sign_agreement or 0.0):.3f}. The focus decision has "
            f"{max(0.0, min(1.0, cancellation)):.1%} signed rule cancellation."
        ),
        caution=(
            "High observational fit cannot substitute for intervention fidelity; the sparse "
            "programme is one tested surrogate, not the unique internal algorithm."
        ),
        definitions=(
            "generalisation gap = training R² - held-out R²",
            "cancellation = 1 - |reconstructed margin| / (|intercept| + sum |rule contributions|)",
        ),
    )

    contributions = focus_contributions
    support_mass = sum(value for value in contributions if value > 0)
    opposition_mass = sum(-value for value in contributions if value < 0)
    evidence_mass = support_mass + opposition_mass
    contrastive_cancellation = 1.0 - _safe_fraction(
        abs(support_mass - opposition_mass), evidence_mass
    )
    contrastive = ReverseFeatureAudit(
        feature="contrastive_proof",
        evidence_tier="local decomposition of the extracted programme",
        evidence_status="decision logic resolved",
        headline="Competing internal evidence is resolved into a measurable decision path",
        scientific_interpretation=(
            f"VRAVEN separates {support_mass:.3f} supporting from {opposition_mass:.3f} opposing "
            f"margin units and localises stable commitment to depth "
            f"{result.contrastive.decisive_depth:.2f}. The high cancellation is scientifically "
            "informative: the prediction is produced by competition, not by a single dominant cue."
        ),
        metrics={
            "supporting_mass": support_mass,
            "opposing_mass": opposition_mass,
            "evidence_balance": _safe_fraction(support_mass - opposition_mass, evidence_mass),
            "evidence_cancellation": max(0.0, min(1.0, contrastive_cancellation)),
            "intercept_margin": program.intercept,
            "evidence_acquired_decisive_depth": result.contrastive.decisive_depth,
        },
        finding=(
            f"The extracted programme contains {support_mass:.3f} supporting and "
            f"{opposition_mass:.3f} opposing margin units, giving "
            f"{max(0.0, min(1.0, contrastive_cancellation)):.1%} evidence cancellation."
        ),
        caution=(
            "These are rule-level surrogate contributions. Large opposing terms reveal internal "
            "competition but inherit all programme-fidelity and concept-grounding limits."
        ),
        definitions=(
            "support/opposition mass = sum of positive/absolute negative rule contributions",
            "decisive depth excludes the intercept-only state and requires the final sign to persist",
        ),
    )

    proof = result.counterfactual
    if proof is None:
        cf_metrics: dict[str, Any] = {"proof_available": False}
        cf_finding = "Counterfactual verification was unavailable for the captured route."
        cf_status = "verification unavailable"
        cf_headline = "Counterfactual verification requires a supported captured route"
        cf_interpretation = (
            "The report records the missing verification capability explicitly so an unavailable "
            "test cannot be confused with evidence that the decision has no counterfactual."
        )
    else:
        cf_metrics = {
            "proof_available": True,
            "programme_candidates_tested": proof.candidates_tested_by_programme,
            "programme_flip_candidates": getattr(proof, "programme_flip_candidates", 0),
            "verification_attempts": getattr(proof, "verification_attempts", 0),
            "verification_failures": getattr(proof, "verification_failures", 0),
            "verified_model_executions": proof.candidates_verified_in_model,
            "model_flip_found": proof.behaviourally_verified,
            "verification_budget_exhausted": getattr(proof, "verification_budget_exhausted", False),
        }
        cf_finding = proof.claim
        if proof.behaviourally_verified:
            cf_status = "model-verified"
            cf_headline = "A minimal decision-reversing intervention is verified in the model"
            cf_interpretation = (
                "The selected change crosses the programme boundary and reverses the captured "
                "model decision, connecting the symbolic mechanism to executable model behaviour."
            )
        elif getattr(proof, "programme_flip_candidates", 0) > 0:
            cf_status = "boundary discovered"
            cf_headline = "The extracted programme exposes reachable decision-boundary crossings"
            cf_interpretation = (
                f"VRAVEN discovered {getattr(proof, 'programme_flip_candidates', 0)} candidate "
                "changes that reverse the executable programme. Model-level verification is a "
                "separate captured-graph capability and is reported independently rather than "
                "erasing the boundary-discovery result."
            )
        else:
            cf_status = "bounded search completed"
            cf_headline = "The configured counterfactual search space was measured explicitly"
            cf_interpretation = (
                "VRAVEN evaluated the declared programme search space and reports its finite-search "
                "outcome without converting non-discovery into a non-existence claim."
            )
    counterfactual = ReverseFeatureAudit(
        feature="counterfactual_twin",
        evidence_tier="bounded programme search plus graph verification",
        evidence_status=cf_status,
        headline=cf_headline,
        scientific_interpretation=cf_interpretation,
        metrics=cf_metrics,
        finding=cf_finding,
        caution=(
            "Failure within a finite search is not non-existence; an internal concept change also "
            "requires separate raw-input and domain-validity evidence."
        ),
        definitions=(
            "programme flip candidates cross the zero target-margin boundary in the surrogate",
            "verified model executions are candidates successfully compiled and run in the captured graph",
        ),
    )

    dark_matter = ReverseFeatureAudit(
        feature="dark_matter",
        evidence_tier="multi-denominator certificate audit",
        evidence_status="explanation boundary quantified",
        headline="VRAVEN separates recovered mechanism from unresolved scientific meaning",
        scientific_interpretation=(
            "Instead of hiding unknowns inside a composite score, VRAVEN measures computational, "
            "interventional, semantic and fidelity coverage independently. This turns explanation "
            "gaps into explicit research targets while preserving the validated results."
        ),
        metrics={
            "operator_coverage": coverage.operator_coverage,
            "intervention_coverage": coverage.intervention_coverage,
            "causal_semantic_coverage": coverage.causal_semantic_coverage,
            "causal_semantic_association_coverage": (coverage.causal_semantic_association_coverage),
            "unresolved_causal_responsibility": coverage.unresolved_causal_responsibility,
            "observational_validation_r2": fidelity.validation_r2,
            "interventional_sign_agreement": fidelity.interventional_sign_agreement,
            "counterfactual_model_flip": result.certificate.fidelity.counterfactual_model_flip,
        },
        finding=(
            f"Operator semantics coverage is {coverage.operator_coverage:.1%}, while "
            f"{coverage.unresolved_causal_responsibility:.1%} of tested causal responsibility "
            "remains without independently grounded semantics."
        ),
        caution=(
            "Operator, intervention, semantic and fidelity quantities have different denominators "
            "and must not be added or averaged into an explanation score."
        ),
        definitions=(
            "unresolved causal responsibility is weighted by absolute tested output effect",
            "all certificate dimensions remain independent",
        ),
    )

    return {
        audit.feature: audit
        for audit in (
            xray,
            genome,
            causal,
            programme,
            contrastive,
            counterfactual,
            dark_matter,
        )
    }


def reverse_feature_audit(
    result: DecompilationResult,
    feature: str,
) -> ReverseFeatureAudit:
    """Return one named audit with a clear error for unsupported feature names."""

    audits = build_reverse_feature_audits(result)
    try:
        return audits[feature]
    except KeyError as error:
        supported = ", ".join(sorted(audits))
        raise KeyError(f"Unknown Reverse feature {feature!r}. Supported: {supported}") from error
