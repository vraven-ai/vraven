"""Concept-level counterfactual search with behavioural and mechanistic verification.

The search operates on concepts discovered from the model's own internal representations.
Candidate changes are first screened with the executable VRAVEN programme and are then
compiled into the captured FX/Export graph.  A result is only described as verified when
the original model changes in the predicted direction.  Domain validity is a separate,
explicit contract because a mathematically small change is not necessarily meaningful in
the user's application domain.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .capture import CapturedModel
from .causal_graph import CausalMechanismGraph
from .concepts import Concept, ConceptAtlas
from .interventions import GraphInterventionCompiler, InterventionKind, InterventionSpec
from .programme import MechanismProgram
from .runtime import FXValueRecorder, extract_logits, predicted_class, target_margin

DomainValidator = Callable[[Mapping[str, float], Mapping[str, float]], tuple[bool, str] | bool]


@dataclass(slots=True)
class CounterfactualConfig:
    """Controls bounded search over concept replacements."""

    candidate_quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    maximum_concepts: int = 10
    maximum_changes: int = 2
    maximum_verified_candidates: int = 40
    programme_flip_margin: float = 0.0
    require_model_flip: bool = True
    cost_power: float = 1.0
    minimum_scale: float = 1e-6
    output_path: str | Sequence[str | int] | None = None
    seed: int = 17


@dataclass(slots=True)
class ConceptChange:
    concept_id: str
    label: str
    node_id: str
    original_value: float
    replacement_value: float
    normalised_cost: float
    programme_contribution_before: float
    programme_contribution_after: float
    causal_output_effect: float | None = None
    semantic_status: str = "unresolved"


@dataclass(slots=True)
class CounterfactualCandidate:
    id: str
    changes: list[ConceptChange]
    total_cost: float
    programme_margin_before: float
    programme_margin_after: float
    programme_flip: bool
    model_margin_before: float | None = None
    model_margin_after: float | None = None
    model_prediction_before: int | None = None
    model_prediction_after: int | None = None
    model_flip: bool | None = None
    domain_valid: bool | None = None
    domain_validation_note: str = "No domain validator supplied."
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class CounterfactualProof:
    target: int
    baseline_prediction: int
    baseline_margin: float
    baseline_programme_margin: float
    selected: CounterfactualCandidate | None
    candidates_tested_by_programme: int
    candidates_verified_in_model: int
    search_exhausted: bool
    claim: str
    limitations: tuple[str, ...]
    programme_flip_candidates: int = 0
    verification_attempts: int = 0
    verification_failures: int = 0
    verification_budget_exhausted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.selected is not None

    @property
    def behaviourally_verified(self) -> bool:
        return bool(self.selected and self.selected.model_flip)

    @property
    def domain_verified(self) -> bool:
        return bool(self.selected and self.selected.domain_valid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "baseline_prediction": self.baseline_prediction,
            "baseline_margin": self.baseline_margin,
            "baseline_programme_margin": self.baseline_programme_margin,
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "candidates_tested_by_programme": self.candidates_tested_by_programme,
            "candidates_verified_in_model": self.candidates_verified_in_model,
            "search_exhausted": self.search_exhausted,
            "claim": self.claim,
            "limitations": list(self.limitations),
            "programme_flip_candidates": self.programme_flip_candidates,
            "verification_attempts": self.verification_attempts,
            "verification_failures": self.verification_failures,
            "verification_budget_exhausted": self.verification_budget_exhausted,
            "metadata": self.metadata,
        }


class CounterfactualProofEngine:
    """Search for a compact concept intervention and verify it in the original graph."""

    def __init__(self, config: CounterfactualConfig | None = None) -> None:
        self.config = config or CounterfactualConfig()

    def prove(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        program: MechanismProgram,
        causal_graph: CausalMechanismGraph,
        sample_args: Sequence[Any],
        *,
        sample_kwargs: Mapping[str, Any] | None = None,
        target: int,
        domain_validator: DomainValidator | None = None,
    ) -> CounterfactualProof:
        if captured.graph_module is None:
            raise ValueError("Counterfactual proof requires an export or FX graph capture")
        kwargs = dict(sample_kwargs or {})
        baseline_record = FXValueRecorder(
            captured.graph_module, detach=False, move_to_cpu=False
        ).execute(tuple(sample_args), kwargs)
        baseline_logits = extract_logits(baseline_record.output, self.config.output_path)
        if int(baseline_logits.shape[0]) != 1:
            raise ValueError("Counterfactual proof currently requires a single focus sample")
        baseline_margin = float(target_margin(baseline_logits, target).detach().cpu().item())
        baseline_prediction = int(predicted_class(baseline_logits).detach().cpu().item())
        concept_values = _sample_concept_values(
            atlas,
            program.concept_ids,
            baseline_record.values,
        )
        baseline_programme_margin = float(program.score(concept_values)[0])
        programme_contributions_before = {
            key: float(value[0]) for key, value in program.contributions(concept_values).items()
        }

        candidate_concepts = self._select_candidate_concepts(
            atlas,
            program,
            causal_graph,
            concept_values,
        )
        replacements = self._replacement_candidates(
            atlas,
            candidate_concepts,
            program,
            concept_values,
        )
        programme_index = {
            concept_id: position for position, concept_id in enumerate(program.concept_ids)
        }
        screened: list[CounterfactualCandidate] = []
        tested = 0
        for size in range(1, min(self.config.maximum_changes, len(candidate_concepts)) + 1):
            for concept_tuple in itertools.combinations(candidate_concepts, size):
                replacement_lists = [replacements[concept.id] for concept in concept_tuple]
                for values in itertools.product(*replacement_lists):
                    replacement_map = {
                        concept.id: float(value)
                        for concept, value in zip(concept_tuple, values, strict=True)
                    }
                    if all(
                        math.isclose(
                            replacement_map[concept.id],
                            float(concept_values[0, programme_index[concept.id]]),
                            abs_tol=1e-12,
                        )
                        for concept in concept_tuple
                    ):
                        continue
                    tested += 1
                    after = float(program.intervene(concept_values, replacement_map)[0])
                    flips = _sign_flip(
                        baseline_programme_margin,
                        after,
                        threshold=self.config.programme_flip_margin,
                    )
                    if not flips:
                        continue
                    changes = self._build_changes(
                        atlas,
                        program,
                        causal_graph,
                        concept_values,
                        replacement_map,
                        programme_contributions_before,
                    )
                    total_cost = float(
                        sum(change.normalised_cost**self.config.cost_power for change in changes)
                    )
                    screened.append(
                        CounterfactualCandidate(
                            id=f"cf_{len(screened):04d}",
                            changes=changes,
                            total_cost=total_cost,
                            programme_margin_before=baseline_programme_margin,
                            programme_margin_after=after,
                            programme_flip=True,
                        )
                    )
        screened.sort(key=lambda item: (item.total_cost, abs(item.programme_margin_after)))

        compiler = GraphInterventionCompiler(captured)
        verified_count = 0
        verification_attempts = 0
        verification_failures = 0
        selected: CounterfactualCandidate | None = None
        for candidate in screened[: self.config.maximum_verified_candidates]:
            verification_attempts += 1
            specs = []
            for change in candidate.changes:
                concept = atlas.concept_map()[change.concept_id]
                concept_set = atlas.set_for_node(concept.node_id)
                try:
                    raw_coefficient = concept_set.raw_projection_for_score(
                        concept.index,
                        change.replacement_value,
                    )
                except ValueError as error:
                    candidate.metadata.setdefault("unsupported_concept_interventions", []).append(
                        f"{concept.id}: {error}"
                    )
                    specs = []
                    break
                specs.append(
                    InterventionSpec(
                        node_id=concept.node_id,
                        kind=InterventionKind.SET_CONCEPT_MEAN,
                        label=f"counterfactual:{concept.id}",
                        feature_axis=concept.feature_axis,
                        direction=concept.direction,
                        coefficient=raw_coefficient,
                        metadata={
                            "concept_id": concept.id,
                            "counterfactual": True,
                            "programme_score_replacement": change.replacement_value,
                            "raw_projection_coefficient": raw_coefficient,
                        },
                    )
                )
            if not specs:
                verification_failures += 1
                continue
            try:
                compiled = compiler.compile(specs)
                with torch.no_grad():
                    output = compiled(*tuple(sample_args), **kwargs)
                logits_after = extract_logits(output, self.config.output_path)
                margin_after = float(target_margin(logits_after, target).detach().cpu().item())
                prediction_after = int(predicted_class(logits_after).detach().cpu().item())
                model_flip = prediction_after != baseline_prediction
                candidate.model_margin_before = baseline_margin
                candidate.model_margin_after = margin_after
                candidate.model_prediction_before = baseline_prediction
                candidate.model_prediction_after = prediction_after
                candidate.model_flip = model_flip
                original_map = {
                    change.concept_id: change.original_value for change in candidate.changes
                }
                replacement_map = {
                    change.concept_id: change.replacement_value for change in candidate.changes
                }
                valid, note = _validate_domain(domain_validator, original_map, replacement_map)
                candidate.domain_valid = valid
                candidate.domain_validation_note = note
                candidate.verified = bool(model_flip and (valid is not False))
                candidate.metadata.update(
                    {
                        "graph_intervention_nodes": compiled.gate_nodes,
                        "mechanistic_prediction_correct": bool(
                            _sign_flip(baseline_margin, margin_after, threshold=0.0)
                        ),
                    }
                )
                verified_count += 1
                if self.config.require_model_flip and not model_flip:
                    continue
                if valid is False:
                    continue
                selected = candidate
                break
            except Exception as error:
                verification_failures += 1
                candidate.metadata["verification_error"] = f"{type(error).__name__}: {error}"

        if selected is not None:
            claim = "A concept-level counterfactual was verified in the captured model graph."
        elif not screened:
            claim = (
                "No programme-level boundary crossing was found among the empirical and analytic "
                "concept candidates."
            )
        elif verified_count == 0 and verification_failures:
            claim = (
                f"{len(screened)} programme-level boundary crossings were found, but none of the "
                "attempted candidates could be successfully evaluated in the captured model graph."
            )
        else:
            claim = (
                f"{len(screened)} programme-level boundary crossings were found, but no tested "
                "candidate produced an accepted model-level counterfactual."
            )
        return CounterfactualProof(
            target=int(target),
            baseline_prediction=baseline_prediction,
            baseline_margin=baseline_margin,
            baseline_programme_margin=baseline_programme_margin,
            selected=selected,
            candidates_tested_by_programme=tested,
            candidates_verified_in_model=verified_count,
            search_exhausted=(len(screened) <= self.config.maximum_verified_candidates),
            claim=claim,
            limitations=(
                "Concept interventions modify internal representations and may not correspond to a reachable raw input.",
                "Domain validity is not inferred automatically; a domain validator must approve application-specific constraints.",
                "The search is bounded and does not prove global minimality.",
                "A successful behavioural flip supports the tested intervention, not a unique causal explanation.",
            ),
            programme_flip_candidates=len(screened),
            verification_attempts=verification_attempts,
            verification_failures=verification_failures,
            verification_budget_exhausted=(len(screened) > self.config.maximum_verified_candidates),
            metadata={
                "candidate_quantiles": list(self.config.candidate_quantiles),
                "maximum_changes": self.config.maximum_changes,
                "maximum_verified_candidates": self.config.maximum_verified_candidates,
                "capture_route": captured.ir.capture_route.value,
                "candidate_generation": (
                    "empirical score quantiles plus exact one-coordinate programme-boundary roots "
                    "when the root lies inside the empirical 1st–99th percentile interval"
                ),
            },
        )

    def _select_candidate_concepts(
        self,
        atlas: ConceptAtlas,
        program: MechanismProgram,
        causal_graph: CausalMechanismGraph,
        concept_values: np.ndarray,
    ) -> list[Concept]:
        concept_map = atlas.concept_map()
        node_map = causal_graph.node_map()
        contributions = program.contributions(concept_values)
        rule_map = {rule.id: rule for rule in program.rules}
        importance: dict[str, float] = {concept_id: 0.0 for concept_id in program.concept_ids}
        for rule_id, values in contributions.items():
            magnitude = abs(float(values[0]))
            rule = rule_map[rule_id]
            share = magnitude / max(1, len(rule.concept_ids))
            for concept_id in rule.concept_ids:
                importance[concept_id] = importance.get(concept_id, 0.0) + share
        for concept_id, node in node_map.items():
            importance[concept_id] = importance.get(concept_id, 0.0) + abs(node.output_effect)
        ordered = sorted(
            (
                concept_map[concept_id]
                for concept_id in program.concept_ids
                if concept_id in concept_map
            ),
            key=lambda concept: importance.get(concept.id, 0.0),
            reverse=True,
        )
        return ordered[: self.config.maximum_concepts]

    def _replacement_candidates(
        self,
        atlas: ConceptAtlas,
        concepts: Sequence[Concept],
        program: MechanismProgram,
        concept_values: np.ndarray,
    ) -> dict[str, list[float]]:
        output: dict[str, list[float]] = {}
        for concept in concepts:
            scores = np.asarray(
                atlas.set_for_node(concept.node_id).score_for(concept.id), dtype=float
            )
            finite = scores[np.isfinite(scores)]
            if finite.size == 0:
                output[concept.id] = [concept.mean_activation]
                continue
            values = [
                float(np.quantile(finite, quantile)) for quantile in self.config.candidate_quantiles
            ]
            values.append(float(np.mean(finite)))
            lower = float(np.quantile(finite, 0.01))
            upper = float(np.quantile(finite, 0.99))
            values.extend(
                _analytic_boundary_candidates(
                    program,
                    concept_values,
                    concept.id,
                    lower=lower,
                    upper=upper,
                )
            )
            # Deterministic de-duplication while retaining sorted search values.
            output[concept.id] = sorted({round(value, 10) for value in values})
        return output

    def _build_changes(
        self,
        atlas: ConceptAtlas,
        program: MechanismProgram,
        causal_graph: CausalMechanismGraph,
        concept_values: np.ndarray,
        replacements: Mapping[str, float],
        contributions_before: Mapping[str, float],
    ) -> list[ConceptChange]:
        concept_index = {concept_id: index for index, concept_id in enumerate(program.concept_ids)}
        causal_nodes = causal_graph.node_map()
        changed_matrix = concept_values.copy()
        for concept_id, value in replacements.items():
            changed_matrix[:, concept_index[concept_id]] = value
        after_contributions = {
            key: float(value[0]) for key, value in program.contributions(changed_matrix).items()
        }
        rule_map = {rule.id: rule for rule in program.rules}
        before_by_concept = {concept_id: 0.0 for concept_id in replacements}
        after_by_concept = {concept_id: 0.0 for concept_id in replacements}
        for rule_id, rule in rule_map.items():
            share = 1.0 / max(1, len(rule.concept_ids))
            for concept_id in rule.concept_ids:
                if concept_id in replacements:
                    before_by_concept[concept_id] += contributions_before.get(rule_id, 0.0) * share
                    after_by_concept[concept_id] += after_contributions.get(rule_id, 0.0) * share
        changes: list[ConceptChange] = []
        concept_map = atlas.concept_map()
        for concept_id, replacement in replacements.items():
            concept = concept_map[concept_id]
            original = float(concept_values[0, concept_index[concept_id]])
            scores = np.asarray(
                atlas.set_for_node(concept.node_id).score_for(concept.id), dtype=float
            )
            scale = max(
                float(np.nanpercentile(scores, 95) - np.nanpercentile(scores, 5)),
                self.config.minimum_scale,
            )
            if concept.semantically_grounded:
                semantic_status = "grounded"
            elif concept.semantically_associated:
                semantic_status = "association_only"
            else:
                semantic_status = "unresolved"
            changes.append(
                ConceptChange(
                    concept_id=concept_id,
                    label=concept.display_label,
                    node_id=concept.node_id,
                    original_value=original,
                    replacement_value=float(replacement),
                    normalised_cost=abs(float(replacement) - original) / scale,
                    programme_contribution_before=float(before_by_concept[concept_id]),
                    programme_contribution_after=float(after_by_concept[concept_id]),
                    causal_output_effect=(
                        float(causal_nodes[concept_id].output_effect)
                        if concept_id in causal_nodes
                        else None
                    ),
                    semantic_status=semantic_status,
                )
            )
        changes.sort(key=lambda item: item.normalised_cost, reverse=True)
        return changes


def _sample_concept_values(
    atlas: ConceptAtlas,
    concept_ids: Sequence[str],
    recorded_values: Mapping[str, Any],
) -> np.ndarray:
    concept_map = atlas.concept_map()
    values: list[float] = []
    for concept_id in concept_ids:
        concept = concept_map[concept_id]
        raw = recorded_values.get(concept.node_id)
        tensor = _first_tensor(raw)
        if tensor is None:
            raise ValueError(f"No tensor was recorded for concept node {concept.node_id!r}")
        coefficient = atlas.set_for_node(concept.node_id).score_component(
            tensor,
            concept.index,
        )
        values.append(float(coefficient.detach().mean().cpu().item()))
    return np.asarray(values, dtype=float).reshape(1, -1)


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            result = _first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    return None


def _validate_domain(
    validator: DomainValidator | None,
    original: Mapping[str, float],
    replacement: Mapping[str, float],
) -> tuple[bool | None, str]:
    if validator is None:
        return (
            None,
            "No domain validator supplied; internal counterfactual validity is unconfirmed.",
        )
    result = validator(original, replacement)
    if isinstance(result, tuple):
        return bool(result[0]), str(result[1])
    return bool(result), "Domain validator returned a Boolean result."


def _sign_flip(before: float, after: float, *, threshold: float) -> bool:
    return (before > threshold and after <= threshold) or (
        before <= threshold and after > threshold
    )


def _analytic_boundary_candidates(
    program: MechanismProgram,
    concept_values: np.ndarray,
    concept_id: str,
    *,
    lower: float,
    upper: float,
) -> list[float]:
    """Find the exact one-coordinate surrogate boundary inside an empirical interval.

    With all other concepts fixed, the current programme is affine in one concept
    because it contains main effects and pairwise products but no squared terms.
    Evaluating the endpoints therefore identifies the slope and exact zero-margin
    root.  A small deterministic step across the root is included so strict
    decision inequalities can be tested without arbitrary optimisation.
    """

    if concept_id not in program.concept_ids or not np.isfinite([lower, upper]).all():
        return []
    if upper - lower <= 1e-12:
        return []
    low_score = float(program.intervene(concept_values, {concept_id: lower})[0])
    high_score = float(program.intervene(concept_values, {concept_id: upper})[0])
    slope = (high_score - low_score) / (upper - lower)
    if not math.isfinite(slope) or abs(slope) <= 1e-12:
        return []
    root = lower - low_score / slope
    tolerance = max(1e-10, (upper - lower) * 1e-9)
    if root < lower - tolerance or root > upper + tolerance:
        return []
    root = min(max(root, lower), upper)
    baseline = float(program.score(concept_values)[0])
    crossing_direction = -math.copysign(1.0, slope) if baseline > 0 else math.copysign(1.0, slope)
    crossed = min(max(root + crossing_direction * tolerance, lower), upper)
    return [root, crossed]
