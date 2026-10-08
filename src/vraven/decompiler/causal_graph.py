"""Intervention-validated causal mechanism graph over discovered internal concepts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils import _pytree as pytree

from .capture import CapturedModel
from .concepts import Concept, ConceptAtlas
from .evolution import ConceptEvolution, EvolutionEdge
from .interventions import (
    GraphInterventionCompiler,
    InterventionKind,
    InterventionSpec,
    bootstrap_interval,
)
from .runtime import (
    FXValueRecorder,
    extract_logits,
    predicted_class,
    target_margin,
)


@dataclass(slots=True)
class CausalGraphConfig:
    max_source_concepts: int = 12
    max_edges: int = 36
    max_samples: int = 96
    minimum_evolution_confidence: float = 0.08
    minimum_absolute_edge_effect: float = 1e-4
    bootstrap_repeats: int = 300
    seed: int = 17
    intervention_strength: float = 1.0
    derivative_screening: bool = True
    derivative_minimum: float = 1e-7
    output_path: str | Sequence[str | int] | None = None


@dataclass(slots=True)
class CausalConceptNode:
    concept_id: str
    node_id: str
    label: str
    depth: float
    output_effect: float
    output_effect_ci: tuple[float, float]
    responsibility: float
    necessity: float
    sign_stability: float
    prediction_flip_rate: float
    manifold_distance: float
    evidence_tier: str
    semantic_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalConceptEdge:
    source: str
    target: str
    source_node: str
    target_node: str
    effect: float
    effect_ci: tuple[float, float]
    derivative_support: float | None
    derivative_sign_agreement: bool | None
    sign_stability: float
    evolution_similarity: float
    evolution_transport: float
    off_manifold_distance: float
    evidence_tier: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalMechanismGraph:
    nodes: list[CausalConceptNode]
    edges: list[CausalConceptEdge]
    target: int
    baseline_margin: np.ndarray
    baseline_prediction: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, CausalConceptNode]:
        return {node.concept_id: node for node in self.nodes}

    def outgoing(self, concept_id: str) -> list[CausalConceptEdge]:
        return [edge for edge in self.edges if edge.source == concept_id]

    def incoming(self, concept_id: str) -> list[CausalConceptEdge]:
        return [edge for edge in self.edges if edge.target == concept_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "target": self.target,
            "baseline_margin": self.baseline_margin.tolist(),
            "baseline_prediction": self.baseline_prediction.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalMechanismGraph:
        return cls(
            nodes=[
                CausalConceptNode(
                    **{
                        **item,
                        "output_effect_ci": tuple(item["output_effect_ci"]),
                    }
                )
                for item in payload.get("nodes", ())
            ],
            edges=[
                CausalConceptEdge(
                    **{
                        **item,
                        "effect_ci": tuple(item["effect_ci"]),
                    }
                )
                for item in payload.get("edges", ())
            ],
            target=int(payload["target"]),
            baseline_margin=np.asarray(payload.get("baseline_margin", ()), dtype=float),
            baseline_prediction=np.asarray(payload.get("baseline_prediction", ()), dtype=int),
            metadata=dict(payload.get("metadata", {})),
        )


class CausalMechanismBuilder:
    """Test candidate concept relations through graph-compiled interventions."""

    def __init__(self, config: CausalGraphConfig | None = None) -> None:
        self.config = config or CausalGraphConfig()

    def build(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        evolution: ConceptEvolution,
        reference_args: Sequence[Any],
        *,
        reference_kwargs: Mapping[str, Any] | None = None,
        target: int,
    ) -> CausalMechanismGraph:
        if captured.graph_module is None:
            raise ValueError("Causal mechanism recovery requires an export or FX graph capture")
        args, kwargs, sample_indices = _limit_reference(
            tuple(reference_args),
            dict(reference_kwargs or {}),
            self.config.max_samples,
        )
        concept_map = atlas.concept_map()
        candidate_edges = self._candidate_edges(evolution, concept_map)
        source_ids = self._select_sources(atlas, candidate_edges)
        target_ids = {edge.target for edge in candidate_edges}
        selected_nodes = sorted(
            {concept_map[concept_id].node_id for concept_id in source_ids | target_ids}
        )

        baseline_recorder = FXValueRecorder(
            captured.graph_module,
            selected_nodes=selected_nodes,
            detach=False,
            move_to_cpu=False,
        )
        baseline_record = baseline_recorder.execute(args, kwargs)
        baseline_logits = extract_logits(baseline_record.output, self.config.output_path)
        baseline_margin = target_margin(baseline_logits, target)
        baseline_prediction = predicted_class(baseline_logits)

        compiler = GraphInterventionCompiler(captured)
        node_results: dict[str, CausalConceptNode] = {}
        intervened_records: dict[str, dict[str, Any]] = {}
        output_effects: dict[str, np.ndarray] = {}

        for source_id in sorted(source_ids):
            concept = concept_map[source_id]
            spec = InterventionSpec(
                node_id=concept.node_id,
                kind=InterventionKind.ERASE_DIRECTION,
                label=f"Erase {concept.display_label}",
                feature_axis=concept.feature_axis,
                direction=concept.direction,
                strength=self.config.intervention_strength,
                metadata={"concept_id": concept.id},
            )
            compiled = compiler.compile([spec])
            recorder = FXValueRecorder(
                compiled.module,
                selected_nodes=selected_nodes,
                detach=True,
                move_to_cpu=True,
            )
            record = recorder.execute(args, kwargs)
            intervened_logits = extract_logits(record.output, self.config.output_path)
            intervened_margin = target_margin(intervened_logits, target)
            effect = (
                baseline_margin.detach().cpu().to(torch.float64)
                - intervened_margin.detach().cpu().to(torch.float64)
            ).numpy()
            output_effects[source_id] = effect
            intervened_records[source_id] = record.values
            low, high = bootstrap_interval(
                effect,
                repeats=self.config.bootstrap_repeats,
                seed=self.config.seed + len(node_results),
            )
            median = float(np.median(effect)) if effect.size else 0.0
            sign_stability = _sign_stability(effect, median)
            intervened_prediction = predicted_class(intervened_logits)
            flip_rate = float(
                (baseline_prediction.detach().cpu() != intervened_prediction.detach().cpu())
                .to(torch.float32)
                .mean()
                .item()
            )
            manifold_distance = _erasure_manifold_distance(atlas, concept, sample_indices)
            tier = _node_evidence_tier(low, high, sign_stability)
            node_results[source_id] = CausalConceptNode(
                concept_id=source_id,
                node_id=concept.node_id,
                label=concept.display_label,
                depth=concept.depth,
                output_effect=median,
                output_effect_ci=(low, high),
                responsibility=0.0,
                necessity=abs(median),
                sign_stability=sign_stability,
                prediction_flip_rate=flip_rate,
                manifold_distance=manifold_distance,
                evidence_tier=tier,
                semantic_confidence=concept.semantic_confidence,
                metadata={
                    "intervention": "orthogonal concept-direction erasure",
                    "sample_count": int(effect.size),
                    "target_margin_definition": "target logit minus logsumexp alternatives",
                },
            )

        responsibility_denominator = sum(
            abs(node.output_effect) for node in node_results.values()
        )
        for node in node_results.values():
            node.responsibility = (
                abs(node.output_effect) / responsibility_denominator
                if responsibility_denominator > 1e-12
                else 0.0
            )

        derivative_support = (
            self._derivative_support(
                captured,
                atlas,
                concept_map,
                candidate_edges,
                args,
                kwargs,
            )
            if self.config.derivative_screening
            else {}
        )

        causal_edges: list[CausalConceptEdge] = []
        for ordinal, candidate in enumerate(candidate_edges):
            if candidate.source not in intervened_records:
                continue
            source_concept = concept_map[candidate.source]
            target_concept = concept_map[candidate.target]
            baseline_value = _first_tensor(baseline_record.values.get(target_concept.node_id))
            intervened_value = _first_tensor(
                intervened_records[candidate.source].get(target_concept.node_id)
            )
            if baseline_value is None or intervened_value is None:
                continue
            concept_set = atlas.set_for_node(target_concept.node_id)
            baseline_score = concept_set.score_component(
                baseline_value.detach(),
                target_concept.index,
            ).cpu().numpy()
            intervened_score = concept_set.score_component(
                intervened_value,
                target_concept.index,
            ).cpu().numpy()
            edge_effect_samples = baseline_score - intervened_score
            median = float(np.median(edge_effect_samples))
            if abs(median) < self.config.minimum_absolute_edge_effect:
                continue
            low, high = bootstrap_interval(
                edge_effect_samples,
                repeats=self.config.bootstrap_repeats,
                seed=self.config.seed + 10_000 + ordinal,
            )
            derivative = derivative_support.get((candidate.source, candidate.target))
            agreement = (
                bool(np.sign(derivative) == np.sign(median))
                if derivative is not None and abs(derivative) > self.config.derivative_minimum
                else None
            )
            stability = _sign_stability(edge_effect_samples, median)
            tier = _edge_evidence_tier(low, high, stability, derivative, agreement)
            causal_edges.append(
                CausalConceptEdge(
                    source=candidate.source,
                    target=candidate.target,
                    source_node=source_concept.node_id,
                    target_node=target_concept.node_id,
                    effect=median,
                    effect_ci=(low, high),
                    derivative_support=derivative,
                    derivative_sign_agreement=agreement,
                    sign_stability=stability,
                    evolution_similarity=candidate.similarity,
                    evolution_transport=candidate.transport_mass,
                    off_manifold_distance=node_results[candidate.source].manifold_distance,
                    evidence_tier=tier,
                    metadata={
                        "effect_definition": (
                            "baseline downstream concept coefficient minus coefficient after "
                            "source concept erasure"
                        ),
                        "association_preselection": True,
                    },
                )
            )
            if len(causal_edges) >= self.config.max_edges:
                break

        # Include target-only concepts as observational nodes so the graph remains readable,
        # but do not assign them causal responsibility without an output intervention test.
        for target_id in sorted(target_ids):
            if target_id in node_results:
                continue
            concept = concept_map[target_id]
            node_results[target_id] = CausalConceptNode(
                concept_id=target_id,
                node_id=concept.node_id,
                label=concept.display_label,
                depth=concept.depth,
                output_effect=0.0,
                output_effect_ci=(0.0, 0.0),
                responsibility=0.0,
                necessity=0.0,
                sign_stability=0.0,
                prediction_flip_rate=0.0,
                manifold_distance=0.0,
                evidence_tier="observed_only",
                semantic_confidence=concept.semantic_confidence,
            )

        return CausalMechanismGraph(
            nodes=sorted(node_results.values(), key=lambda item: (item.depth, item.concept_id)),
            edges=sorted(causal_edges, key=lambda item: abs(item.effect), reverse=True),
            target=int(target),
            baseline_margin=baseline_margin.detach().cpu().numpy(),
            baseline_prediction=baseline_prediction.detach().cpu().numpy(),
            metadata={
                "samples": len(sample_indices),
                "sample_indices": sample_indices.tolist(),
                "intervention_strength": self.config.intervention_strength,
                "bootstrap_repeats": self.config.bootstrap_repeats,
                "candidate_edges": len(candidate_edges),
                "validated_edges": len(causal_edges),
                "warning": (
                    "Edges are local to the tested cohort, intervention type and activation "
                    "baseline. Off-manifold distance is reported because internal erasure may "
                    "produce states the original model rarely encounters."
                ),
            },
        )

    def _candidate_edges(
        self,
        evolution: ConceptEvolution,
        concept_map: Mapping[str, Concept],
    ) -> list[EvolutionEdge]:
        retained = [
            edge
            for edge in evolution.edges
            if edge.source in concept_map
            and edge.target in concept_map
            and edge.confidence >= self.config.minimum_evolution_confidence
        ]
        retained.sort(
            key=lambda edge: edge.confidence * (0.5 + edge.transport_mass),
            reverse=True,
        )
        return retained[: self.config.max_edges * 2]

    def _select_sources(
        self,
        atlas: ConceptAtlas,
        candidate_edges: Sequence[EvolutionEdge],
    ) -> set[str]:
        candidates = {edge.source for edge in candidate_edges}
        concept_map = atlas.concept_map()
        ranked = sorted(
            candidates,
            key=lambda concept_id: (
                abs(concept_map[concept_id].class_effect_size or 0.0),
                concept_map[concept_id].score_variance,
            ),
            reverse=True,
        )
        return set(ranked[: self.config.max_source_concepts])

    def _derivative_support(
        self,
        captured: CapturedModel,
        atlas: ConceptAtlas,
        concept_map: Mapping[str, Concept],
        candidate_edges: Sequence[EvolutionEdge],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[tuple[str, str], float]:
        if captured.graph_module is None:
            return {}
        node_ids = sorted(
            {
                concept_map[edge.source].node_id
                for edge in candidate_edges
                if edge.source in concept_map and edge.target in concept_map
            }
            | {
                concept_map[edge.target].node_id
                for edge in candidate_edges
                if edge.source in concept_map and edge.target in concept_map
            }
        )
        recorder = FXValueRecorder(
            captured.graph_module,
            selected_nodes=node_ids,
            detach=False,
            move_to_cpu=False,
        )
        try:
            record = recorder.execute(args, kwargs)
        except Exception:
            return {}
        output: dict[tuple[str, str], float] = {}
        for edge in candidate_edges:
            source = concept_map.get(edge.source)
            target = concept_map.get(edge.target)
            if source is None or target is None:
                continue
            source_value = _first_tensor(record.values.get(source.node_id))
            target_value = _first_tensor(record.values.get(target.node_id))
            if source_value is None or target_value is None or not source_value.requires_grad:
                continue
            try:
                target_score = atlas.set_for_node(target.node_id).score_component(
                    target_value,
                    target.index,
                ).mean()
                gradient = torch.autograd.grad(
                    target_score,
                    source_value,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if gradient is None:
                    continue
                derivative = _directional_gradient(
                    gradient,
                    source.direction,
                    source.feature_axis,
                )
                output[(edge.source, edge.target)] = derivative
            except (RuntimeError, ValueError):
                continue
        return output


def _directional_gradient(gradient: Tensor, direction: Tensor, feature_axis: int) -> float:
    axis = feature_axis if feature_axis >= 0 else gradient.ndim + feature_axis
    moved = gradient.movedim(axis, -1)
    direction = direction.to(device=moved.device, dtype=moved.dtype).reshape(-1)
    direction = direction / direction.norm().clamp_min(1e-12)
    projected = torch.einsum("...f,f->...", moved, direction)
    return float(projected.mean().detach().cpu().item())


def _erasure_manifold_distance(
    atlas: ConceptAtlas,
    concept: Concept,
    sample_indices: np.ndarray,
) -> float:
    scores = atlas.set_for_node(concept.node_id).score_for(concept.id)
    scores = np.asarray(scores)[sample_indices]
    standard_deviation = float(np.std(scores))
    if standard_deviation < 1e-12:
        return 0.0
    erased = scores * 0.0
    mean = float(np.mean(scores))
    return float(np.mean(np.abs((erased - mean) / standard_deviation)))


def _node_evidence_tier(low: float, high: float, stability: float) -> str:
    excludes_zero = low > 0 or high < 0
    if excludes_zero and stability >= 0.8:
        return "intervention_supported"
    if excludes_zero:
        return "intervention_detected_unstable"
    return "observed_intervention_inconclusive"


def _edge_evidence_tier(
    low: float,
    high: float,
    stability: float,
    derivative: float | None,
    agreement: bool | None,
) -> str:
    excludes_zero = low > 0 or high < 0
    if excludes_zero and stability >= 0.8 and agreement is True:
        return "bidirectional_intervention_supported"
    if excludes_zero and stability >= 0.8:
        return "intervention_supported"
    if excludes_zero:
        return "intervention_detected_unstable"
    if derivative is not None:
        return "derivative_supported_only"
    return "observational_candidate"


def _sign_stability(values: np.ndarray, centre: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not values.size:
        return 0.0
    if abs(centre) < 1e-12:
        return float(np.mean(np.abs(values) < 1e-12))
    return float(np.mean(np.sign(values) == np.sign(centre)))


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    flat, _ = pytree.tree_flatten(value)
    return next((item for item in flat if isinstance(item, Tensor)), None)


def _limit_reference(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    maximum: int,
) -> tuple[tuple[Any, ...], dict[str, Any], np.ndarray]:
    flat, _ = pytree.tree_flatten((args, kwargs))
    lengths = [int(item.shape[0]) for item in flat if isinstance(item, Tensor) and item.ndim > 0]
    if not lengths:
        raise ValueError("Reference arguments must contain at least one batched tensor")
    if len(set(lengths)) != 1:
        raise ValueError("All batched reference tensors must have the same first dimension")
    total = lengths[0]
    retained = min(total, int(maximum))
    indices = np.linspace(0, total - 1, retained, dtype=int) if retained < total else np.arange(total)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)

    def select(value: Any) -> Any:
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == total:
            return value.index_select(0, index_tensor.to(value.device))
        return value

    return (
        tuple(pytree.tree_map(select, args)),
        dict(pytree.tree_map(select, kwargs)),
        indices,
    )
