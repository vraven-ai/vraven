"""Cross-layer concept evolution using activation-profile transport.

The same samples are traced through every selected layer.  Concepts are aligned by their
standardised activation profiles, avoiding the invalid practice of treating independently
fitted two-dimensional embeddings as literal movement through depth.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .concepts import ConceptAtlas, ConceptSet


class EvolutionEvent(str, Enum):
    PERSIST = "persist"
    SPLIT = "split"
    MERGE = "merge"
    TRANSFORM = "transform"
    BIRTH = "birth"
    DEATH = "death"


@dataclass(slots=True)
class EvolutionEdge:
    source: str
    target: str
    source_node: str
    target_node: str
    similarity: float
    signed_similarity: float
    transport_mass: float
    event: EvolutionEvent
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvolutionState:
    concept_id: str
    node_id: str
    depth: float
    event: EvolutionEvent
    mass: float


@dataclass(slots=True)
class ConceptEvolution:
    edges: list[EvolutionEdge]
    states: list[EvolutionState]
    layer_pairs: list[tuple[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def outgoing(self, concept_id: str) -> list[EvolutionEdge]:
        return [edge for edge in self.edges if edge.source == concept_id]

    def incoming(self, concept_id: str) -> list[EvolutionEdge]:
        return [edge for edge in self.edges if edge.target == concept_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": [
                {**asdict(edge), "event": edge.event.value}
                for edge in self.edges
            ],
            "states": [
                {**asdict(state), "event": state.event.value}
                for state in self.states
            ],
            "layer_pairs": [list(pair) for pair in self.layer_pairs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConceptEvolution:
        return cls(
            edges=[
                EvolutionEdge(
                    source=item["source"],
                    target=item["target"],
                    source_node=item["source_node"],
                    target_node=item["target_node"],
                    similarity=float(item["similarity"]),
                    signed_similarity=float(item["signed_similarity"]),
                    transport_mass=float(item["transport_mass"]),
                    event=EvolutionEvent(item["event"]),
                    confidence=float(item["confidence"]),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("edges", ())
            ],
            states=[
                EvolutionState(
                    concept_id=item["concept_id"],
                    node_id=item["node_id"],
                    depth=float(item["depth"]),
                    event=EvolutionEvent(item["event"]),
                    mass=float(item["mass"]),
                )
                for item in payload.get("states", ())
            ],
            layer_pairs=[tuple(pair) for pair in payload.get("layer_pairs", ())],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class EvolutionConfig:
    entropy_regularisation: float = 0.08
    sinkhorn_iterations: int = 300
    transport_threshold: float = 0.04
    similarity_threshold: float = 0.25
    split_mass_threshold: float = 0.12
    merge_mass_threshold: float = 0.12
    epsilon: float = 1e-9


class ConceptEvolutionBuilder:
    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()

    def build(self, atlas: ConceptAtlas) -> ConceptEvolution:
        concept_sets = sorted(atlas.concept_sets, key=lambda item: item.depth)
        edges: list[EvolutionEdge] = []
        states: list[EvolutionState] = []
        layer_pairs: list[tuple[str, str]] = []
        for source_set, target_set in zip(concept_sets, concept_sets[1:], strict=False):
            layer_pairs.append((source_set.node_id, target_set.node_id))
            pair_edges, pair_states = self._align_pair(source_set, target_set)
            edges.extend(pair_edges)
            states.extend(pair_states)
        return ConceptEvolution(
            edges=edges,
            states=_deduplicate_states(states),
            layer_pairs=layer_pairs,
            metadata={
                "alignment": "entropic optimal transport over standardised activation profiles",
                "entropy_regularisation": self.config.entropy_regularisation,
                "transport_threshold": self.config.transport_threshold,
                "similarity_threshold": self.config.similarity_threshold,
                "warning": (
                    "Transport links concept activation profiles across the same samples. They are "
                    "candidate evolution relations, not causal edges until separately intervened upon."
                ),
            },
        )

    def _align_pair(
        self,
        source_set: ConceptSet,
        target_set: ConceptSet,
    ) -> tuple[list[EvolutionEdge], list[EvolutionState]]:
        signed = _profile_similarity(source_set.scores, target_set.scores)
        similarity = np.abs(signed)
        cost = 1.0 - similarity
        source_mass = _concept_mass(source_set)
        target_mass = _concept_mass(target_set)
        transport = sinkhorn_transport(
            source_mass,
            target_mass,
            cost,
            regularisation=self.config.entropy_regularisation,
            iterations=self.config.sinkhorn_iterations,
            epsilon=self.config.epsilon,
        )
        candidate_mask = (
            (transport >= self.config.transport_threshold)
            & (similarity >= self.config.similarity_threshold)
        )
        outgoing_counts = candidate_mask.sum(axis=1)
        incoming_counts = candidate_mask.sum(axis=0)
        edges: list[EvolutionEdge] = []
        for source_index, source_concept in enumerate(source_set.concepts):
            for target_index, target_concept in enumerate(target_set.concepts):
                mass = float(transport[source_index, target_index])
                sim = float(similarity[source_index, target_index])
                if not candidate_mask[source_index, target_index]:
                    continue
                event = EvolutionEvent.TRANSFORM
                if outgoing_counts[source_index] > 1 and mass >= self.config.split_mass_threshold:
                    event = EvolutionEvent.SPLIT
                if incoming_counts[target_index] > 1 and mass >= self.config.merge_mass_threshold:
                    event = EvolutionEvent.MERGE
                if outgoing_counts[source_index] == 1 and incoming_counts[target_index] == 1:
                    event = EvolutionEvent.PERSIST
                confidence = float(math.sqrt(max(mass, 0.0) * max(sim, 0.0)))
                edges.append(
                    EvolutionEdge(
                        source=source_concept.id,
                        target=target_concept.id,
                        source_node=source_set.node_id,
                        target_node=target_set.node_id,
                        similarity=sim,
                        signed_similarity=float(signed[source_index, target_index]),
                        transport_mass=mass,
                        event=event,
                        confidence=confidence,
                    )
                )
        linked_sources = {edge.source for edge in edges}
        linked_targets = {edge.target for edge in edges}
        states = []
        for index, concept in enumerate(source_set.concepts):
            states.append(
                EvolutionState(
                    concept_id=concept.id,
                    node_id=source_set.node_id,
                    depth=source_set.depth,
                    event=(EvolutionEvent.PERSIST if concept.id in linked_sources else EvolutionEvent.DEATH),
                    mass=float(source_mass[index]),
                )
            )
        for index, concept in enumerate(target_set.concepts):
            states.append(
                EvolutionState(
                    concept_id=concept.id,
                    node_id=target_set.node_id,
                    depth=target_set.depth,
                    event=(EvolutionEvent.PERSIST if concept.id in linked_targets else EvolutionEvent.BIRTH),
                    mass=float(target_mass[index]),
                )
            )
        return edges, states


def sinkhorn_transport(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    cost: np.ndarray,
    *,
    regularisation: float = 0.08,
    iterations: int = 300,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Compute a small entropic optimal-transport plan without an external OT library."""

    a = np.asarray(source_mass, dtype=float)
    b = np.asarray(target_mass, dtype=float)
    c = np.asarray(cost, dtype=float)
    if c.shape != (a.size, b.size):
        raise ValueError("cost shape must match source and target masses")
    a = np.clip(a, epsilon, None)
    b = np.clip(b, epsilon, None)
    a = a / a.sum()
    b = b / b.sum()
    reg = max(float(regularisation), epsilon)
    kernel = np.exp(-c / reg)
    kernel = np.clip(kernel, epsilon, None)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(iterations)):
        previous = u.copy()
        u = a / np.clip(kernel @ v, epsilon, None)
        v = b / np.clip(kernel.T @ u, epsilon, None)
        if np.max(np.abs(u - previous)) < 1e-10:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    plan = np.maximum(plan, 0.0)
    total = plan.sum()
    return plan / total if total > 0 else np.outer(a, b)


def _profile_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape[0] != target.shape[0]:
        raise ValueError("Concept sets must be evaluated on the same aligned samples")
    source_z = _standardise_columns(source)
    target_z = _standardise_columns(target)
    denominator = max(1, source.shape[0] - 1)
    return (source_z.T @ target_z) / denominator


def _standardise_columns(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    centred = value - value.mean(axis=0, keepdims=True)
    scale = value.std(axis=0, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return centred / scale


def _concept_mass(concept_set: ConceptSet) -> np.ndarray:
    variance = np.var(concept_set.scores, axis=0)
    variance = np.clip(variance, 1e-12, None)
    return variance / variance.sum()


def _deduplicate_states(states: list[EvolutionState]) -> list[EvolutionState]:
    priority = {
        EvolutionEvent.BIRTH: 0,
        EvolutionEvent.DEATH: 0,
        EvolutionEvent.PERSIST: 1,
        EvolutionEvent.TRANSFORM: 2,
        EvolutionEvent.SPLIT: 3,
        EvolutionEvent.MERGE: 3,
    }
    retained: dict[str, EvolutionState] = {}
    for state in states:
        previous = retained.get(state.concept_id)
        if previous is None or priority[state.event] > priority[previous.event]:
            retained[state.concept_id] = state
    return sorted(retained.values(), key=lambda item: (item.depth, item.concept_id))
