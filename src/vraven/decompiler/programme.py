"""Compile discovered mechanisms into a compact executable causal programme."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .causal_graph import CausalMechanismGraph
from .concepts import ConceptAtlas


@dataclass(slots=True)
class ProgrammeConfig:
    maximum_concepts: int = 16
    maximum_rules: int = 12
    include_interactions: bool = True
    maximum_interactions: int = 12
    l1_strength: float = 0.015
    fista_iterations: int = 1200
    fista_tolerance: float = 1e-8
    validation_fraction: float = 0.25
    seed: int = 17
    coefficient_floor: float = 1e-6
    interventional_weight: float = 16.0
    minimum_intervention_effect: float = 1e-6


@dataclass(slots=True)
class ProgrammeRule:
    id: str
    kind: str
    concept_ids: tuple[str, ...]
    labels: tuple[str, ...]
    coefficient: float
    term_mean: float
    term_scale: float
    evidence_tier: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def raw_term(self, concept_values: np.ndarray, concept_index: Mapping[str, int]) -> np.ndarray:
        if self.kind == "main":
            return concept_values[:, concept_index[self.concept_ids[0]]]
        if self.kind == "interaction":
            left = concept_values[:, concept_index[self.concept_ids[0]]]
            right = concept_values[:, concept_index[self.concept_ids[1]]]
            return left * right
        raise ValueError(f"Unsupported programme rule kind: {self.kind}")

    def standardised_term(self, concept_values: np.ndarray, concept_index: Mapping[str, int]) -> np.ndarray:
        return (self.raw_term(concept_values, concept_index) - self.term_mean) / max(
            self.term_scale, 1e-12
        )


@dataclass(slots=True)
class ProgrammeFidelity:
    train_mae: float
    validation_mae: float
    train_r2: float
    validation_r2: float
    train_correlation: float
    validation_correlation: float
    train_decision_agreement: float
    validation_decision_agreement: float
    interventional_correlation: float | None = None
    interventional_mae: float | None = None
    interventional_sign_agreement: float | None = None
    rules: int = 0
    concepts: int = 0


@dataclass(slots=True)
class MechanismProgram:
    target: int
    concept_ids: tuple[str, ...]
    concept_labels: dict[str, str]
    intercept: float
    rules: list[ProgrammeRule]
    fidelity: ProgrammeFidelity
    validity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, concept_values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(concept_values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(self.concept_ids):
            raise ValueError(
                f"Expected {len(self.concept_ids)} concept columns, received {matrix.shape[1]}"
            )
        index = {concept_id: position for position, concept_id in enumerate(self.concept_ids)}
        output = np.full(matrix.shape[0], self.intercept, dtype=float)
        for rule in self.rules:
            output += rule.coefficient * rule.standardised_term(matrix, index)
        return output

    def contributions(self, concept_values: np.ndarray | Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
        matrix = np.asarray(concept_values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        index = {concept_id: position for position, concept_id in enumerate(self.concept_ids)}
        return {
            rule.id: rule.coefficient * rule.standardised_term(matrix, index)
            for rule in self.rules
        }

    def predict_target(self, concept_values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        return self.score(concept_values) > 0.0

    def intervene(
        self,
        concept_values: np.ndarray | Sequence[Sequence[float]],
        replacements: Mapping[str, float | np.ndarray],
    ) -> np.ndarray:
        matrix = np.asarray(concept_values, dtype=float).copy()
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        index = {concept_id: position for position, concept_id in enumerate(self.concept_ids)}
        for concept_id, replacement in replacements.items():
            if concept_id not in index:
                raise KeyError(concept_id)
            matrix[:, index[concept_id]] = replacement
        return self.score(matrix)

    def to_pseudocode(self) -> str:
        lines = [
            f"PROGRAMME TARGET CLASS {self.target}",
            f"margin = {self.intercept:+.6f}",
        ]
        for ordinal, rule in enumerate(self.rules, start=1):
            if rule.kind == "main":
                expression = rule.labels[0]
            else:
                expression = f"({rule.labels[0]}) × ({rule.labels[1]})"
            lines.append(
                f"RULE {ordinal:02d}: margin += {rule.coefficient:+.6f} × "
                f"standardise[{expression}; mean={rule.term_mean:.6f}; scale={rule.term_scale:.6f}]"
            )
        lines.extend(
            [
                "DECISION: choose target when margin > 0; otherwise choose the strongest alternative.",
                (
                    "VALIDATION: observational and interventional fidelity are recorded separately; "
                    "this programme is not asserted to be the unique true mechanism."
                ),
            ]
        )
        return "\n".join(lines)

    def to_python_code(self) -> str:
        """Export the decompiled causal mechanism program as a pure standalone executable Python function."""
        code_lines = [
            "#!/usr/bin/env python3",
            '"""Executable symbolic causal program decompiled by VRAVEN 0.9.4 SE."""',
            "from typing import Any, Mapping",
            "",
            f"TARGET_CLASS = {self.target}",
            f"CONCEPT_IDS = {list(self.concept_ids)!r}",
            "",
            "def predict_mechanism(concept_values: Mapping[str, float]) -> dict[str, Any]:",
            "    margin = " + f"{self.intercept:+.8f}",
            "    contributions = {}",
        ]
        for ordinal, rule in enumerate(self.rules, start=1):
            var_name = f"rule_{ordinal:02d}"
            if rule.kind == "main":
                concept_id = rule.concept_ids[0]
                raw_expr = f"concept_values.get({concept_id!r}, 0.0)"
            else:
                c1, c2 = rule.concept_ids[0], rule.concept_ids[1]
                raw_expr = f"concept_values.get({c1!r}, 0.0) * concept_values.get({c2!r}, 0.0)"
            
            code_lines.append(f"    # {rule.id}: {rule.labels}")
            code_lines.append(f"    raw_{var_name} = {raw_expr}")
            code_lines.append(
                f"    norm_{var_name} = (raw_{var_name} - {rule.term_mean:.8f}) / {max(rule.term_scale, 1e-12):.8f}"
            )
            code_lines.append(f"    contrib_{var_name} = {rule.coefficient:+.8f} * norm_{var_name}")
            code_lines.append(f"    contributions[{rule.id!r}] = contrib_{var_name}")
            code_lines.append(f"    margin += contrib_{var_name}")
            code_lines.append("")
        
        code_lines.extend([
            "    decision = TARGET_CLASS if margin > 0.0 else 0",
            "    return {",
            "        'target_class': TARGET_CLASS,",
            "        'decision': decision,",
            "        'decision_margin': margin,",
            "        'decision_probability': 1.0 / (1.0 + pow(2.718281828459045, -margin)),",
            "        'rule_contributions': contributions,",
            "    }",
            "",
            'if __name__ == "__main__":',
            "    # Sample evaluation with baseline zero concepts",
            "    sample_inputs = {c: 0.0 for c in CONCEPT_IDS}",
            "    result = predict_mechanism(sample_inputs)",
            "    print('VRAVEN Decompiled Programme Execution Result:')",
            "    print(f'  Target Class: {result[\"target_class\"]}')",
            "    print(f'  Decision:     {result[\"decision\"]}')",
            "    print(f'  Margin:       {result[\"decision_margin\"]:+.6f}')",
            "    print(f'  Probability:  {result[\"decision_probability\"]:.4%}')",
        ])
        return "\n".join(code_lines)

    def to_c_code(self) -> str:
        """Export the decompiled causal mechanism program into pure standalone ANSI C source code."""
        c_lines = [
            "/* Executable symbolic causal program decompiled by VRAVEN 0.9.4 SE. */",
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define TARGET_CLASS {self.target}",
            f"#define NUM_CONCEPTS {len(self.concept_ids)}",
            "",
            "typedef struct {",
            "    int target_class;",
            "    int decision;",
            "    double decision_margin;",
            "    double decision_probability;",
            "} VRAVEN_Prediction;",
            "",
            "VRAVEN_Prediction predict_mechanism(const double concept_values[NUM_CONCEPTS]) {",
            "    VRAVEN_Prediction res;",
            f"    double margin = {self.intercept:+.8f};",
        ]
        
        concept_idx_map = {cid: idx for idx, cid in enumerate(self.concept_ids)}
        
        for ordinal, rule in enumerate(self.rules, start=1):
            if rule.kind == "main":
                idx0 = concept_idx_map.get(rule.concept_ids[0], 0)
                raw_expr = f"concept_values[{idx0}]"
            else:
                idx0 = concept_idx_map.get(rule.concept_ids[0], 0)
                idx1 = concept_idx_map.get(rule.concept_ids[1], 0)
                raw_expr = f"(concept_values[{idx0}] * concept_values[{idx1}])"
            
            c_lines.append(f"    /* Rule {ordinal:02d}: {rule.id} */")
            c_lines.append(f"    double norm_r{ordinal:02d} = ({raw_expr} - {rule.term_mean:.8f}) / {max(rule.term_scale, 1e-12):.8f};")
            c_lines.append(f"    margin += {rule.coefficient:+.8f} * norm_r{ordinal:02d};")
        
        c_lines.extend([
            "    res.target_class = TARGET_CLASS;",
            "    res.decision = (margin > 0.0) ? TARGET_CLASS : 0;",
            "    res.decision_margin = margin;",
            "    res.decision_probability = 1.0 / (1.0 + exp(-margin));",
            "    return res;",
            "}",
            "",
            "int main(void) {",
            "    double sample_inputs[NUM_CONCEPTS] = {0.0};",
            "    VRAVEN_Prediction p = predict_mechanism(sample_inputs);",
            '    printf("VRAVEN C-Decompiled Program Execution Result:\\n");',
            '    printf("  Target Class: %d\\n", p.target_class);',
            '    printf("  Decision:     %d\\n", p.decision);',
            '    printf("  Margin:       %+.6f\\n", p.decision_margin);',
            '    printf("  Probability:  %.4f%%\\n", p.decision_probability * 100.0);',
            "    return 0;",
            "}",
        ])
        return "\n".join(c_lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "concept_ids": list(self.concept_ids),
            "concept_labels": self.concept_labels,
            "intercept": self.intercept,
            "rules": [asdict(rule) for rule in self.rules],
            "fidelity": asdict(self.fidelity),
            "validity": self.validity,
            "metadata": self.metadata,
            "pseudocode": self.to_pseudocode(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanismProgram:
        return cls(
            target=int(payload["target"]),
            concept_ids=tuple(payload["concept_ids"]),
            concept_labels=dict(payload.get("concept_labels", {})),
            intercept=float(payload["intercept"]),
            rules=[
                ProgrammeRule(
                    **{
                        **item,
                        "concept_ids": tuple(item["concept_ids"]),
                        "labels": tuple(item["labels"]),
                    }
                )
                for item in payload.get("rules", ())
            ],
            fidelity=ProgrammeFidelity(**payload["fidelity"]),
            validity=dict(payload.get("validity", {})),
            metadata=dict(payload.get("metadata", {})),
        )


class ProgrammeCompiler:
    def __init__(self, config: ProgrammeConfig | None = None) -> None:
        self.config = config or ProgrammeConfig()

    def compile(
        self,
        atlas: ConceptAtlas,
        causal_graph: CausalMechanismGraph,
        target_margin: Sequence[float] | np.ndarray,
        *,
        target: int,
    ) -> MechanismProgram:
        y = np.asarray(target_margin, dtype=float).reshape(-1)
        if y.size != len(atlas.sample_ids):
            raise ValueError("target_margin length must match the ConceptAtlas samples")
        selected = self._select_concepts(atlas, causal_graph)
        concept_matrix, selected = atlas.score_matrix(selected)
        if not selected:
            raise ValueError("No concepts are available for programme extraction")
        concept_map = atlas.concept_map()
        term_matrix, term_specs = self._build_terms(
            concept_matrix,
            selected,
            causal_graph,
        )
        train_indices, validation_indices = _train_validation_split(
            len(y), self.config.validation_fraction, self.config.seed
        )
        train_x = term_matrix[train_indices]
        train_y = y[train_indices]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_z = (train_x - mean) / scale
        y_mean = float(train_y.mean())
        intervention_x, intervention_y, intervention_ids = _intervention_constraints(
            concept_matrix,
            selected,
            term_specs,
            mean,
            scale,
            causal_graph,
            minimum_effect=self.config.minimum_intervention_effect,
        )
        if intervention_x.size:
            root_weight = math.sqrt(max(self.config.interventional_weight, 0.0))
            selection_x = np.vstack([train_z, intervention_x * root_weight])
            selection_y = np.concatenate([
                train_y - y_mean,
                intervention_y * root_weight,
            ])
        else:
            selection_x = train_z
            selection_y = train_y - y_mean
        weights = _fista_lasso(
            selection_x,
            selection_y,
            l1=self.config.l1_strength,
            iterations=self.config.fista_iterations,
            tolerance=self.config.fista_tolerance,
        )
        nonzero = np.flatnonzero(np.abs(weights) >= self.config.coefficient_floor)
        if not nonzero.size:
            nonzero = np.array([int(np.argmax(np.abs(weights)))])
        order = nonzero[np.argsort(np.abs(weights[nonzero]))[::-1]]
        retained = order[: self.config.maximum_rules]
        # Refit selected terms without L1 bias while retaining the sparse support.
        # Intervention constraints use a zero intercept because they represent score
        # differences, not absolute margins.
        observational_design = np.column_stack(
            [np.ones(train_z.shape[0]), train_z[:, retained]]
        )
        observational_target = train_y
        if intervention_x.size:
            intervention_design = np.column_stack(
                [np.zeros(intervention_x.shape[0]), intervention_x[:, retained]]
            )
            root_weight = math.sqrt(max(self.config.interventional_weight, 0.0))
            design = np.vstack(
                [observational_design, intervention_design * root_weight]
            )
            regression_target = np.concatenate(
                [observational_target, intervention_y * root_weight]
            )
        else:
            design = observational_design
            regression_target = observational_target
        coefficients, *_ = np.linalg.lstsq(design, regression_target, rcond=None)
        intercept = float(coefficients[0])
        selected_weights = coefficients[1:]
        rules: list[ProgrammeRule] = []
        for rule_ordinal, (term_index, coefficient) in enumerate(
            zip(retained, selected_weights, strict=True), start=1
        ):
            kind, concept_ids = term_specs[int(term_index)]
            labels = tuple(concept_map[item].display_label for item in concept_ids)
            evidence_tier = _term_evidence_tier(concept_ids, causal_graph)
            rules.append(
                ProgrammeRule(
                    id=f"R{rule_ordinal:02d}",
                    kind=kind,
                    concept_ids=concept_ids,
                    labels=labels,
                    coefficient=float(coefficient),
                    term_mean=float(mean[term_index]),
                    term_scale=float(scale[term_index]),
                    evidence_tier=evidence_tier,
                )
            )
        program = MechanismProgram(
            target=int(target),
            concept_ids=tuple(selected),
            concept_labels={item: concept_map[item].display_label for item in selected},
            intercept=intercept,
            rules=rules,
            fidelity=ProgrammeFidelity(
                train_mae=0.0,
                validation_mae=0.0,
                train_r2=0.0,
                validation_r2=0.0,
                train_correlation=0.0,
                validation_correlation=0.0,
                train_decision_agreement=0.0,
                validation_decision_agreement=0.0,
                rules=len(rules),
                concepts=len(selected),
            ),
            validity={
                "reference_samples": int(len(y)),
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(validation_indices)),
                "target_margin_range": [float(np.min(y)), float(np.max(y))],
            },
            metadata={
                "compiler": "sparse continuous mechanism programme",
                "l1_strength": self.config.l1_strength,
                "interaction_terms": self.config.include_interactions,
                "interventional_weight": self.config.interventional_weight,
                "intervention_constraints": intervention_ids,
                "warning": (
                    "The programme is a compact high-level causal surrogate. It is accepted only "
                    "within the separately reported observational, intervention and counterfactual "
                    "fidelity envelope."
                ),
            },
        )
        program.fidelity = _fidelity(
            program,
            concept_matrix,
            y,
            train_indices,
            validation_indices,
        )
        interventional = _interventional_fidelity(program, concept_matrix, causal_graph)
        program.fidelity.interventional_correlation = interventional[0]
        program.fidelity.interventional_mae = interventional[1]
        program.fidelity.interventional_sign_agreement = interventional[2]
        return program

    def _select_concepts(
        self,
        atlas: ConceptAtlas,
        causal_graph: CausalMechanismGraph,
    ) -> list[str]:
        graph_nodes = causal_graph.node_map()

        def rank_key(concept):
            node = graph_nodes.get(concept.id)
            intervention_supported = bool(
                node is not None
                and node.evidence_tier != "observed_only"
                and abs(node.output_effect) > 0.0
            )
            return (
                int(intervention_supported),
                abs(node.output_effect) if node is not None else 0.0,
                abs(concept.class_effect_size or 0.0),
                concept.score_variance,
            )

        ranked = sorted(atlas.concepts, key=rank_key, reverse=True)
        supported = [
            concept.id
            for concept in ranked
            if (
                concept.id in graph_nodes
                and graph_nodes[concept.id].evidence_tier != "observed_only"
                and abs(graph_nodes[concept.id].output_effect) > 0.0
            )
        ]
        # The executable mechanism should be built from intervention-tested concepts
        # whenever they are available. Purely observational readout concepts can make
        # an ordinary surrogate look perfect while failing every causal intervention.
        if supported:
            return supported[: self.config.maximum_concepts]
        return [concept.id for concept in ranked[: self.config.maximum_concepts]]

    def _build_terms(
        self,
        concept_matrix: np.ndarray,
        concept_ids: Sequence[str],
        causal_graph: CausalMechanismGraph,
    ) -> tuple[np.ndarray, list[tuple[str, tuple[str, ...]]]]:
        columns = [concept_matrix[:, index] for index in range(concept_matrix.shape[1])]
        specs: list[tuple[str, tuple[str, ...]]] = [
            ("main", (concept_id,)) for concept_id in concept_ids
        ]
        if self.config.include_interactions:
            index = {concept_id: position for position, concept_id in enumerate(concept_ids)}
            interactions = [
                edge
                for edge in causal_graph.edges
                if edge.source in index and edge.target in index
            ]
            interactions.sort(key=lambda edge: abs(edge.effect), reverse=True)
            for edge in interactions[: self.config.maximum_interactions]:
                columns.append(
                    concept_matrix[:, index[edge.source]]
                    * concept_matrix[:, index[edge.target]]
                )
                specs.append(("interaction", (edge.source, edge.target)))
        return np.column_stack(columns), specs


def _term_matrix_from_specs(
    concept_matrix: np.ndarray,
    concept_ids: Sequence[str],
    term_specs: Sequence[tuple[str, tuple[str, ...]]],
) -> np.ndarray:
    index = {concept_id: position for position, concept_id in enumerate(concept_ids)}
    columns: list[np.ndarray] = []
    for kind, members in term_specs:
        if kind == "main":
            columns.append(concept_matrix[:, index[members[0]]])
        elif kind == "interaction":
            columns.append(
                concept_matrix[:, index[members[0]]]
                * concept_matrix[:, index[members[1]]]
            )
        else:
            raise ValueError(f"Unsupported programme term kind: {kind}")
    return np.column_stack(columns) if columns else np.empty((concept_matrix.shape[0], 0))


def _intervention_constraints(
    concept_matrix: np.ndarray,
    concept_ids: Sequence[str],
    term_specs: Sequence[tuple[str, tuple[str, ...]]],
    term_mean: np.ndarray,
    term_scale: np.ndarray,
    causal_graph: CausalMechanismGraph,
    *,
    minimum_effect: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    node_map = causal_graph.node_map()
    index = {concept_id: position for position, concept_id in enumerate(concept_ids)}
    baseline_terms = _term_matrix_from_specs(concept_matrix, concept_ids, term_specs)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    identifiers: list[str] = []
    for concept_id, position in index.items():
        node = node_map.get(concept_id)
        if (
            node is None
            or node.evidence_tier == "observed_only"
            or abs(node.output_effect) < minimum_effect
        ):
            continue
        changed = concept_matrix.copy()
        changed[:, position] = 0.0
        changed_terms = _term_matrix_from_specs(changed, concept_ids, term_specs)
        # The intercept cancels. Standardisation is affine, so the difference in
        # standardised terms is simply the raw difference divided by scale.
        delta = np.median((baseline_terms - changed_terms) / term_scale, axis=0)
        rows.append(delta)
        targets.append(float(node.output_effect))
        identifiers.append(concept_id)
    if not rows:
        return (
            np.empty((0, len(term_specs)), dtype=float),
            np.empty((0,), dtype=float),
            [],
        )
    return np.vstack(rows), np.asarray(targets, dtype=float), identifiers


def _fista_lasso(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l1: float,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    if x.size == 0:
        return np.zeros(x.shape[1], dtype=float)
    spectral = np.linalg.norm(x, ord=2)
    lipschitz = max((spectral * spectral) / max(1, x.shape[0]), 1e-8)
    step = 1.0 / lipschitz
    weights = np.zeros(x.shape[1], dtype=float)
    momentum = weights.copy()
    t_value = 1.0
    for _ in range(iterations):
        gradient = (x.T @ (x @ momentum - y)) / max(1, x.shape[0])
        proposal = _soft_threshold(momentum - step * gradient, step * l1)
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_value * t_value)) / 2.0
        momentum = proposal + ((t_value - 1.0) / t_next) * (proposal - weights)
        if np.max(np.abs(proposal - weights)) < tolerance:
            weights = proposal
            break
        weights = proposal
        t_value = t_next
    return weights


def _soft_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def _train_validation_split(
    samples: int,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples < 4:
        indices = np.arange(samples)
        return indices, indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(samples)
    validation_size = max(1, min(samples - 2, int(round(samples * fraction))))
    return np.sort(indices[validation_size:]), np.sort(indices[:validation_size])


def _fidelity(
    program: MechanismProgram,
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> ProgrammeFidelity:
    prediction = program.score(x)

    def metrics(indices: np.ndarray) -> tuple[float, float, float, float]:
        observed = y[indices]
        predicted = prediction[indices]
        mae = float(np.mean(np.abs(observed - predicted)))
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        r2 = 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator if denominator > 1e-12 else 0.0
        correlation = _correlation(observed, predicted)
        agreement = float(np.mean((observed > 0) == (predicted > 0)))
        return mae, r2, correlation, agreement

    train_metrics = metrics(train)
    validation_metrics = metrics(validation)
    return ProgrammeFidelity(
        train_mae=train_metrics[0],
        validation_mae=validation_metrics[0],
        train_r2=train_metrics[1],
        validation_r2=validation_metrics[1],
        train_correlation=train_metrics[2],
        validation_correlation=validation_metrics[2],
        train_decision_agreement=train_metrics[3],
        validation_decision_agreement=validation_metrics[3],
        rules=len(program.rules),
        concepts=len(program.concept_ids),
    )


def _interventional_fidelity(
    program: MechanismProgram,
    concept_matrix: np.ndarray,
    causal_graph: CausalMechanismGraph,
) -> tuple[float | None, float | None, float | None]:
    graph_nodes = causal_graph.node_map()
    index = {concept_id: position for position, concept_id in enumerate(program.concept_ids)}
    baseline_score = program.score(concept_matrix)
    model_effects: list[float] = []
    programme_effects: list[float] = []
    for concept_id, position in index.items():
        node = graph_nodes.get(concept_id)
        if node is None or node.evidence_tier == "observed_only":
            continue
        changed = concept_matrix.copy()
        changed[:, position] = 0.0
        programme_effect = float(np.median(baseline_score - program.score(changed)))
        model_effects.append(float(node.output_effect))
        programme_effects.append(programme_effect)
    if not model_effects:
        return None, None, None
    model_array = np.asarray(model_effects)
    programme_array = np.asarray(programme_effects)
    return (
        _correlation(model_array, programme_array),
        float(np.mean(np.abs(model_array - programme_array))),
        float(np.mean(np.sign(model_array) == np.sign(programme_array))),
    )


def _term_evidence_tier(
    concept_ids: tuple[str, ...],
    causal_graph: CausalMechanismGraph,
) -> str:
    node_map = causal_graph.node_map()
    if len(concept_ids) == 1:
        node = node_map.get(concept_ids[0])
        return node.evidence_tier if node is not None else "observed_only"
    for edge in causal_graph.edges:
        if edge.source == concept_ids[0] and edge.target == concept_ids[1]:
            return edge.evidence_tier
    return "observed_interaction"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.size < 2 or right.size != left.size:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
