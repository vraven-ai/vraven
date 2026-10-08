"""Deterministic contrastive explanations generated from an executable programme."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .causal_graph import CausalMechanismGraph
from .programme import MechanismProgram, ProgrammeRule


@dataclass(slots=True)
class ContrastiveEvidence:
    rule_id: str
    concept_ids: tuple[str, ...]
    label: str
    contribution: float
    direction: str
    evidence_tier: str
    depth: float


@dataclass(slots=True)
class ContrastiveExplanation:
    target: int
    alternative: int | None
    target_label: str
    alternative_label: str
    programme_margin: float
    supporting: list[ContrastiveEvidence]
    opposing: list[ContrastiveEvidence]
    decisive_depth: float | None
    cumulative_by_depth: list[tuple[float, float]]
    plain_english: str
    limitations: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "alternative": self.alternative,
            "target_label": self.target_label,
            "alternative_label": self.alternative_label,
            "programme_margin": self.programme_margin,
            "supporting": [asdict(item) for item in self.supporting],
            "opposing": [asdict(item) for item in self.opposing],
            "decisive_depth": self.decisive_depth,
            "cumulative_by_depth": [list(item) for item in self.cumulative_by_depth],
            "plain_english": self.plain_english,
            "limitations": list(self.limitations),
            "metadata": self.metadata,
        }


class ContrastiveExplainer:
    def explain(
        self,
        program: MechanismProgram,
        concept_values: Sequence[float] | np.ndarray,
        causal_graph: CausalMechanismGraph,
        *,
        alternative: int | None = None,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
        maximum_items: int = 5,
    ) -> ContrastiveExplanation:
        values = np.asarray(concept_values, dtype=float).reshape(1, -1)
        if values.shape[1] != len(program.concept_ids):
            raise ValueError("concept_values must follow program.concept_ids order")
        contributions = program.contributions(values)
        node_map = causal_graph.node_map()
        evidence: list[ContrastiveEvidence] = []
        rule_map = {rule.id: rule for rule in program.rules}
        for rule_id, array in contributions.items():
            rule = rule_map[rule_id]
            contribution = float(array[0])
            depths = [node_map[item].depth for item in rule.concept_ids if item in node_map]
            depth = max(depths) if depths else 0.0
            label = _rule_label(rule)
            evidence.append(
                ContrastiveEvidence(
                    rule_id=rule_id,
                    concept_ids=rule.concept_ids,
                    label=label,
                    contribution=contribution,
                    direction="supports_target" if contribution >= 0 else "supports_alternative",
                    evidence_tier=rule.evidence_tier,
                    depth=depth,
                )
            )
        supporting = sorted(
            (item for item in evidence if item.contribution > 0),
            key=lambda item: item.contribution,
            reverse=True,
        )[:maximum_items]
        opposing = sorted(
            (item for item in evidence if item.contribution < 0),
            key=lambda item: abs(item.contribution),
            reverse=True,
        )[:maximum_items]
        cumulative = _cumulative_depth(program.intercept, evidence)
        decisive_depth = _decisive_depth(cumulative, exclude_intercept=True)
        target_label = _class_label(program.target, class_names)
        alternative_label = (
            _class_label(alternative, class_names)
            if alternative is not None
            else "the strongest alternative"
        )
        margin = float(program.score(values)[0])
        text = _plain_english(
            target_label=target_label,
            alternative_label=alternative_label,
            margin=margin,
            supporting=supporting,
            opposing=opposing,
            decisive_depth=decisive_depth,
        )
        return ContrastiveExplanation(
            target=program.target,
            alternative=alternative,
            target_label=target_label,
            alternative_label=alternative_label,
            programme_margin=margin,
            supporting=supporting,
            opposing=opposing,
            decisive_depth=decisive_depth,
            cumulative_by_depth=cumulative,
            plain_english=text,
            limitations=(
                "The explanation is generated deterministically from the extracted programme; it is not free-form language-model narration.",
                "Its validity is bounded by the programme's observational and interventional fidelity certificate.",
                "Unresolved concepts retain technical identifiers rather than receiving invented semantic labels.",
            ),
            metadata={
                "programme_supports_target": bool(margin > 0.0),
                "target_decision_rule": "target margin greater than zero",
                "intercept_margin": float(program.intercept),
                "decisive_depth_definition": (
                    "earliest evidence-bearing depth after which the reconstructed margin "
                    "retains its final sign; the intercept-only state is excluded"
                ),
                "evidence_cancellation": _evidence_cancellation(evidence),
            },
        )


def _rule_label(rule: ProgrammeRule) -> str:
    if rule.kind == "main":
        return rule.labels[0]
    return f"interaction between {rule.labels[0]} and {rule.labels[1]}"


def _cumulative_depth(
    intercept: float,
    evidence: Sequence[ContrastiveEvidence],
) -> list[tuple[float, float]]:
    grouped: dict[float, float] = {}
    for item in evidence:
        depth = round(float(item.depth), 6)
        grouped[depth] = grouped.get(depth, 0.0) + item.contribution
    cumulative = float(intercept)
    output: list[tuple[float, float]] = [(0.0, cumulative)]
    for depth in sorted(grouped):
        cumulative += grouped[depth]
        output.append((depth, cumulative))
    return output


def _decisive_depth(
    cumulative: Sequence[tuple[float, float]],
    *,
    exclude_intercept: bool = True,
) -> float | None:
    if not cumulative:
        return None
    final_margin = float(cumulative[-1][1])
    if math.isclose(final_margin, 0.0, abs_tol=1e-12):
        return None
    final_sign = 1.0 if final_margin > 0 else -1.0
    start = 1 if exclude_intercept else 0
    for index, (depth, margin) in enumerate(cumulative[start:], start=start):
        if final_sign * margin <= 0:
            continue
        if all(final_sign * later_margin > 0 for _, later_margin in cumulative[index:]):
            return float(depth)
    return None


def _evidence_cancellation(evidence: Sequence[ContrastiveEvidence]) -> float:
    contributions = np.asarray([item.contribution for item in evidence], dtype=float)
    absolute_mass = float(np.abs(contributions).sum())
    if absolute_mass <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - abs(float(contributions.sum())) / absolute_mass, 0.0, 1.0))


def _class_label(index: int | None, names: Mapping[int, str] | Sequence[str] | None) -> str:
    if index is None:
        return "the strongest alternative"
    if names is None:
        return f"class {index}"
    if isinstance(names, Mapping):
        return str(names.get(index, f"class {index}"))
    if 0 <= index < len(names):
        return str(names[index])
    return f"class {index}"


def _plain_english(
    *,
    target_label: str,
    alternative_label: str,
    margin: float,
    supporting: Sequence[ContrastiveEvidence],
    opposing: Sequence[ContrastiveEvidence],
    decisive_depth: float | None,
) -> str:
    support_text = _join_evidence(supporting, positive=True)
    opposition_text = _join_evidence(opposing, positive=False)
    if margin > 0.0:
        decision = (
            f"The extracted programme preferred {target_label} over {alternative_label} "
            f"with a reconstructed margin of {margin:.3f}."
        )
    else:
        decision = (
            f"The extracted programme did not reproduce the model's preference for "
            f"{target_label} on this sample. Its reconstructed margin was {margin:.3f}, "
            f"which favours {alternative_label}. The local contrastive account is therefore "
            "reported as a failed fidelity case rather than a valid explanation."
        )
    if support_text:
        decision += f" The strongest supporting mechanism was {support_text}."
    if opposition_text:
        if margin > 0.0:
            decision += f" Evidence favouring {alternative_label} was present through {opposition_text}, but it did not outweigh the supporting mechanism."
        else:
            decision += f" Evidence favouring {alternative_label} was present through {opposition_text} and outweighed the reconstructed support for {target_label}."
    if decisive_depth is not None and margin > 0.0:
        decision += f" The programme first became stably favourable to {target_label} at normalised model depth {decisive_depth:.2f}."
    return decision


def _join_evidence(items: Sequence[ContrastiveEvidence], *, positive: bool) -> str:
    if not items:
        return ""
    phrases = []
    for item in items[:3]:
        magnitude = abs(item.contribution)
        phrases.append(f"{item.label} ({magnitude:.3f} margin units)")
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]
