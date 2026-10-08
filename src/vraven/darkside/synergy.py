"""Pairwise causal synergy, redundancy and compensation diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from vraven.core.config import VRavenConfig

from .contracts import AnalysisResult, UnitGroup
from .interventions import InterventionEngine


def analyse_causal_synergy(
    model: torch.nn.Module,
    inputs: Any,
    *,
    target: int | None,
    groups: Sequence[UnitGroup],
    baseline: str = "zero",
    maximum_pairs: int = 256,
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Estimate pairwise interaction using single and joint ablations.

    For target score ``s`` and score drops ``d(A)``, ``d(B)`` and ``d(A,B)``, the
    reported interaction is ``d(A,B) - d(A) - d(B)``. Positive values are
    super-additive under this intervention, while negative values indicate
    redundancy or compensation. The quantity is baseline and context specific.
    """

    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    if len({group.identifier for group in groups}) != len(groups):
        raise ValueError("groups must have unique identifiers")
    if baseline not in {"zero", "mean"}:
        raise ValueError("baseline must be 'zero' or 'mean'")
    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be positive")
    pairs = len(groups) * (len(groups) - 1) // 2
    if pairs > maximum_pairs:
        raise ValueError(
            f"Requested {pairs} pairs, exceeding maximum_pairs={maximum_pairs}"
        )
    engine = InterventionEngine(model, inputs, target=target, config=config)
    original = engine.evaluate()
    original_score = float(original.trace.target_score)
    single_drops = {}
    evaluations = 1
    for group in groups:
        selection = engine.intervene([group], baseline=baseline)
        evaluations += 1
        single_drops[group.identifier] = original_score - float(selection.trace.target_score)

    matrix = np.zeros((len(groups), len(groups)), dtype=np.float64)
    rows = []
    for i, left in enumerate(groups):
        for j in range(i + 1, len(groups)):
            right = groups[j]
            selection = engine.intervene([left, right], baseline=baseline)
            evaluations += 1
            joint_drop = original_score - float(selection.trace.target_score)
            interaction = joint_drop - single_drops[left.identifier] - single_drops[right.identifier]
            matrix[i, j] = matrix[j, i] = interaction
            rows.append(
                {
                    "group_a": left.identifier,
                    "group_b": right.identifier,
                    "drop_a": single_drops[left.identifier],
                    "drop_b": single_drops[right.identifier],
                    "joint_drop": joint_drop,
                    "interaction": interaction,
                    "interaction_type": (
                        "super_additive"
                        if interaction > 0
                        else "redundant_or_compensatory"
                        if interaction < 0
                        else "additive"
                    ),
                }
            )
    strongest = max(rows, key=lambda row: abs(row["interaction"]))
    return AnalysisResult(
        analysis="causal_synergy",
        metrics={
            "groups": len(groups),
            "pairs": len(rows),
            "evaluations": evaluations,
            "strongest_pair": [strongest["group_a"], strongest["group_b"]],
            "strongest_interaction": strongest["interaction"],
        },
        tables={
            "group_names": [group.identifier for group in groups],
            "interaction_matrix": matrix.tolist(),
            "pairs": rows,
        },
        metadata={
            "claim_type": "pairwise intervention interaction",
            "baseline": baseline,
            "interaction_definition": "joint_drop - drop_a - drop_b",
        },
        limitations=[
            "Pairwise analysis does not recover higher-order interactions.",
            "The result changes with the baseline, target score and surrounding active mechanism.",
        ],
    )
