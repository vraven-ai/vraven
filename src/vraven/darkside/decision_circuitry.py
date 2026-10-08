"""Approximate sparse decision circuitry through constrained causal pruning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vraven.core.config import VRavenConfig
from vraven.visualisation.labels import unit_group_source_label

from .contracts import AnalysisResult, UnitGroup
from .interventions import InterventionEngine


def _unit_axis(module: torch.nn.Module, tensor: torch.Tensor) -> int:
    convolutional = (
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )
    if isinstance(module, convolutional) and tensor.ndim >= 2:
        return 1
    return max(tensor.ndim - 1, 0)


def _default_layers(model: torch.nn.Module) -> tuple[str, ...]:
    supported = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.MultiheadAttention,
        torch.nn.GRU,
        torch.nn.LSTM,
    )
    names = [name for name, module in model.named_modules() if name and isinstance(module, supported)]
    if not names:
        names = [
            name
            for name, module in model.named_modules()
            if name and not any(True for _ in module.children())
        ]
    return tuple(names)


def discover_decision_circuitry(
    model: torch.nn.Module,
    inputs: Any,
    *,
    target: int | None,
    layer_names: Sequence[str] | None = None,
    group_size: int = 8,
    relative_score_tolerance: float = 0.05,
    absolute_score_tolerance: float | None = None,
    baseline: str = "zero",
    maximum_groups: int = 256,
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Find an approximate epsilon-sufficient sparse internal core.

    The candidate universe is formed by contiguous unit groups from the selected
    layer outputs. The routine first measures each group's individual ablation
    effect, then greedily removes low-effect groups while the target prediction
    and declared score tolerance are preserved. The result is order-dependent
    and is not claimed to be the globally minimal circuit.
    """

    if group_size < 1:
        raise ValueError("group_size must be positive")
    if relative_score_tolerance < 0:
        raise ValueError("relative_score_tolerance cannot be negative")
    if absolute_score_tolerance is not None and absolute_score_tolerance < 0:
        raise ValueError("absolute_score_tolerance cannot be negative")
    if baseline not in {"zero", "mean"}:
        raise ValueError("baseline must be 'zero' or 'mean'")
    if maximum_groups < 1:
        raise ValueError("maximum_groups must be positive")

    selected_layers = tuple(layer_names or _default_layers(model))
    if not selected_layers:
        raise ValueError("No candidate layers were selected")

    engine = InterventionEngine(model, inputs, target=target, config=config)
    original = engine.evaluate()
    captured = engine.capture(selected_layers)
    modules = dict(model.named_modules())

    candidates: list[UnitGroup] = []
    layer_coverage: dict[str, dict[str, int]] = {}
    for identifier, tensor in captured.items():
        layer_name, call_text = identifier.rsplit("#", 1)
        axis = _unit_axis(modules[layer_name], tensor)
        width = int(tensor.shape[axis]) if tensor.ndim else 1
        groups = []
        for start in range(0, width, group_size):
            indices = tuple(range(start, min(start + group_size, width)))
            groups.append(
                UnitGroup(
                    layer_name=layer_name,
                    indices=indices,
                    axis=axis,
                    call_index=int(call_text),
                    label=unit_group_source_label(indices),
                )
            )
        candidates.extend(groups)
        layer_coverage[identifier] = {"units": width, "groups": len(groups), "axis": axis}

    if not candidates:
        raise ValueError("Selected layers did not expose any tensor unit groups")
    if len(candidates) > maximum_groups:
        raise ValueError(
            f"Candidate universe contains {len(candidates)} groups, exceeding maximum_groups="
            f"{maximum_groups}. Increase group_size or narrow layer_names."
        )

    original_score = float(original.trace.target_score)
    original_prediction = original.trace.predicted_index
    tolerance = (
        float(absolute_score_tolerance)
        if absolute_score_tolerance is not None
        else relative_score_tolerance * max(abs(original_score), 1e-12)
    )

    individual_effects: dict[str, float] = {}
    individual_predictions: dict[str, int | None] = {}
    evaluations = 1
    for group in candidates:
        selection = engine.intervene([group], baseline=baseline)
        evaluations += 1
        individual_effects[group.identifier] = original_score - float(selection.trace.target_score)
        individual_predictions[group.identifier] = selection.trace.predicted_index

    ordered = sorted(candidates, key=lambda group: abs(individual_effects[group.identifier]))
    removed: list[UnitGroup] = []
    retained = list(candidates)
    pruning_rows = []
    for group in ordered:
        proposed = [*removed, group]
        selection = engine.intervene(proposed, baseline=baseline)
        evaluations += 1
        score = float(selection.trace.target_score)
        prediction_preserved = (
            original_prediction is None or selection.trace.predicted_index == original_prediction
        )
        score_preserved = abs(score - original_score) <= tolerance
        accepted = bool(prediction_preserved and score_preserved)
        pruning_rows.append(
            {
                "group": group.identifier,
                "candidate_score": score,
                "score_change": score - original_score,
                "prediction": selection.trace.predicted_index,
                "accepted_removal": accepted,
            }
        )
        if accepted:
            removed.append(group)
            retained.remove(group)

    final_selection = engine.intervene(removed, baseline=baseline) if removed else original
    if removed:
        evaluations += 1
    final_score = float(final_selection.trace.target_score)
    necessity_rows = []
    for group in retained:
        selection = engine.intervene([*removed, group], baseline=baseline)
        evaluations += 1
        necessity_rows.append(
            {
                "group": group.identifier,
                "conditional_score_drop": final_score - float(selection.trace.target_score),
                "prediction_flipped": selection.trace.predicted_index != final_selection.trace.predicted_index,
            }
        )

    retained_ids = {group.identifier for group in retained}
    group_rows = []
    for group in candidates:
        group_rows.append(
            {
                "group": group.identifier,
                "layer": group.layer_identifier,
                "axis": group.axis,
                "indices": list(group.indices),
                "individual_score_drop": individual_effects[group.identifier],
                "individual_prediction": individual_predictions[group.identifier],
                "retained": group.identifier in retained_ids,
            }
        )

    retained_fraction = len(retained) / len(candidates) if candidates else 0.0
    score_retention = (
        1.0 - abs(final_score - original_score) / max(abs(original_score), 1e-12)
    )
    return AnalysisResult(
        analysis="decision_circuitry",
        metrics={
            "original_target_score": original_score,
            "final_target_score": final_score,
            "score_retention": score_retention,
            "original_prediction": original_prediction,
            "final_prediction": final_selection.trace.predicted_index,
            "candidate_groups": len(candidates),
            "retained_groups": len(retained),
            "removed_groups": len(removed),
            "retained_fraction": retained_fraction,
            "evaluations": evaluations,
            "absolute_score_tolerance": tolerance,
        },
        tables={
            "groups": group_rows,
            "pruning_path": pruning_rows,
            "conditional_necessity": necessity_rows,
            "layer_coverage": layer_coverage,
        },
        metadata={
            "claim_type": "intervention-based approximate circuit",
            "algorithm": "individual-effect ordering plus greedy backward elimination",
            "baseline": baseline,
            "group_size": group_size,
            "candidate_layers": list(selected_layers),
            "candidate_universe_complete_for_selected_axes": True,
        },
        warnings=[
            "The sparse core is conditional on the chosen layers, grouping, baseline, sample and pruning order."
        ],
        limitations=[
            "The routine does not solve the combinatorial global minimum circuit problem.",
            "Ablated activations may be off the model's normal activation manifold.",
            "Interactions can make individually weak groups jointly important.",
        ],
    )
