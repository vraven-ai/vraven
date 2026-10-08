"""Activation patching and bounded mediation-style recovery analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vraven.core.config import VRavenConfig

from .contracts import AnalysisResult
from .interventions import InterventionEngine


def analyse_activation_mediation(
    model: torch.nn.Module,
    clean_inputs: Any,
    corrupted_inputs: Any,
    *,
    target: int | None,
    layer_names: Sequence[str],
    include_reverse_noising: bool = True,
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Patch clean activations into a corrupted run and measure score recovery.

    This is an intervention-specific mediation diagnostic. Recovery can contain
    interaction effects and must not be interpreted as a pure natural indirect
    effect without stronger causal assumptions.
    """

    if not layer_names:
        raise ValueError("layer_names cannot be empty")
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("layer_names must be unique")
    clean_engine = InterventionEngine(model, clean_inputs, target=target, config=config)
    corrupt_engine = InterventionEngine(model, corrupted_inputs, target=target, config=config)
    clean_selection = clean_engine.evaluate()
    corrupt_selection = corrupt_engine.evaluate()
    clean_score = float(clean_selection.trace.target_score)
    corrupt_score = float(corrupt_selection.trace.target_score)
    total_effect = clean_score - corrupt_score
    clean_activations = clean_engine.capture(layer_names)
    corrupt_activations = corrupt_engine.capture(layer_names)
    if set(clean_activations) != set(corrupt_activations):
        raise ValueError("Clean and corrupted runs exposed different layer invocations")
    if not clean_activations:
        raise ValueError("Selected layers did not expose tensor activations")
    rows = []
    for identifier, activation in clean_activations.items():
        layer_name, call_text = identifier.rsplit("#", 1)
        patched = corrupt_engine.patch(
            layer_name=layer_name,
            call_index=int(call_text),
            reference_activation=activation,
        )
        patched_score = float(patched.trace.target_score)
        recovery = patched_score - corrupt_score
        row = {
            "layer": identifier,
            "patched_corrupt_score": patched_score,
            "clean_to_corrupt_recovery": recovery,
            "recovery_fraction": recovery / total_effect if abs(total_effect) > 1e-12 else None,
        }
        if include_reverse_noising and identifier in corrupt_activations:
            noised = clean_engine.patch(
                layer_name=layer_name,
                call_index=int(call_text),
                reference_activation=corrupt_activations[identifier],
            )
            noised_score = float(noised.trace.target_score)
            row.update(
                {
                    "patched_clean_score": noised_score,
                    "corrupt_to_clean_damage": clean_score - noised_score,
                    "damage_fraction": (
                        (clean_score - noised_score) / total_effect
                        if abs(total_effect) > 1e-12
                        else None
                    ),
                }
            )
        rows.append(row)
    strongest = max(rows, key=lambda row: abs(row["clean_to_corrupt_recovery"]))
    return AnalysisResult(
        analysis="activation_mediation",
        metrics={
            "clean_score": clean_score,
            "corrupted_score": corrupt_score,
            "total_effect": total_effect,
            "layers_patched": len(rows),
            "strongest_recovery_layer": strongest["layer"],
            "strongest_recovery": strongest["clean_to_corrupt_recovery"],
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "activation patching intervention",
            "direction": "clean activation into corrupted execution",
            "reverse_noising": include_reverse_noising,
        },
        warnings=[
            "Patching estimates can include interaction effects and off-manifold hybrid states."
        ],
        limitations=[
            "Recovery depends on how the corrupted input was constructed.",
            "A large patched recovery does not identify a self-contained mechanism.",
        ],
    )
