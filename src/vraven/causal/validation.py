"""Targeted intervention validation for candidate important layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vraven.causal.interventions import temporary_output_intervention
from vraven.core.device import infer_model_device, prepare_model_inputs
from vraven.core.output import compute_prediction, extract_primary_tensor
from vraven.core.trace import CausalInterventionTrace, NeuralTrace
from vraven.reasoning.pathways import rank_layers


def _base_module_name(trace: NeuralTrace, layer_name: str) -> str:
    for layer in trace.layers:
        if layer.name == layer_name:
            return str(layer.metadata.get("base_module_name", layer.name.split("#", 1)[0]))
    return layer_name.split("#", 1)[0]


def candidate_layer_names(trace: NeuralTrace, top_k: int) -> list[str]:
    ranked = rank_layers(trace, limit=top_k)
    if ranked:
        names = [item.name for item in ranked]
    else:
        candidates = sorted(
            trace.layers,
            key=lambda layer: float(layer.activation_energy or 0.0),
            reverse=True,
        )
        names = [layer.name for layer in candidates[:top_k]]
    resolved: list[str] = []
    for name in names:
        base = _base_module_name(trace, name)
        if base not in resolved:
            resolved.append(base)
    return resolved


def validate_layer_interventions(
    model: torch.nn.Module,
    inputs: Any,
    trace: NeuralTrace,
    *,
    target: int | Sequence[int] | torch.Tensor | None,
    top_k: int = 3,
    strategy: str = "zero",
    scale: float = 0.0,
    auto_move_inputs: bool = True,
) -> list[CausalInterventionTrace]:
    """Ablate a small set of candidate layers and measure target-score change."""

    device = infer_model_device(model)
    args, kwargs = prepare_model_inputs(
        inputs,
        device=device,
        move=auto_move_inputs,
        require_grad=False,
    )
    previous_training = model.training
    results: list[CausalInterventionTrace] = []
    try:
        model.eval()
        with torch.inference_mode():
            baseline_output = model(*args, **kwargs)
            baseline = compute_prediction(extract_primary_tensor(baseline_output), target)
            fixed_targets = baseline.target_indices
            baseline_scores = baseline.selected_scores.detach().float()
            baseline_predicted = (
                baseline.predicted_classes.detach().cpu().tolist()
                if baseline.predicted_classes is not None
                else [None] * baseline_scores.numel()
            )
            for layer_name in candidate_layer_names(trace, top_k):
                with temporary_output_intervention(
                    model,
                    layer_name,
                    strategy=strategy,
                    scale=scale,
                ):
                    intervened_output = model(*args, **kwargs)
                intervened = compute_prediction(
                    extract_primary_tensor(intervened_output),
                    fixed_targets,
                )
                intervened_scores = intervened.selected_scores.detach().float()
                delta = baseline_scores - intervened_scores
                baseline_mean = float(baseline_scores.mean().cpu())
                intervened_mean = float(intervened_scores.mean().cpu())
                delta_mean = float(delta.mean().cpu())
                after_predicted = (
                    intervened.predicted_classes.detach().cpu().tolist()
                    if intervened.predicted_classes is not None
                    else [None] * intervened_scores.numel()
                )
                target_index = (
                    int(fixed_targets.reshape(-1)[0].cpu()) if fixed_targets is not None else None
                )
                results.append(
                    CausalInterventionTrace(
                        layer_name=layer_name,
                        strategy=strategy,
                        baseline_target_score=baseline_mean,
                        intervened_target_score=intervened_mean,
                        score_delta=delta_mean,
                        relative_delta=(
                            delta_mean / abs(baseline_mean) if abs(baseline_mean) > 1e-8 else None
                        ),
                        prediction_before=(
                            int(baseline_predicted[0])
                            if baseline_predicted and baseline_predicted[0] is not None
                            else None
                        ),
                        prediction_after=(
                            int(after_predicted[0])
                            if after_predicted and after_predicted[0] is not None
                            else None
                        ),
                        target_index=target_index,
                        successful=abs(delta_mean) > 1e-8,
                        notes=(
                            "Positive score_delta means the intervention reduced the selected "
                            "target score. This supports intervention sensitivity, not a complete "
                            "causal account of the model."
                        ),
                    )
                )
    finally:
        model.train(previous_training)
    return results
