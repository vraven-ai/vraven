"""Constrained gradient counterfactuals for floating PyTorch inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from vraven.core.device import infer_model_device
from vraven.core.errors import CapabilityError
from vraven.core.output import compute_prediction, extract_primary_tensor


@dataclass(slots=True)
class CounterfactualResult:
    target_class: int
    original_prediction: int
    counterfactual_prediction: int
    original_target_score: float
    counterfactual_target_score: float
    success: bool
    steps_completed: int
    original_shape: tuple[int, ...]
    original_values: list[float]
    counterfactual_values: list[float]
    delta_values: list[float]
    top_changed_indices: list[int]
    objective_history: list[float]
    metadata: dict[str, Any]


def generate_counterfactual(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    target_class: int,
    steps: int = 100,
    learning_rate: float = 0.05,
    l2_weight: float = 0.01,
    max_linf: float | None = None,
    clamp: tuple[float, float] | None = None,
    top_changes: int = 20,
) -> CounterfactualResult:
    """Optimise a bounded input change towards one target class.

    This generic method is intended for continuous tensors. Domain constraints
    such as valid bytecode, medical physiology or token grammar require a domain
    adapter and must not be inferred from this routine.
    """

    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise CapabilityError("Generic counterfactual generation requires a floating tensor input")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be greater than zero")
    device = infer_model_device(model)
    original = inputs.detach().to(device)
    delta = torch.zeros_like(original, requires_grad=True)
    previous_training = model.training
    history: list[float] = []
    completed = 0
    try:
        model.eval()
        with torch.no_grad():
            original_output = extract_primary_tensor(model(original))
            original_computation = compute_prediction(original_output, target_class)
        def _objective_and_prediction(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Multiclass logits.
            if output.ndim == 2 and output.shape[-1] > 1:
                if output.shape[-1] <= target_class:
                    raise CapabilityError(
                        "Counterfactual class optimisation requires [batch, classes] logits "
                        "covering the requested target_class"
                    )
                targets = torch.full(
                    (output.shape[0],), target_class, dtype=torch.long, device=output.device
                )
                objective = functional.cross_entropy(output, targets)
                prediction = output.argmax(dim=-1)
                return objective, prediction
            # Binary logits [batch] or [batch, 1].
            if output.ndim == 1 or (output.ndim == 2 and output.shape[-1] == 1):
                logits = output.reshape(-1)
                if target_class not in (0, 1):
                    raise CapabilityError("Binary counterfactuals only support target_class 0 or 1")
                targets = torch.full_like(logits, float(target_class))
                objective = functional.binary_cross_entropy_with_logits(logits, targets)
                prediction = logits.ge(0).long()
                return objective, prediction
            raise CapabilityError(
                "Counterfactual optimisation supports [batch, classes], [batch] or [batch, 1] outputs"
            )

        for step in range(steps):
            candidate = original + delta
            if clamp is not None:
                candidate = candidate.clamp(*clamp)
            output = extract_primary_tensor(model(candidate))
            class_objective, prediction = _objective_and_prediction(output)
            objective = class_objective + l2_weight * delta.square().mean()
            gradient = torch.autograd.grad(objective, delta, create_graph=False)[0]
            with torch.no_grad():
                delta -= learning_rate * gradient
                if max_linf is not None:
                    delta.clamp_(-max_linf, max_linf)
            completed = step + 1
            history.append(float(objective.detach().cpu()))
            with torch.no_grad():
                if bool(torch.all(prediction.eq(target_class))):
                    break
        with torch.no_grad():
            counterfactual = original + delta
            if clamp is not None:
                counterfactual = counterfactual.clamp(*clamp)
            final_output = extract_primary_tensor(model(counterfactual))
            final_computation = compute_prediction(final_output, target_class)
            _, original_prediction_tensor = _objective_and_prediction(original_output)
            _, counterfactual_prediction_tensor = _objective_and_prediction(final_output)
            original_prediction = int(original_prediction_tensor.reshape(-1)[0].cpu())
            counterfactual_prediction = int(counterfactual_prediction_tensor.reshape(-1)[0].cpu())
            original_flat = original[0].reshape(-1) if original.ndim > 1 else original.reshape(-1)
            final_flat = (
                counterfactual[0].reshape(-1)
                if counterfactual.ndim > 1
                else counterfactual.reshape(-1)
            )
            change = final_flat - original_flat
            count = min(top_changes, change.numel())
            indices = change.abs().topk(k=count).indices if count else torch.empty(0, dtype=torch.long)
            return CounterfactualResult(
                target_class=target_class,
                original_prediction=original_prediction,
                counterfactual_prediction=counterfactual_prediction,
                original_target_score=float(original_computation.selected_scores.mean().cpu()),
                counterfactual_target_score=float(final_computation.selected_scores.mean().cpu()),
                success=counterfactual_prediction == target_class,
                steps_completed=completed,
                original_shape=tuple(original.shape),
                original_values=original_flat.index_select(0, indices).cpu().tolist(),
                counterfactual_values=final_flat.index_select(0, indices).cpu().tolist(),
                delta_values=change.index_select(0, indices).cpu().tolist(),
                top_changed_indices=indices.cpu().tolist(),
                objective_history=history,
                metadata={
                    "method": "constrained_gradient_input_optimisation",
                    "l2_weight": l2_weight,
                    "max_linf": max_linf,
                    "clamp": clamp,
                    "domain_validity": "not_guaranteed_without_domain_adapter",
                },
            )
    finally:
        model.train(previous_training)
