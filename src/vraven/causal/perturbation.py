"""Prediction-stability tests under controlled input perturbations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from vraven.core.device import infer_model_device, prepare_model_inputs
from vraven.core.errors import CapabilityError
from vraven.core.output import compute_prediction, extract_primary_tensor

PerturbKind = Literal["noise", "feature_mask", "shuffle"]


@dataclass(slots=True)
class PerturbationCurve:
    kind: PerturbKind
    levels: list[float]
    stability: list[float]


@dataclass(slots=True)
class PerturbationResult:
    levels: list[float]
    curves: list[PerturbationCurve]
    baseline_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _floating_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, Tensor) and value.is_floating_point():
            return value
    raise CapabilityError("Perturbation tests require a floating tensor input")


def _score(model: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], target: int | None) -> float:
    output = model(*args, **kwargs)
    logits = extract_primary_tensor(output)
    return float(compute_prediction(logits, target).score.detach().cpu())


def _replace_first_float(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    replacement: Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    for index, value in enumerate(new_args):
        if isinstance(value, Tensor) and value.is_floating_point():
            new_args[index] = replacement
            return tuple(new_args), kwargs
    new_kwargs = dict(kwargs)
    for key, value in new_kwargs.items():
        if isinstance(value, Tensor) and value.is_floating_point():
            new_kwargs[key] = replacement
            return args, new_kwargs
    raise CapabilityError("Unable to replace floating input for perturbation")


def run_perturbation_stability(
    model: nn.Module,
    inputs: Any,
    *,
    target: int | None = None,
    levels: list[float] | None = None,
    kinds: tuple[PerturbKind, ...] = ("noise", "feature_mask", "shuffle"),
    repeats: int = 5,
    random_seed: int = 17,
    auto_move_inputs: bool = True,
) -> PerturbationResult:
    """Estimate score stability under increasing perturbation strength.

    Stability at each level is the mean cosine similarity between the vector
    ``[baseline_score]`` and per-repeat perturbed scores after normalising by
    the baseline magnitude, clipped to ``[0, 1]`` as a practical stability index:
    ``1 - |score' - score| / (|score| + eps)``.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    resolved_levels = levels or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    device = infer_model_device(model)
    args, kwargs = prepare_model_inputs(
        inputs,
        device=device,
        move=auto_move_inputs,
        require_grad=False,
    )
    base = _floating_input(args, kwargs).detach()
    previous = model.training
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    try:
        model.eval()
        with torch.inference_mode():
            baseline = _score(model, args, kwargs, target)
            curves: list[PerturbationCurve] = []
            for kind in kinds:
                stability_values: list[float] = []
                for level in resolved_levels:
                    strengths: list[float] = []
                    for _ in range(repeats):
                        perturbed = base.clone()
                        if kind == "noise":
                            noise = torch.randn(
                                perturbed.shape,
                                generator=generator,
                                dtype=perturbed.dtype,
                            )
                            scale = level * (perturbed.std().clamp_min(1e-6))
                            perturbed = perturbed + noise.to(perturbed.device) * scale
                        elif kind == "feature_mask":
                            mask = torch.rand(perturbed.shape, generator=generator) >= level
                            perturbed = perturbed * mask.to(device=perturbed.device, dtype=perturbed.dtype)
                        elif kind == "shuffle":
                            flat = perturbed.reshape(perturbed.shape[0], -1)
                            if flat.shape[1] > 1 and level > 0:
                                count = max(1, int(round(level * flat.shape[1])))
                                for row in range(flat.shape[0]):
                                    idx = torch.randperm(flat.shape[1], generator=generator)[:count]
                                    values = flat[row, idx].clone()
                                    perm = torch.randperm(count, generator=generator)
                                    flat[row, idx] = values[perm]
                            perturbed = flat.reshape_as(perturbed)
                        else:
                            raise ValueError(f"Unsupported perturbation kind: {kind}")
                        p_args, p_kwargs = _replace_first_float(args, kwargs, perturbed)
                        score = _score(model, p_args, p_kwargs, target)
                        strengths.append(
                            float(
                                max(
                                    0.0,
                                    1.0 - abs(score - baseline) / (abs(baseline) + 1e-6),
                                )
                            )
                        )
                    stability_values.append(float(np.mean(strengths)))
                curves.append(
                    PerturbationCurve(
                        kind=kind,
                        levels=[float(level) for level in resolved_levels],
                        stability=stability_values,
                    )
                )
        return PerturbationResult(
            levels=[float(level) for level in resolved_levels],
            curves=curves,
            baseline_score=baseline,
            metadata={"repeats": repeats, "random_seed": random_seed},
        )
    finally:
        model.train(previous)
