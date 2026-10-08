"""Layer-wise Relevance Propagation for supported PyTorch modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from vraven.attribution.rules import is_supported_module, lrp_embedding, lrp_linear, lrp_relu
from vraven.core.device import infer_model_device, prepare_model_inputs
from vraven.core.errors import CapabilityError
from vraven.core.output import compute_prediction, extract_primary_tensor


@dataclass(slots=True)
class LayerRelevance:
    name: str
    module_type: str
    relevance_values: list[float]
    relevance_shape: tuple[int, ...]
    relevance_sum: float
    mean_abs: float
    max_abs: float


@dataclass(slots=True)
class LRPResult:
    layers: list[LayerRelevance]
    input_relevance: list[float] | None
    input_relevance_shape: tuple[int, ...] | None
    target_score: float
    conservation_ratio: float
    epsilon: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def layer_matrix(self, max_units: int = 64) -> tuple[list[str], np.ndarray]:
        """Return layer names and a [layers, units] relevance magnitude matrix."""

        names = [layer.name for layer in self.layers]
        rows: list[np.ndarray] = []
        for layer in self.layers:
            values = np.abs(np.asarray(layer.relevance_values, dtype=np.float64)).reshape(-1)
            if values.size >= max_units:
                index = np.linspace(0, values.size - 1, max_units, dtype=int)
                rows.append(values[index])
            else:
                rows.append(np.pad(values, (0, max_units - values.size)))
        if not rows:
            return [], np.zeros((0, max_units), dtype=np.float64)
        return names, np.vstack(rows)


def _compress(values: Tensor, limit: int = 256) -> tuple[list[float], tuple[int, ...]]:
    flat = values.detach().float().reshape(-1).cpu()
    shape = tuple(int(dim) for dim in values.shape)
    if flat.numel() <= limit:
        return flat.tolist(), shape
    index = torch.linspace(0, flat.numel() - 1, steps=limit).long()
    return flat[index].tolist(), shape


def _module_relevance(
    module: nn.Module,
    activation: Tensor,
    relevance: Tensor,
    *,
    epsilon: float,
    indices: Tensor | None = None,
) -> Tensor:
    if isinstance(module, nn.Linear):
        return lrp_linear(module, activation, relevance, epsilon=epsilon)
    if isinstance(module, nn.ReLU):
        return lrp_relu(module, activation, relevance, epsilon=epsilon)
    if isinstance(module, nn.Embedding):
        if indices is None:
            raise CapabilityError("Embedding LRP requires the forward indices tensor")
        return lrp_embedding(module, indices, relevance, epsilon=epsilon)
    raise CapabilityError(f"No LRP rule for module type {type(module).__name__}")


def explain_lrp(
    model: nn.Module,
    inputs: Any,
    *,
    target: int | Sequence[int] | torch.Tensor | None = None,
    epsilon: float = 1e-6,
    auto_move_inputs: bool = True,
) -> LRPResult:
    """Run epsilon-rule LRP over Linear, ReLU and Embedding modules.

    Unsupported module types between supported layers are skipped for storage but
    do not receive a redistribution rule. Models composed only of unsupported
    ops raise ``CapabilityError``.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    device = infer_model_device(model)
    args, kwargs = prepare_model_inputs(
        inputs,
        device=device,
        move=auto_move_inputs,
        require_grad=False,
    )

    activations: dict[str, Tensor] = {}
    module_types: dict[str, str] = {}
    embedding_indices: dict[str, Tensor] = {}
    order: list[str] = []
    handles = []

    named = [
        (name, module)
        for name, module in model.named_modules()
        if name and is_supported_module(module)
    ]
    if not named:
        raise CapabilityError(
            "LRP currently supports only nn.Linear, nn.ReLU and nn.Embedding modules"
        )

    def make_hook(name: str, module: nn.Module):
        def hook(_module: nn.Module, module_inputs: tuple[Any, ...], output: Any) -> None:
            if name not in order:
                order.append(name)
            module_types[name] = type(module).__name__
            if isinstance(module, nn.Embedding):
                if not module_inputs or not isinstance(module_inputs[0], Tensor):
                    raise CapabilityError(f"Embedding module {name!r} did not receive index tensor")
                embedding_indices[name] = module_inputs[0].detach()
                activations[name] = output.detach() if isinstance(output, Tensor) else output
            else:
                if not module_inputs or not isinstance(module_inputs[0], Tensor):
                    raise CapabilityError(f"Module {name!r} did not receive a tensor activation")
                activations[name] = module_inputs[0].detach()

        return hook

    for name, module in named:
        handles.append(module.register_forward_hook(make_hook(name, module)))

    previous = model.training
    try:
        model.eval()
        with torch.inference_mode():
            output = model(*args, **kwargs)
        logits = extract_primary_tensor(output)
        computation = compute_prediction(logits, target)
        score = computation.score.detach()
        # Seed relevance at the model output.
        if logits.ndim == 0:
            relevance = score.reshape(1).expand_as(logits.reshape(1)).reshape_as(logits)
        elif logits.ndim == 1:
            relevance = logits.detach()
        else:
            # Prefer selected class channel when available.
            if computation.target_indices is not None and logits.shape[-1] > 1:
                relevance = torch.zeros_like(logits)
                relevance.scatter_(
                    1,
                    computation.target_indices.reshape(-1, 1),
                    computation.selected_scores.reshape(-1, 1),
                )
            else:
                relevance = logits.detach()

        layer_records: list[LayerRelevance] = []
        current = relevance
        # Propagate through supported modules in reverse execution order.
        for name in reversed(order):
            module = dict(named)[name]
            activation = activations[name]
            if isinstance(module, nn.Embedding):
                # Embedding relevance is summarised as a table; stop feature chain here.
                table = _module_relevance(
                    module,
                    activation,
                    current if current.shape == activation.shape else activation * 0 + current.mean(),
                    epsilon=epsilon,
                    indices=embedding_indices[name],
                )
                values, shape = _compress(table)
                layer_records.append(
                    LayerRelevance(
                        name=name,
                        module_type=module_types[name],
                        relevance_values=values,
                        relevance_shape=shape,
                        relevance_sum=float(table.sum().cpu()),
                        mean_abs=float(table.abs().mean().cpu()),
                        max_abs=float(table.abs().max().cpu()),
                    )
                )
                continue
            if (
                current.shape != activation.shape
                and current.ndim == activation.ndim
                and current.shape[0] == activation.shape[0]
                and current.numel() != activation.numel()
            ):
                # Fall back to uniform redistribution onto the activation shape.
                current = torch.full_like(
                    activation,
                    float(current.sum() / max(activation.numel(), 1)),
                )
            try:
                redistributed = _module_relevance(
                    module,
                    activation,
                    current,
                    epsilon=epsilon,
                    indices=embedding_indices.get(name),
                )
            except RuntimeError:
                # Shape mismatch in exotic graphs: skip redistribution for this module.
                values, shape = _compress(current)
                layer_records.append(
                    LayerRelevance(
                        name=name,
                        module_type=module_types[name],
                        relevance_values=values,
                        relevance_shape=shape,
                        relevance_sum=float(current.sum().cpu()),
                        mean_abs=float(current.abs().mean().cpu()),
                        max_abs=float(current.abs().max().cpu()),
                    )
                )
                continue
            # Store relevance redistributed onto this module's input.
            values, shape = _compress(redistributed)
            layer_records.append(
                LayerRelevance(
                    name=name,
                    module_type=module_types[name],
                    relevance_values=values,
                    relevance_shape=shape,
                    relevance_sum=float(redistributed.sum().cpu()),
                    mean_abs=float(redistributed.abs().mean().cpu()),
                    max_abs=float(redistributed.abs().max().cpu()),
                )
            )
            current = redistributed

        layer_records.reverse()
        input_values = None
        input_shape = None
        if isinstance(args[0], Tensor) if args else False:
            # Best-effort: project remaining relevance onto the first floating input.
            first = args[0]
            if first.is_floating_point():
                if current.shape == first.shape:
                    mapped = current
                elif current.numel() == first.numel():
                    mapped = current.reshape_as(first)
                else:
                    mapped = torch.full_like(first, float(current.sum() / max(first.numel(), 1)))
                input_values, input_shape = _compress(mapped)

        output_sum = float(relevance.sum().cpu())
        input_sum = float(np.sum(input_values)) if input_values is not None else float(current.sum().cpu())
        conservation = float(input_sum / output_sum) if abs(output_sum) > 1e-12 else 1.0

        return LRPResult(
            layers=layer_records,
            input_relevance=input_values,
            input_relevance_shape=input_shape,
            target_score=float(score.cpu()),
            conservation_ratio=conservation,
            epsilon=epsilon,
            metadata={
                "supported_module_types": sorted({module_types[name] for name in order}),
                "module_count": len(layer_records),
                "rules": ["epsilon-linear", "relu-mask", "embedding-index-add"],
            },
        )
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
