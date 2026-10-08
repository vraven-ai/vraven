"""Temporary unit, layer and activation interventions for live PyTorch models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from vraven.core.config import VRavenConfig
from vraven.core.output import TargetSelection, select_target
from vraven.core.tree import call_model, extract_tensors, replace_first_tensor

from .contracts import UnitGroup


@contextmanager
def evaluation_mode(model: torch.nn.Module):
    """Run deterministic interventions and restore every module training flag."""

    states = {module: bool(module.training) for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in states.items():
            module.training = training


def _normalise_axis(axis: int, ndim: int) -> int:
    resolved = axis if axis >= 0 else ndim + axis
    if not 0 <= resolved < ndim:
        raise IndexError(f"Axis {axis} is invalid for {ndim}-dimensional output")
    return resolved


def _replace_units(
    tensor: torch.Tensor,
    groups: Sequence[UnitGroup],
    *,
    baseline: str,
    reference: torch.Tensor | None,
) -> torch.Tensor:
    result = tensor.clone()
    for group in groups:
        axis = _normalise_axis(group.axis, result.ndim)
        indices = torch.as_tensor(group.indices, device=result.device, dtype=torch.long)
        if indices.numel() == 0:
            continue
        if int(indices.min()) < 0 or int(indices.max()) >= result.shape[axis]:
            raise IndexError(
                f"Intervention indices for {group.identifier} exceed axis width {result.shape[axis]}"
            )
        selection = [slice(None)] * result.ndim
        selection[axis] = indices
        selection_tuple = tuple(selection)
        if baseline == "zero":
            replacement = torch.zeros_like(result[selection_tuple])
        elif baseline == "mean":
            mean = result.mean(dim=axis, keepdim=True)
            replacement = mean.expand_as(result).index_select(axis, indices)
        elif baseline == "reference":
            if reference is None:
                raise ValueError("reference baseline requires a captured reference tensor")
            if reference.shape != result.shape:
                raise ValueError("Reference activation shape does not match intervened activation")
            replacement = reference.to(result.device, dtype=result.dtype).index_select(axis, indices)
        else:
            raise ValueError("baseline must be 'zero', 'mean' or 'reference'")
        result[selection_tuple] = replacement
    return result


class InterventionEngine:
    """Evaluate explicit activation interventions without permanently editing a model."""

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Any,
        *,
        target: int | None,
        config: VRavenConfig | None = None,
    ) -> None:
        self.model = model
        self.inputs = inputs
        self.target = target
        self.config = config or VRavenConfig(mode="fast")
        self.modules = dict(model.named_modules())

    def _selection(self, output: Any) -> TargetSelection:
        config = self.config
        return select_target(
            output,
            target=self.target,
            class_dim=config.class_dim,
            probability_transform=config.probability_transform,
            target_reduction=config.target_reduction,
            class_names=config.class_names,
            target_label=config.target_label,
            output_path=config.output_path,
        )

    def evaluate(self) -> TargetSelection:
        with evaluation_mode(self.model), torch.inference_mode():
            return self._selection(call_model(self.model, self.inputs))

    def capture(self, layer_names: Sequence[str]) -> dict[str, torch.Tensor]:
        requested = set(layer_names)
        missing = requested.difference(self.modules)
        if missing:
            raise KeyError(f"Unknown model layers: {sorted(missing)}")
        records: dict[str, torch.Tensor] = {}
        counts: defaultdict[str, int] = defaultdict(int)
        handles = []
        for name in requested:
            module = self.modules[name]

            def hook(_module: torch.nn.Module, _inputs: Any, output: Any, *, layer_name: str = name):
                call_index = counts[layer_name]
                counts[layer_name] += 1
                tensors = extract_tensors(output)
                if tensors:
                    records[f"{layer_name}#{call_index}"] = tensors[0].detach().clone()

            handles.append(module.register_forward_hook(hook))
        try:
            with evaluation_mode(self.model), torch.inference_mode():
                call_model(self.model, self.inputs)
        finally:
            for handle in handles:
                handle.remove()
        return records

    def intervene(
        self,
        groups: Sequence[UnitGroup],
        *,
        baseline: str = "zero",
        reference_activations: Mapping[str, torch.Tensor] | None = None,
    ) -> TargetSelection:
        grouped: defaultdict[str, list[UnitGroup]] = defaultdict(list)
        for group in groups:
            if group.layer_name not in self.modules:
                raise KeyError(f"Unknown model layer {group.layer_name!r}")
            grouped[group.layer_name].append(group)
        counts: defaultdict[str, int] = defaultdict(int)
        handles = []
        for layer_name, layer_groups in grouped.items():
            by_call: defaultdict[int, list[UnitGroup]] = defaultdict(list)
            for group in layer_groups:
                by_call[group.call_index].append(group)

            def hook(
                _module: torch.nn.Module,
                _inputs: Any,
                output: Any,
                *,
                name: str = layer_name,
                call_map: Mapping[int, list[UnitGroup]] = by_call,
            ):
                # Bind name/call_map as defaults so each layer keeps its own map
                # (loop-variable closure would otherwise share the final layer).
                call_index = counts[name]
                counts[name] += 1
                selected = call_map.get(call_index)
                if not selected:
                    return output
                reference = None
                if reference_activations is not None:
                    reference = reference_activations.get(f"{name}#{call_index}")
                return replace_first_tensor(
                    output,
                    lambda tensor: _replace_units(
                        tensor,
                        selected,
                        baseline=baseline,
                        reference=reference,
                    ),
                )

            handles.append(self.modules[layer_name].register_forward_hook(hook))
        try:
            with evaluation_mode(self.model), torch.inference_mode():
                output = call_model(self.model, self.inputs)
                return self._selection(output)
        finally:
            for handle in handles:
                handle.remove()

    def patch(
        self,
        *,
        layer_name: str,
        call_index: int,
        reference_activation: torch.Tensor,
    ) -> TargetSelection:
        """Replace the complete selected activation with a captured reference."""

        if layer_name not in self.modules:
            raise KeyError(f"Unknown model layer {layer_name!r}")
        invocation = 0

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any):
            nonlocal invocation
            current = invocation
            invocation += 1
            if current != call_index:
                return output
            return replace_first_tensor(
                output,
                lambda tensor: reference_activation.to(tensor.device, dtype=tensor.dtype),
            )

        handle = self.modules[layer_name].register_forward_hook(hook)
        try:
            with evaluation_mode(self.model), torch.inference_mode():
                return self._selection(call_model(self.model, self.inputs))
        finally:
            handle.remove()
