"""Scoped PyTorch hooks that create reduced layer traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from vraven.capture.selectors import ModuleSelector
from vraven.capture.tensor_utils import (
    compressed_attention,
    compressed_representation,
    extract_tensor,
    unit_matrix,
)
from vraven.core.config import VRavenConfig
from vraven.core.trace import LayerTrace
from vraven.metrics.basic import activation_statistics, gradient_statistics
from vraven.metrics.influence import activation_gradient_totals


@dataclass(slots=True)
class _RetainedOutput:
    layer: LayerTrace
    tensor: torch.Tensor
    module: torch.nn.Module


@dataclass
class HookSession:
    """Register hooks for one forward pass and remove them unconditionally."""

    model: torch.nn.Module
    config: VRavenConfig
    traces: list[LayerTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._starts: dict[int, list[float]] = defaultdict(list)
        self._execution_index = 0
        self._name_counts: dict[str, int] = defaultdict(int)
        self.retained_outputs: list[_RetainedOutput] = []
        self.full_tensors: dict[str, torch.Tensor] = {}
        self.selected_modules = ModuleSelector(self.config).select(self.model)

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...] | None:
        tensor = extract_tensor(value)
        return tuple(tensor.shape) if tensor is not None else None

    def _pre_hook(self, module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if self.config.capture_timing:
            self._starts[id(module)].append(perf_counter())

    def _display_name(self, base_name: str) -> str:
        count = self._name_counts[base_name]
        self._name_counts[base_name] += 1
        return base_name if count == 0 else f"{base_name}#{count + 1}"

    def _post_hook(
        self,
        base_name: str,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        output_tensor = extract_tensor(output)
        if output_tensor is None:
            return
        elapsed_ms: float | None = None
        if self.config.capture_timing and self._starts[id(module)]:
            elapsed_ms = (perf_counter() - self._starts[id(module)].pop()) * 1000.0

        stats = activation_statistics(
            output_tensor,
            epsilon=self.config.epsilon,
            capture_entropy=self.config.capture_entropy,
            histogram_bins=self.config.histogram_bins,
            entropy_sample_limit=self.config.entropy_sample_limit,
        )
        display_name = self._display_name(base_name)
        representation: list[list[float]] | None = None
        units: list[list[float]] | None = None
        unit_activations: list[float] | None = None
        attention_values: list[list[list[float]]] | None = None
        attention_shape: tuple[int, ...] | None = None

        if self.config.capture_representations:
            representation_tensor = compressed_representation(
                output_tensor, self.config.representation_size
            )
            representation = representation_tensor.detach().cpu().tolist()
        if self.config.capture_unit_profiles:
            unit_tensor = unit_matrix(
                output_tensor,
                module=module,
                max_units=self.config.max_units,
            )
            units = unit_tensor.detach().cpu().tolist()
            unit_activations = unit_tensor.mean(dim=0).detach().cpu().tolist()
        if self.config.capture_attention:
            attention = compressed_attention(output, self.config.max_attention_tokens)
            if attention is not None:
                attention_shape = tuple(attention.shape)
                attention_values = attention.cpu().tolist()

        layer = LayerTrace(
            name=display_name,
            layer_type=module.__class__.__name__,
            execution_index=self._execution_index,
            input_shape=self._shape(inputs),
            output_shape=tuple(output_tensor.shape),
            device=str(output_tensor.device),
            dtype=str(output_tensor.dtype),
            parameter_count=sum(
                parameter.numel() for parameter in module.parameters(recurse=False)
            ),
            activation_mean=stats["mean"],
            activation_variance=stats["variance"],
            activation_std=stats["std"],
            activation_energy=stats["energy"],
            mean_energy=stats["mean_energy"],
            l1_magnitude=stats["l1"],
            absolute_mean=stats["absolute_mean"],
            minimum=stats["minimum"],
            maximum=stats["maximum"],
            sparsity=stats["sparsity"],
            positive_fraction=stats["positive_fraction"],
            negative_fraction=stats["negative_fraction"],
            activation_dispersion_entropy=stats["dispersion_entropy"],
            execution_time_ms=elapsed_ms,
            representation=representation,
            unit_matrix=units,
            unit_activations=unit_activations,
            attention=attention_values,
            attention_shape=attention_shape,
            metadata={
                "base_module_name": base_name,
                "timing_semantics": "host_observed_dispatch_time",
            },
        )
        self._execution_index += 1
        self.traces.append(layer)

        if self.config.capture_full_tensors:
            self.full_tensors[display_name] = output_tensor.detach()
        if self.config.capture_gradients and output_tensor.requires_grad:
            self.retained_outputs.append(_RetainedOutput(layer, output_tensor, module))

    def gradient_tensors(self) -> list[torch.Tensor]:
        return [item.tensor for item in self.retained_outputs]

    def attach_gradients(self, gradients: list[torch.Tensor | None]) -> None:
        if len(gradients) != len(self.retained_outputs):
            raise ValueError("Gradient count does not match retained layer outputs")
        for retained, gradient in zip(self.retained_outputs, gradients, strict=True):
            if gradient is None:
                continue
            layer = retained.layer
            gradient_stats = gradient_statistics(gradient)
            totals = activation_gradient_totals(retained.tensor, gradient)
            layer.gradient_norm = gradient_stats["norm"]
            layer.gradient_mean_abs = gradient_stats["mean_abs"]
            layer.activation_gradient_signed = totals["signed"]
            layer.activation_gradient_influence = totals["absolute"]
            if self.config.capture_unit_profiles:
                gradient_units = unit_matrix(
                    gradient,
                    module=retained.module,
                    max_units=self.config.max_units,
                )
                product_units = unit_matrix(
                    retained.tensor.detach().float() * gradient.detach().float(),
                    module=retained.module,
                    max_units=self.config.max_units,
                )
                layer.unit_gradients = gradient_units.mean(dim=0).cpu().tolist()
                layer.unit_contributions = product_units.mean(dim=0).cpu().tolist()
        self.retained_outputs.clear()

    def __enter__(self) -> HookSession:
        for name, module in self.selected_modules:
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, module_name=name: self._post_hook(
                        module_name, mod, inp, out
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._starts.clear()
        if exc_type is not None:
            self.retained_outputs.clear()
