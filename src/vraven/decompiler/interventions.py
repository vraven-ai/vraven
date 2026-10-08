"""Graph-compiled interventions and causal effect estimation.

Unlike hook-only approaches, the compiler rewrites an FX/Export graph and inserts
explicit intervention modules after selected graph nodes.  The resulting GraphModule is
executable, serialisable with normal PyTorch tooling, inspectable, and deterministic.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Node

from .capture import CapturedModel
from .runtime import (
    FXValueRecorder,
    erase_concept,
    extract_logits,
    predicted_class,
    set_concept_coefficient,
    set_concept_mean,
    target_margin,
)


class InterventionKind(str, Enum):
    ZERO = "zero"
    REPLACE = "replace"
    SCALE = "scale"
    ADD = "add"
    ERASE_DIRECTION = "erase_direction"
    SET_CONCEPT = "set_concept"
    SET_CONCEPT_MEAN = "set_concept_mean"
    CLAMP = "clamp"


@dataclass(slots=True)
class InterventionSpec:
    node_id: str
    kind: InterventionKind
    label: str | None = None
    feature_axis: int = -1
    direction: Tensor | None = None
    baseline: Tensor | float | None = None
    strength: float = 1.0
    coefficient: float | Tensor | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def identifier(self) -> str:
        payload = (
            self.node_id,
            self.kind.value,
            self.label,
            self.feature_axis,
            self.strength,
            float(self.coefficient) if isinstance(self.coefficient, (int, float)) else None,
        )
        digest = hashlib.sha1(repr(payload).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        return f"{_safe_name(self.node_id)}_{self.kind.value}_{digest}"


class InterventionGate(nn.Module):
    """Small graph-insertable module implementing one declared intervention."""

    def __init__(self, spec: InterventionSpec) -> None:
        super().__init__()
        self.kind = spec.kind.value
        self.feature_axis = int(spec.feature_axis)
        self.strength = float(spec.strength)
        self.clamp_min = spec.clamp_min
        self.clamp_max = spec.clamp_max
        if spec.direction is not None:
            self.register_buffer("direction", spec.direction.detach().clone(), persistent=True)
        else:
            self.direction = None  # type: ignore[assignment]
        if isinstance(spec.baseline, Tensor):
            self.register_buffer("baseline_tensor", spec.baseline.detach().clone(), persistent=True)
            self.baseline_scalar = None
        else:
            self.baseline_tensor = None  # type: ignore[assignment]
            self.baseline_scalar = float(spec.baseline) if spec.baseline is not None else None
        if isinstance(spec.coefficient, Tensor):
            self.register_buffer("coefficient_tensor", spec.coefficient.detach().clone(), persistent=True)
            self.coefficient_scalar = None
        else:
            self.coefficient_tensor = None  # type: ignore[assignment]
            self.coefficient_scalar = (
                float(spec.coefficient) if spec.coefficient is not None else None
            )

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("VRAVEN intervention gates require tensor-valued graph nodes")
        if self.kind == InterventionKind.ZERO.value:
            return torch.zeros_like(value)
        if self.kind == InterventionKind.REPLACE.value:
            baseline = self._baseline(value)
            return torch.broadcast_to(baseline, value.shape).clone()
        if self.kind == InterventionKind.SCALE.value:
            return value * self.strength
        if self.kind == InterventionKind.ADD.value:
            return value + self._baseline(value) * self.strength
        if self.kind == InterventionKind.ERASE_DIRECTION.value:
            if self.direction is None:
                raise RuntimeError("erase_direction requires a direction tensor")
            return erase_concept(value, self.direction, self.feature_axis, self.strength)
        if self.kind in {
            InterventionKind.SET_CONCEPT.value,
            InterventionKind.SET_CONCEPT_MEAN.value,
        }:
            if self.direction is None:
                raise RuntimeError("concept setting requires a direction tensor")
            coefficient: Tensor | float
            if self.coefficient_tensor is not None:
                coefficient = self.coefficient_tensor
            elif self.coefficient_scalar is not None:
                coefficient = self.coefficient_scalar
            else:
                raise RuntimeError("concept setting requires a target coefficient")
            function = (
                set_concept_mean
                if self.kind == InterventionKind.SET_CONCEPT_MEAN.value
                else set_concept_coefficient
            )
            return function(
                value,
                self.direction,
                self.feature_axis,
                coefficient,
            )
        if self.kind == InterventionKind.CLAMP.value:
            return value.clamp(min=self.clamp_min, max=self.clamp_max)
        raise RuntimeError(f"Unknown intervention kind: {self.kind}")

    def _baseline(self, value: Tensor) -> Tensor:
        if self.baseline_tensor is not None:
            return self.baseline_tensor.to(device=value.device, dtype=value.dtype)
        if self.baseline_scalar is not None:
            return torch.as_tensor(self.baseline_scalar, device=value.device, dtype=value.dtype)
        return torch.zeros((), device=value.device, dtype=value.dtype)


@dataclass(slots=True)
class CompiledIntervention:
    module: GraphModule
    specs: tuple[InterventionSpec, ...]
    gate_modules: dict[str, str]
    gate_nodes: dict[str, str]
    graph_code: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class GraphInterventionCompiler:
    """Insert explicit gates into a captured FX or Export graph."""

    def __init__(self, captured: CapturedModel) -> None:
        if captured.graph_module is None:
            raise ValueError(
                "Graph-compiled interventions require an export or FX capture route. "
                f"The current route is {captured.ir.capture_route.value}."
            )
        self.captured = captured

    def compile(self, specs: Sequence[InterventionSpec]) -> CompiledIntervention:
        if not specs:
            raise ValueError("At least one intervention specification is required")
        graph_module = copy.deepcopy(self.captured.graph_module)
        nodes_by_name = {node.name: node for node in graph_module.graph.nodes}
        grouped: dict[str, list[InterventionSpec]] = {}
        for spec in specs:
            if spec.node_id not in nodes_by_name:
                available = ", ".join(list(nodes_by_name)[:20])
                raise KeyError(
                    f"No graph node named {spec.node_id!r}. First available nodes: {available}"
                )
            target_node = nodes_by_name[spec.node_id]
            if target_node.op in {"placeholder", "get_attr", "output"}:
                raise ValueError(
                    f"Node {spec.node_id!r} has op {target_node.op!r} and is not a supported "
                    "internal intervention point"
                )
            grouped.setdefault(spec.node_id, []).append(spec)

        gate_modules: dict[str, str] = {}
        gate_nodes: dict[str, str] = {}
        for target_name, target_specs in grouped.items():
            target_node = nodes_by_name[target_name]
            original_users = list(target_node.users)
            current: Node = target_node
            for ordinal, spec in enumerate(target_specs):
                gate_name = f"_vraven_gate_{spec.identifier()}_{ordinal}"
                while hasattr(graph_module, gate_name):
                    gate_name = f"{gate_name}_x"
                graph_module.add_module(gate_name, InterventionGate(spec))
                with graph_module.graph.inserting_after(current):
                    gate_node = graph_module.graph.call_module(gate_name, args=(current,))
                gate_node.meta.update(dict(target_node.meta))
                gate_node.meta["vraven_intervention"] = {
                    "node": spec.node_id,
                    "kind": spec.kind.value,
                    "label": spec.label,
                }
                current = gate_node
                gate_modules[spec.identifier()] = gate_name
                gate_nodes[spec.identifier()] = gate_node.name
            for user in original_users:
                user.replace_input_with(target_node, current)

        graph_module.graph.lint()
        graph_module.recompile()
        return CompiledIntervention(
            module=graph_module,
            specs=tuple(specs),
            gate_modules=gate_modules,
            gate_nodes=gate_nodes,
            graph_code=graph_module.code,
        )


@dataclass(slots=True)
class InterventionResult:
    specs: tuple[InterventionSpec, ...]
    target: int
    baseline_margin: np.ndarray
    intervened_margin: np.ndarray
    effect: np.ndarray
    mean_effect: float
    median_effect: float
    ci_low: float
    ci_high: float
    sign_stability: float
    prediction_flip_rate: float
    baseline_prediction: np.ndarray
    intervened_prediction: np.ndarray
    baseline_logits: np.ndarray
    intervened_logits: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "interventions": [spec.label or spec.identifier() for spec in self.specs],
            "target": self.target,
            "samples": int(self.effect.size),
            "mean_effect": self.mean_effect,
            "median_effect": self.median_effect,
            "ci_95": [self.ci_low, self.ci_high],
            "sign_stability": self.sign_stability,
            "prediction_flip_rate": self.prediction_flip_rate,
        }


class InterventionExperiment:
    def __init__(
        self,
        captured: CapturedModel,
        *,
        output_path: str | Sequence[str | int] | None = None,
        bootstrap_repeats: int = 500,
        seed: int = 17,
    ) -> None:
        self.captured = captured
        self.output_path = output_path
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.seed = int(seed)
        self.compiler = GraphInterventionCompiler(captured)

    def run(
        self,
        specs: Sequence[InterventionSpec],
        args: Sequence[Any],
        *,
        kwargs: Mapping[str, Any] | None = None,
        target: int,
    ) -> InterventionResult:
        kwargs = dict(kwargs or {})
        compiled = self.compiler.compile(specs)
        with torch.no_grad():
            baseline_output = self.captured.graph_module(*tuple(args), **kwargs)  # type: ignore[misc]
            intervened_output = compiled.module(*tuple(args), **kwargs)
        baseline_logits = extract_logits(baseline_output, self.output_path)
        intervened_logits = extract_logits(intervened_output, self.output_path)
        baseline_margin = target_margin(baseline_logits, target)
        intervened_margin = target_margin(intervened_logits, target)
        effect = baseline_margin - intervened_margin
        effect_np = effect.detach().to("cpu", dtype=torch.float64).numpy()
        low, high = bootstrap_interval(
            effect_np,
            repeats=self.bootstrap_repeats,
            seed=self.seed,
        )
        baseline_pred = predicted_class(baseline_logits)
        intervened_pred = predicted_class(intervened_logits)
        median = float(np.median(effect_np)) if effect_np.size else 0.0
        if abs(median) < 1e-12:
            sign_stability = float(np.mean(np.abs(effect_np) < 1e-12)) if effect_np.size else 0.0
        else:
            sign_stability = float(np.mean(np.sign(effect_np) == np.sign(median)))
        return InterventionResult(
            specs=tuple(specs),
            target=int(target),
            baseline_margin=baseline_margin.detach().cpu().numpy(),
            intervened_margin=intervened_margin.detach().cpu().numpy(),
            effect=effect_np,
            mean_effect=float(np.mean(effect_np)) if effect_np.size else 0.0,
            median_effect=median,
            ci_low=low,
            ci_high=high,
            sign_stability=sign_stability,
            prediction_flip_rate=float(
                (baseline_pred != intervened_pred).to(torch.float32).mean().item()
            ),
            baseline_prediction=baseline_pred.detach().cpu().numpy(),
            intervened_prediction=intervened_pred.detach().cpu().numpy(),
            baseline_logits=baseline_logits.detach().cpu().numpy(),
            intervened_logits=intervened_logits.detach().cpu().numpy(),
            metadata={
                "graph_code_hash": hashlib.sha256(compiled.graph_code.encode("utf-8")).hexdigest(),
                "bootstrap_repeats": self.bootstrap_repeats,
                "seed": self.seed,
                "effect_definition": "baseline target margin minus intervened target margin",
            },
        )


def bootstrap_interval(
    values: np.ndarray,
    *,
    repeats: int = 500,
    confidence: float = 0.95,
    seed: int = 17,
    statistic: str = "median",
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size:
        return 0.0, 0.0
    if array.size == 1 or repeats <= 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(repeats, array.size))
    samples = array[indices]
    estimates = np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def capture_intervention_values(
    compiled: CompiledIntervention,
    args: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recorder = FXValueRecorder(
        compiled.module,
        selected_nodes=compiled.gate_nodes.values(),
        detach=True,
        move_to_cpu=True,
    )
    record = recorder.execute(tuple(args), dict(kwargs or {}))
    return record.values


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_") or "node"
