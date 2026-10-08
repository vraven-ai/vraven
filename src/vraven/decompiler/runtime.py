"""Runtime utilities shared by concept discovery, intervention and programme validation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Interpreter, Node
from torch.utils import _pytree as pytree


@dataclass(slots=True)
class ExecutionRecord:
    output: Any
    values: dict[str, Any]
    durations_ms: dict[str, float] = field(default_factory=dict)


class FXValueRecorder(Interpreter):
    """Execute an FX graph while preserving selected intermediate values.

    Values remain attached to autograd unless ``detach=True``.  This makes the recorder
    suitable for derivative screening as well as observational activation capture.
    """

    def __init__(
        self,
        module: GraphModule,
        *,
        selected_nodes: Iterable[str] | None = None,
        detach: bool = False,
        move_to_cpu: bool = False,
        profile: bool = False,
    ) -> None:
        super().__init__(module, garbage_collect_values=False)
        self.selected_nodes = set(selected_nodes or ())
        self.capture_all = selected_nodes is None
        self.detach = detach
        self.move_to_cpu = move_to_cpu
        self.profile = profile
        self.values: dict[str, Any] = {}
        self.durations_ms: dict[str, float] = {}

    def run_node(self, node: Node) -> Any:
        start = time.perf_counter() if self.profile else 0.0
        result = super().run_node(node)
        if self.profile:
            _synchronise_value(result)
            self.durations_ms[node.name] = (time.perf_counter() - start) * 1000.0
        if self.capture_all or node.name in self.selected_nodes:
            self.values[node.name] = _prepare_value(
                result,
                detach=self.detach,
                move_to_cpu=self.move_to_cpu,
            )
        return result

    def execute(self, args: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> ExecutionRecord:
        ordered = bind_graph_inputs(self.module, tuple(args), dict(kwargs or {}))
        output = self.run(*ordered)
        return ExecutionRecord(
            output=output,
            values=dict(self.values),
            durations_ms=dict(self.durations_ms),
        )


def bind_graph_inputs(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Bind user arguments to a GraphModule's forward signature in parameter order."""

    if not kwargs:
        return args
    signature = inspect.signature(module.forward)
    bound = signature.bind_partial(*args, **dict(kwargs))
    bound.apply_defaults()
    ordered: list[Any] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError("VRAVEN graph execution does not support variadic GraphModule inputs")
        if name not in bound.arguments:
            raise TypeError(f"No value supplied for GraphModule input {name!r}")
        ordered.append(bound.arguments[name])
    return tuple(ordered)


def extract_logits(output: Any, output_path: str | Sequence[str | int] | None = None) -> Tensor:
    """Extract a logits tensor from common model-output structures.

    No heuristic silently selects among several tensor outputs.  When more than one
    plausible tensor exists, callers must provide ``output_path``.
    """

    if output_path is not None:
        current = output
        path = output_path.split(".") if isinstance(output_path, str) else list(output_path)
        for key in path:
            if isinstance(current, Mapping) or isinstance(key, int):
                current = current[key]
            elif hasattr(current, str(key)):
                current = getattr(current, str(key))
            else:
                current = current[int(key)]
        if not isinstance(current, Tensor):
            raise TypeError("The selected output_path does not resolve to a torch.Tensor")
        return current

    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits") and isinstance(output.logits, Tensor):
        return output.logits
    if isinstance(output, Mapping):
        for preferred in ("logits", "scores", "output"):
            value = output.get(preferred)
            if isinstance(value, Tensor):
                return value
    tensors, _ = pytree.tree_flatten(output)
    candidates = [value for value in tensors if isinstance(value, Tensor)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Model output contains multiple tensors. Supply output_path so VRAVEN does not "
        "guess which tensor contains prediction scores."
    )


def target_margin(logits: Tensor, target: int | Tensor) -> Tensor:
    """Return target-versus-alternatives margin for each sample.

    For multiclass logits this is ``z_t - logsumexp(z_not_t)``.  For a single binary
    logit, class 1 uses ``z`` and class 0 uses ``-z``.
    """

    if logits.ndim == 0:
        logits = logits.reshape(1, 1)
    elif logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shaped [batch, classes], received {tuple(logits.shape)}")
    batch, classes = logits.shape
    if isinstance(target, int):
        target_tensor = torch.full((batch,), int(target), device=logits.device, dtype=torch.long)
    else:
        target_tensor = target.to(device=logits.device, dtype=torch.long).reshape(-1)
        if target_tensor.numel() == 1 and batch > 1:
            target_tensor = target_tensor.expand(batch)
        if target_tensor.numel() != batch:
            raise ValueError("target tensor length must equal the batch size")
    if classes == 1:
        signed = torch.where(target_tensor == 1, logits[:, 0], -logits[:, 0])
        return signed
    if torch.any((target_tensor < 0) | (target_tensor >= classes)):
        raise ValueError(f"target must be between 0 and {classes - 1}")
    target_score = logits.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target_tensor.unsqueeze(1), False)
    alternative = torch.logsumexp(logits.masked_fill(~mask, float("-inf")), dim=1)
    return target_score - alternative


def predicted_class(logits: Tensor) -> Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape[-1] == 1:
        return (logits[:, 0] >= 0).to(torch.long)
    return logits.argmax(dim=-1)


def infer_feature_axis(value: Tensor, hint: str = "auto") -> int:
    """Infer the axis whose coordinates form candidate concepts.

    ``channel`` uses axis 1 for convolutional outputs.  ``last`` uses the final axis for
    token, recurrent and linear representations.  Automatic mode favours axis 1 for
    common channel-first tensors and the final axis for two-dimensional tensors.
    """

    if value.ndim < 2:
        raise ValueError("A concept-bearing activation must include batch and feature dimensions")
    if hint == "channel":
        return 1
    if hint == "last":
        return value.ndim - 1
    if value.ndim == 2:
        return 1
    # Channel-first convolutional activations usually have a relatively small axis 1.
    if value.ndim in (3, 4, 5) and value.shape[1] <= max(4096, value.shape[-1]):
        return 1
    return value.ndim - 1


def activation_matrix(value: Tensor, feature_axis: int) -> Tensor:
    """Reduce an activation to a batch-by-feature matrix without mixing features."""

    if value.ndim < 2:
        raise ValueError("Activation must have at least two dimensions")
    axis = feature_axis if feature_axis >= 0 else value.ndim + feature_axis
    if axis == 0 or axis >= value.ndim:
        raise ValueError("feature_axis must identify a non-batch dimension")
    moved = value.movedim(axis, 1)
    if moved.ndim == 2:
        return moved
    return moved.flatten(start_dim=2).mean(dim=2)


def concept_coefficients(value: Tensor, direction: Tensor, feature_axis: int) -> Tensor:
    matrix = activation_matrix(value, feature_axis)
    direction = _normalise_direction(direction, matrix.shape[1], matrix.device, matrix.dtype)
    return matrix @ direction


def erase_concept(value: Tensor, direction: Tensor, feature_axis: int, strength: float = 1.0) -> Tensor:
    """Remove a concept direction while preserving the orthogonal residual."""

    axis = feature_axis if feature_axis >= 0 else value.ndim + feature_axis
    moved = value.movedim(axis, -1)
    direction = _normalise_direction(direction, moved.shape[-1], moved.device, moved.dtype)
    coefficient = torch.einsum("...f,f->...", moved, direction)
    erased = moved - float(strength) * coefficient.unsqueeze(-1) * direction
    return erased.movedim(-1, axis)


def set_concept_mean(
    value: Tensor,
    direction: Tensor,
    feature_axis: int,
    coefficient: float | Tensor,
) -> Tensor:
    """Shift a concept so its batch item mean coefficient equals ``coefficient``.

    Unlike setting every spatial or token position to the same value, this operation
    preserves local variation and only shifts the mean concept activation.
    """

    axis = feature_axis if feature_axis >= 0 else value.ndim + feature_axis
    moved = value.movedim(axis, -1)
    direction = _normalise_direction(direction, moved.shape[-1], moved.device, moved.dtype)
    local = torch.einsum("...f,f->...", moved, direction)
    current_mean = local if local.ndim == 1 else local.flatten(start_dim=1).mean(dim=1)
    desired = torch.as_tensor(coefficient, device=moved.device, dtype=moved.dtype)
    if desired.ndim == 0:
        desired = desired.expand_as(current_mean)
    elif desired.numel() == 1:
        desired = desired.reshape(1).expand_as(current_mean)
    elif desired.reshape(-1).numel() != current_mean.numel():
        raise ValueError("Concept mean replacement must be scalar or have one value per batch item")
    else:
        desired = desired.reshape_as(current_mean)
    delta = desired - current_mean
    while delta.ndim < moved.ndim - 1:
        delta = delta.unsqueeze(-1)
    adjusted = moved + delta.unsqueeze(-1) * direction
    return adjusted.movedim(-1, axis)


def set_concept_coefficient(
    value: Tensor,
    direction: Tensor,
    feature_axis: int,
    coefficient: float | Tensor,
) -> Tensor:
    """Set a concept coefficient and retain all orthogonal activation information."""

    axis = feature_axis if feature_axis >= 0 else value.ndim + feature_axis
    moved = value.movedim(axis, -1)
    direction = _normalise_direction(direction, moved.shape[-1], moved.device, moved.dtype)
    current = torch.einsum("...f,f->...", moved, direction)
    desired = torch.as_tensor(coefficient, device=moved.device, dtype=moved.dtype)
    while desired.ndim < current.ndim:
        desired = desired.unsqueeze(-1)
    adjusted = moved + (desired - current).unsqueeze(-1) * direction
    return adjusted.movedim(-1, axis)


def _normalise_direction(direction: Tensor, features: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    result = direction.to(device=device, dtype=dtype).reshape(-1)
    if result.numel() != features:
        raise ValueError(
            f"Concept direction has {result.numel()} elements, but activation axis has {features}"
        )
    return result / result.norm().clamp_min(torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12)


def _prepare_value(value: Any, *, detach: bool, move_to_cpu: bool) -> Any:
    def transform(item: Any) -> Any:
        if not isinstance(item, Tensor):
            return item
        result = item.detach() if detach else item
        if move_to_cpu:
            result = result.to("cpu")
        return result

    return pytree.tree_map(transform, value)


def _synchronise_value(value: Any) -> None:
    tensors, _ = pytree.tree_flatten(value)
    devices = {item.device.type for item in tensors if isinstance(item, Tensor)}
    if "cuda" in devices and torch.cuda.is_available():
        torch.cuda.synchronize()
    if "mps" in devices and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
