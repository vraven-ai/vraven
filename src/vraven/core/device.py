"""Device and nested-input utilities.

VRAVEN never moves a user's model. Inputs may be copied to the model's existing
backend when ``auto_move_inputs`` is enabled.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch


def infer_model_device(model: torch.nn.Module) -> torch.device:
    """Return the current model device without modifying the model."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def infer_backend(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        return "cuda"
    if resolved.type == "mps":
        return "mps"
    if resolved.type in {"xpu", "hip"}:
        return resolved.type
    return "cpu"


def move_nested(value: Any, device: torch.device) -> Any:
    """Move tensors in a nested input structure while preserving containers."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)((key, move_nested(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    return value


def clone_for_gradient(value: Any) -> Any:
    """Clone floating tensors and enable input gradients without altering callers."""

    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, Mapping):
        return type(value)((key, clone_for_gradient(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(clone_for_gradient(item) for item in value)
    if isinstance(value, list):
        return [clone_for_gradient(item) for item in value]
    return value


def detach_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return type(value)((key, detach_nested(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(detach_nested(item) for item in value)
    if isinstance(value, list):
        return [detach_nested(item) for item in value]
    return value


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensors from any supported nested input structure."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_tensors(item)


def floating_tensors(value: Any) -> list[torch.Tensor]:
    return [tensor for tensor in iter_tensors(value) if tensor.requires_grad]


def normalise_model_inputs(inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Convert the public input form into positional and keyword arguments.

    A mapping is treated as keyword arguments, a tuple as positional arguments,
    and every other object as one positional argument.
    """

    if isinstance(inputs, Mapping):
        return (), dict(inputs)
    if isinstance(inputs, tuple):
        return inputs, {}
    return (inputs,), {}


def prepare_model_inputs(
    inputs: Any,
    *,
    device: torch.device,
    move: bool,
    require_grad: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args, kwargs = normalise_model_inputs(inputs)
    if move:
        args = move_nested(args, device)
        kwargs = move_nested(kwargs, device)
    if require_grad:
        args = clone_for_gradient(args)
        kwargs = clone_for_gradient(kwargs)
    return args, kwargs


def synchronise_device(device: torch.device) -> None:
    """Synchronise only at whole-run boundaries, never for every layer."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
