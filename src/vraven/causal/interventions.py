"""Temporary, reversible PyTorch layer interventions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from vraven.core.errors import CapabilityError


def resolve_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    try:
        return modules[module_name]
    except KeyError as error:
        raise CapabilityError(f"Model does not contain a module named {module_name!r}") from error


def _intervene_tensor(
    tensor: torch.Tensor,
    *,
    strategy: str,
    scale: float,
) -> torch.Tensor:
    if strategy == "zero":
        return torch.zeros_like(tensor)
    if strategy == "mean":
        return torch.full_like(tensor, tensor.detach().mean())
    if strategy == "scale":
        return tensor * scale
    raise ValueError(f"Unsupported intervention strategy: {strategy}")


def intervene_output(value: Any, *, strategy: str, scale: float = 0.0) -> Any:
    if isinstance(value, torch.Tensor):
        return _intervene_tensor(value, strategy=strategy, scale=scale)
    if isinstance(value, tuple):
        return tuple(intervene_output(item, strategy=strategy, scale=scale) for item in value)
    if isinstance(value, list):
        return [intervene_output(item, strategy=strategy, scale=scale) for item in value]
    if isinstance(value, Mapping):
        return type(value)(
            (key, intervene_output(item, strategy=strategy, scale=scale))
            for key, item in value.items()
        )
    return value


@contextmanager
def temporary_output_intervention(
    model: torch.nn.Module,
    module_name: str,
    *,
    strategy: str = "zero",
    scale: float = 0.0,
) -> Iterator[None]:
    """Modify one module output for the duration of a controlled inference."""

    module = resolve_module(model, module_name)

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        return intervene_output(output, strategy=strategy, scale=scale)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
