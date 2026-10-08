"""Nested model-input / output helpers used by DarkSide live interventions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from vraven.core.device import iter_tensors, normalise_model_inputs


def call_model(model: torch.nn.Module, inputs: Any) -> Any:
    """Invoke ``model`` with the public VRAVEN input forms."""

    args, kwargs = normalise_model_inputs(inputs)
    return model(*args, **kwargs)


def extract_tensors(value: Any) -> list[torch.Tensor]:
    """Collect tensors from a nested input or module output."""

    return list(iter_tensors(value))


def replace_first_tensor(
    value: Any,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Any:
    """Replace the first tensor found in a nested structure."""

    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, Mapping):
        updated = dict(value)
        for key, item in updated.items():
            if isinstance(item, torch.Tensor):
                updated[key] = transform(item)
                return type(value)(updated) if not isinstance(value, dict) else updated
            replaced = replace_first_tensor(item, transform)
            if replaced is not item:
                updated[key] = replaced
                return type(value)(updated) if not isinstance(value, dict) else updated
        return value
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                items[index] = transform(item)
                return tuple(items)
            replaced = replace_first_tensor(item, transform)
            if replaced is not item:
                items[index] = replaced
                return tuple(items)
        return value
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                items[index] = transform(item)
                return items
            replaced = replace_first_tensor(item, transform)
            if replaced is not item:
                items[index] = replaced
                return items
        return value
    return value
