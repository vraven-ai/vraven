"""Adapter discovery for public VRAVEN APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vraven.adapters.base import ModelAdapter
from vraven.adapters.pytorch import PyTorchAdapter
from vraven.core.config import VRavenConfig
from vraven.core.errors import UnsupportedModelError


def resolve_adapter(
    model: Any,
    config: VRavenConfig,
    *,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
) -> ModelAdapter:
    if isinstance(model, torch.nn.Module):
        return PyTorchAdapter(model, config, class_names=class_names)
    raise UnsupportedModelError(
        "No VRAVEN adapter is installed for this model type. The complete alpha release "
        "supports live torch.nn.Module objects. ONNX and TensorFlow adapters remain planned."
    )
