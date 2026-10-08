"""Reproducibility metadata helpers."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Iterable
from typing import Any

import torch


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_structure_hash(model: torch.nn.Module) -> str:
    """Hash structure and tensor metadata without copying all weights to CPU."""

    digest = hashlib.sha256()
    digest.update(model.__class__.__qualname__.encode("utf-8"))
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def stable_value_hash(values: Iterable[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(float(value).hex().encode("ascii"))
    return digest.hexdigest()


def tensor_input_fingerprint(
    tensors: Iterable[torch.Tensor],
    *,
    values_per_tensor: int = 64,
) -> str:
    """Return a bounded reproducibility fingerprint for model inputs.

    The fingerprint hashes tensor shape, dtype and deterministically sampled
    values. It deliberately avoids copying complete inputs to CPU. It is a
    reproducibility identifier, not a secure commitment to every input byte.
    """

    if values_per_tensor <= 0:
        raise ValueError("values_per_tensor must be greater than zero")
    digest = hashlib.sha256()
    seen = 0
    for tensor in tensors:
        seen += 1
        value = tensor.detach()
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.device.type).encode("utf-8"))
        flat = value.reshape(-1)
        if flat.numel():
            count = min(values_per_tensor, flat.numel())
            positions = torch.linspace(
                0, flat.numel() - 1, steps=count, device=flat.device
            ).long()
            sampled = flat.index_select(0, positions).detach().to("cpu")
            if sampled.dtype == torch.bfloat16:
                sampled = sampled.float()
            digest.update(sampled.contiguous().numpy().tobytes())
    digest.update(str(seen).encode("ascii"))
    return digest.hexdigest()
