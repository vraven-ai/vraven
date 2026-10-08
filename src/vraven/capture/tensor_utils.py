"""On-device tensor reduction utilities used by capture adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as functional


def extract_tensor(value: Any) -> torch.Tensor | None:
    """Return the first tensor in a nested value without changing it."""

    if isinstance(value, torch.Tensor):
        return value
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _batch_flatten(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(value.shape[0], 1)
    return value.reshape(value.shape[0], -1)


def compressed_representation(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Produce a deterministic fixed-width representation for each batch item."""

    flat = _batch_flatten(tensor)
    width = flat.shape[1]
    if width == size:
        return flat
    if width > size:
        return functional.adaptive_avg_pool1d(flat.unsqueeze(1), size).squeeze(1)
    return functional.pad(flat, (0, size - width))


def infer_unit_axis(tensor: torch.Tensor, module: torch.nn.Module | None = None) -> int:
    """Infer the channel or feature axis for common deep-learning modules."""

    if tensor.ndim <= 1:
        return 0
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        return 1
    semantic_last_axis = (
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.LayerNorm,
        torch.nn.MultiheadAttention,
        torch.nn.RNN,
        torch.nn.GRU,
        torch.nn.LSTM,
    )
    if isinstance(module, semantic_last_axis):
        return tensor.ndim - 1
    if tensor.ndim == 4:
        return 1
    return tensor.ndim - 1


def unit_matrix(
    tensor: torch.Tensor,
    *,
    module: torch.nn.Module | None,
    max_units: int,
) -> torch.Tensor:
    """Reduce a tensor to ``[batch, units]`` while retaining unit direction."""

    value = tensor.detach().float()
    if value.ndim == 0:
        matrix = value.reshape(1, 1)
    elif value.ndim == 1:
        matrix = value.reshape(value.shape[0], 1)
    else:
        axis = infer_unit_axis(value, module)
        if axis < 0:
            axis += value.ndim
        if axis == 0:
            matrix = value.reshape(value.shape[0], -1).mean(dim=1, keepdim=True)
        else:
            remaining = [dimension for dimension in range(value.ndim) if dimension not in {0, axis}]
            permutation = [0, axis, *remaining]
            reordered = value.permute(permutation)
            if remaining:
                matrix = reordered.reshape(reordered.shape[0], reordered.shape[1], -1).mean(dim=2)
            else:
                matrix = reordered
    if matrix.shape[1] > max_units:
        matrix = functional.adaptive_avg_pool1d(matrix.unsqueeze(1), max_units).squeeze(1)
    return matrix


def _attention_candidates(value: Any) -> list[torch.Tensor]:
    candidates: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        if value.ndim in {3, 4} and value.shape[-1] == value.shape[-2]:
            candidates.append(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if "att" in str(key).lower():
                candidates.extend(_attention_candidates(item))
    elif isinstance(value, (tuple, list)):
        for item in value[1:]:
            candidates.extend(_attention_candidates(item))
    else:
        attentions = getattr(value, "attentions", None)
        if attentions is not None:
            candidates.extend(_attention_candidates(attentions))
    return candidates


def compressed_attention(value: Any, max_tokens: int) -> torch.Tensor | None:
    """Return mean-batch attention with shape ``[heads, query, key]``."""

    candidates = _attention_candidates(value)
    if not candidates:
        return None
    attention = candidates[0].detach().float()
    if attention.ndim == 4:
        attention = attention.mean(dim=0)
    elif attention.ndim == 3:
        # Common PyTorch shape is [batch, query, key] when heads are averaged.
        attention = attention.mean(dim=0, keepdim=True)
    if attention.shape[-2] > max_tokens or attention.shape[-1] > max_tokens:
        attention = functional.adaptive_avg_pool2d(
            attention.unsqueeze(0),
            output_size=(min(max_tokens, attention.shape[-2]), min(max_tokens, attention.shape[-1])),
        ).squeeze(0)
    return attention


def compress_flat_values(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    """Deterministically sample a flat tensor for serialisable input views."""

    flat = tensor.detach().float().reshape(-1)
    if flat.numel() <= limit:
        return flat
    positions = torch.linspace(0, flat.numel() - 1, steps=limit, device=flat.device).long()
    return flat.index_select(0, positions)
