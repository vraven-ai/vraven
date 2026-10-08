"""LRP redistribution rules for supported PyTorch modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vraven.core.errors import CapabilityError


def _stabilize(values: Tensor, epsilon: float) -> Tensor:
    return values + epsilon * torch.where(
        values >= 0,
        torch.ones_like(values),
        -torch.ones_like(values),
    )


def lrp_linear(module: nn.Linear, activation: Tensor, relevance: Tensor, *, epsilon: float) -> Tensor:
    """Epsilon-rule LRP for a dense linear layer."""

    weight = module.weight.detach()
    bias = module.bias.detach() if module.bias is not None else None
    z = activation @ weight.T
    if bias is not None:
        z = z + bias
    z = _stabilize(z, epsilon)
    s = relevance / z
    return (s @ weight) * activation


def lrp_relu(_module: nn.ReLU, activation: Tensor, relevance: Tensor, *, epsilon: float) -> Tensor:
    """Pass relevance through ReLU; inactive units receive zero relevance."""

    del epsilon
    mask = (activation > 0).to(dtype=relevance.dtype)
    return relevance * mask


def lrp_embedding(
    module: nn.Embedding,
    indices: Tensor,
    relevance: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Redistribute embedding-output relevance onto selected rows (summed)."""

    del epsilon
    if relevance.ndim != 2 or relevance.shape[-1] != module.embedding_dim:
        raise CapabilityError("Embedding relevance must have shape [batch, embedding_dim]")
    if indices.ndim != 1 or indices.shape[0] != relevance.shape[0]:
        raise CapabilityError("Embedding indices must be a 1D batch of token/binary ids")
    table = torch.zeros(
        (module.num_embeddings, module.embedding_dim),
        dtype=relevance.dtype,
        device=relevance.device,
    )
    table.index_add_(0, indices.long(), relevance)
    return table


SUPPORTED_RULES: dict[type[nn.Module], str] = {
    nn.Linear: "epsilon",
    nn.ReLU: "relu_mask",
    nn.Embedding: "embedding_index_add",
}


def is_supported_module(module: nn.Module) -> bool:
    return type(module) in SUPPORTED_RULES
