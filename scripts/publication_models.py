"""Compact neural architectures for the VRAVEN cross-domain publication suite."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.net(inputs))


class PublicationResidualMLP(nn.Module):
    """Small ReLU residual MLP suitable for tabular certification studies."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        blocks: int = 2,
        dropout: float = 0.05,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(hidden_dim, dropout) for _ in range(blocks))
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(inputs)))


class PublicationAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        normalised = self.norm1(inputs)
        attended, _ = self.attention(
            normalised,
            normalised,
            normalised,
            need_weights=False,
        )
        residual = inputs + attended
        return residual + self.feed_forward(self.norm2(residual))


class PublicationFeatureTransformer(nn.Module):
    """Transformer over contiguous feature chunks, avoiding O(d²) feature attention."""

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.token_count = math.ceil(input_dim / chunk_size)
        self.padded_dim = self.token_count * chunk_size
        self.projection = nn.Linear(chunk_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.token_count, d_model))
        self.blocks = nn.ModuleList(
            PublicationAttentionBlock(d_model, nhead, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.padded_dim != self.input_dim:
            inputs = nn.functional.pad(inputs, (0, self.padded_dim - self.input_dim))
        tokens = inputs.reshape(inputs.shape[0], self.token_count, self.chunk_size)
        tokens = self.projection(tokens) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens).mean(dim=1))


class PublicationSequenceCNN(nn.Module):
    """1-D convolutional classifier for inputs shaped [batch, time, channels]."""

    def __init__(
        self,
        input_channels: int,
        channels: tuple[int, ...] = (32, 64),
        num_classes: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_channels
        for width in channels:
            layers.extend(
                (
                    nn.Conv1d(current, width, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2, ceil_mode=True),
                )
            )
            current = width
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(current, num_classes))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.backbone(inputs.transpose(1, 2))
        return self.head(self.pool(features))


class PublicationSequenceTransformer(nn.Module):
    """Compact Transformer for multichannel temporal inputs."""

    def __init__(
        self,
        seq_len: int,
        input_channels: int,
        d_model: int = 48,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.projection = nn.Linear(input_channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList(
            PublicationAttentionBlock(d_model, nhead, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self.projection(inputs) + self.position[:, : inputs.shape[1]]
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens).mean(dim=1))


def build_publication_model(kind: str, **kwargs) -> nn.Module:
    builders = {
        "publication_residual_mlp": PublicationResidualMLP,
        "publication_feature_transformer": PublicationFeatureTransformer,
        "publication_sequence_cnn": PublicationSequenceCNN,
        "publication_sequence_transformer": PublicationSequenceTransformer,
    }
    if kind not in builders:
        raise ValueError(
            f"Unknown publication model kind {kind!r}; expected one of {sorted(builders)}"
        )
    return builders[kind](**kwargs)

