"""Tabular malware classifiers for Kronodroid (MLP, CNN1D, ResNet1D, BiGRU)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MalwareMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128, 64), num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for size in hidden:
            layers.extend((nn.Linear(current, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(0.2)))
            current = size
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MalwareCNN1D(nn.Module):
    """Treat the feature vector as a length-``input_dim`` 1-channel sequence."""

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (32, 64, 128), num_classes: int = 2):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            blocks.extend(
                (
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, ceil_mode=True),
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        with torch.no_grad():
            probe = self.backbone(torch.zeros(1, 1, input_dim))
            flat = probe.numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x.unsqueeze(1)))


class _ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class MalwareResNet1D(nn.Module):
    def __init__(self, input_dim: int, width: int = 64, blocks: int = 3, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[_ResBlock1D(width) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, num_classes)
        self._input_dim = input_dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.body(self.stem(x.unsqueeze(1)))
        h = self.pool(h).squeeze(-1)
        return self.head(h)


class MalwareBiGRU(nn.Module):
    """Fourth architecture: bidirectional GRU over feature patches."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        hidden: int = 64,
        layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pad_to = ((input_dim + patch_size - 1) // patch_size) * patch_size
        self.gru = nn.GRU(
            input_size=patch_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.pad_to:
            x = torch.nn.functional.pad(x, (0, self.pad_to - x.shape[-1]))
        seq = x.view(x.shape[0], -1, self.patch_size)
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])


MODEL_BUILDERS = {
    "mlp": lambda dim: MalwareMLP(dim),
    "cnn1d": lambda dim: MalwareCNN1D(dim),
    "resnet1d": lambda dim: MalwareResNet1D(dim),
    "bigru": lambda dim: MalwareBiGRU(dim),
}
