from __future__ import annotations

import pytest
import torch
from torch import nn


class TinyMechanismNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detect = nn.Linear(6, 5, bias=False)
        self.gate = nn.ReLU()
        self.readout = nn.Linear(5, 2, bias=False)
        with torch.no_grad():
            self.detect.weight.zero_()
            self.detect.weight[0, 0] = 1.0
            self.detect.weight[0, 1] = 0.8
            self.detect.weight[1, 2] = 1.1
            self.detect.weight[1, 3] = -0.7
            self.detect.weight[2, 0] = 0.5
            self.detect.weight[2, 1] = 0.4
            self.detect.weight[3, 4] = 1.0
            self.detect.weight[4, 5] = 1.0
            self.readout.weight.zero_()
            margin = torch.tensor([2.0, 1.5, 0.6, 0.0, -1.0])
            self.readout.weight[1].copy_(margin / 2.0)
            self.readout.weight[0].copy_(-margin / 2.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.readout(self.gate(self.detect(value)))


@pytest.fixture()
def tiny_model() -> TinyMechanismNet:
    return TinyMechanismNet().eval()


@pytest.fixture()
def tiny_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    value = torch.randn(40, 6, generator=generator)
    margin = 2.0 * torch.relu(value[:, 0] + 0.8 * value[:, 1])
    margin += 1.5 * torch.relu(1.1 * value[:, 2] - 0.7 * value[:, 3])
    margin += 0.6 * torch.relu(0.5 * value[:, 0] + 0.4 * value[:, 1])
    margin -= torch.relu(value[:, 5])
    return value, (margin > 0).long()
