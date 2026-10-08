from __future__ import annotations

import pytest
import torch

import vraven


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_cuda_smoke() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).cuda()
    report = vraven.explain(model, torch.randn(3, 4, device="cuda"), target=1, mode="gradient")
    assert report.trace.metadata["backend"] == "cuda"
    assert all(layer.device.startswith("cuda") for layer in report.trace.layers)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Apple MPS is unavailable in this environment",
)
def test_mps_smoke() -> None:
    device = torch.device("mps")
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(device)
    report = vraven.explain(model, torch.randn(3, 4, device=device), target=1, mode="gradient")
    assert report.trace.metadata["backend"] == "mps"
    assert all(layer.device.startswith("mps") for layer in report.trace.layers)
