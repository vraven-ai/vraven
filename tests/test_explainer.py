import torch

import vraven


def test_vraven_captures_layers() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    x = torch.randn(1, 4)
    report = vraven.explain(model, x)
    summary = report.summary()
    assert summary["layers_captured"] == 3
    assert summary["framework"] == "pytorch"
