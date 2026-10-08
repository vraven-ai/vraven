import torch

from vraven.metrics.basic import activation_statistics


def test_activation_statistics() -> None:
    x = torch.tensor([1.0, -1.0, 0.0, 2.0])
    stats = activation_statistics(x)
    assert stats["mean"] == 0.5
    assert stats["energy"] == 6.0
    assert stats["sparsity"] == 0.25
