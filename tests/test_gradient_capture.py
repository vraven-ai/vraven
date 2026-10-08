from __future__ import annotations

import torch

import vraven


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(5, 9),
        torch.nn.ReLU(),
        torch.nn.Linear(9, 3),
    )


def test_gradient_mode_captures_target_specific_signals_without_parameter_grads() -> None:
    torch.manual_seed(3)
    model = _model()
    model.train()
    inputs = torch.randn(6, 5)
    assert all(parameter.grad is None for parameter in model.parameters())

    report = vraven.explain(model, inputs, target=1, mode="gradient")

    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(layer.gradient_norm is not None for layer in report.trace.layers)
    assert all(layer.activation_gradient_influence is not None for layer in report.trace.layers)
    assert report.trace.inputs is not None
    assert report.trace.inputs.attribution_available is True
    assert report.capabilities()["decision_flow"] is True


def test_repeated_capture_removes_hooks() -> None:
    model = _model()
    inputs = torch.randn(2, 5)
    modules = list(model.modules())
    before = [len(module._forward_hooks) for module in modules]
    vraven.explain(model, inputs, mode="fast")
    vraven.explain(model, inputs, target=0, mode="gradient")
    after = [len(module._forward_hooks) for module in modules]
    assert after == before
