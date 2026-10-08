from __future__ import annotations

import torch

from vraven.causal import generate_counterfactual
from vraven.visualisation.counterfactual_delta import counterfactual_delta_figure


def test_continuous_counterfactual_can_flip_simple_linear_classifier() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.0], [-2.0, 0.0]]))
    result = generate_counterfactual(
        model,
        torch.tensor([[1.0, 0.0]]),
        target_class=1,
        steps=80,
        learning_rate=0.25,
        l2_weight=0.0,
    )
    assert result.success is True
    assert result.counterfactual_prediction == 1
    assert all(parameter.grad is None for parameter in model.parameters())
    figure = counterfactual_delta_figure(result, mode="publication_light")
    assert len(figure.data) == 3
