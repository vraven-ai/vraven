from __future__ import annotations

import torch

import vraven


class AttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.output = torch.nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _weights = self.attention(
            inputs,
            inputs,
            inputs,
            need_weights=True,
            average_attn_weights=False,
        )
        return self.output(encoded.mean(dim=1))


def test_attention_weights_are_captured_when_model_exposes_them() -> None:
    torch.manual_seed(2)
    report = vraven.explain(
        AttentionModel(),
        torch.randn(4, 6, 8),
        target=1,
        mode="gradient",
    )
    attention_layers = [layer for layer in report.trace.layers if layer.attention is not None]
    assert len(attention_layers) == 1
    assert attention_layers[0].attention_shape == (2, 6, 6)
    figure = report.attention_structure(head=1, mode="publication_light")
    assert figure.layout.paper_bgcolor == "#FFFFFF"
