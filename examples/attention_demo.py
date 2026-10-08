"""Capture attention weights from a model that explicitly returns them."""

from pathlib import Path

import torch

import vraven


class AttentionClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(16, 4, batch_first=True)
        self.classifier = torch.nn.Linear(16, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _weights = self.attention(
            inputs,
            inputs,
            inputs,
            need_weights=True,
            average_attn_weights=False,
        )
        return self.classifier(encoded.mean(dim=1))


def main() -> None:
    torch.manual_seed(17)
    model = AttentionClassifier()
    samples = torch.randn(20, 32, 16)
    report = vraven.explain(model, samples, target=1, mode="gradient")
    output = Path("vraven_attention_demo")
    output.mkdir(exist_ok=True)
    report.attention_structure(head=2, mode="interactive_dark").write_html(
        output / "attention_head_2.html", include_plotlyjs=True
    )


if __name__ == "__main__":
    main()
