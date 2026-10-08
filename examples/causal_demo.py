"""Run limited temporary output interventions on candidate layers."""

from pathlib import Path

import torch

import vraven


def main() -> None:
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 12),
        torch.nn.GELU(),
        torch.nn.Linear(12, 2),
    )
    samples = torch.randn(16, 10)
    report = vraven.explain(
        model,
        samples,
        target=1,
        mode="causal",
        causal_top_k=3,
        causal_strategy="zero",
    )
    output = Path("vraven_causal_demo")
    output.mkdir(exist_ok=True)
    report.causal_pathway(mode="publication_light").write_html(
        output / "causal_pathway.html", include_plotlyjs=True
    )
    report.export_html(output / "full_local_report.html")


if __name__ == "__main__":
    main()
