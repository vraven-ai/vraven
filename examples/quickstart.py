"""Minimal live-model integration with gradient explanations."""

from pathlib import Path

import torch

import vraven


def main() -> None:
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 12),
        torch.nn.ReLU(),
        torch.nn.Linear(12, 3),
    )
    samples = torch.randn(32, 16)
    report = vraven.explain(
        model,
        samples,
        target=1,
        mode="gradient",
        class_names=["benign", "malware", "unknown"],
    )

    output = Path("vraven_quickstart")
    output.mkdir(exist_ok=True)
    report.save_trace(output / "trace.json")
    report.decision_flow(mode="interactive_dark").write_html(
        output / "decision_flow_dark.html", include_plotlyjs=True
    )
    report.reasoning_dna(mode="publication_light").write_html(
        output / "reasoning_dna_publication.html", include_plotlyjs=True
    )
    report.latent_flight(mode="publication_light").write_html(
        output / "latent_flight_publication.html", include_plotlyjs=True
    )
    print(report.summary())


if __name__ == "__main__":
    main()
