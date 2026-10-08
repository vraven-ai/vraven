"""Render identical VRAVEN measurements in both official visual modes."""

from pathlib import Path

import torch

import vraven


def main() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 4),
    )
    report = vraven.explain(model, torch.randn(24, 8), target=2, mode="gradient")
    output = Path("vraven_visual_modes")
    output.mkdir(exist_ok=True)

    for visual in ("decision_flow", "spectral_trace", "reasoning_dna", "conflict_field"):
        report.figure(visual, mode="interactive_dark").write_html(
            output / f"{visual}_dark.html", include_plotlyjs=True
        )
        report.figure(visual, mode="publication_light").write_html(
            output / f"{visual}_publication.html", include_plotlyjs=True
        )


if __name__ == "__main__":
    main()
