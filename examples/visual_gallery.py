"""Create one standalone HTML file for every compatible single-report figure."""

from pathlib import Path

import numpy as np
import torch

import vraven


def main() -> None:
    torch.manual_seed(44)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 24),
        torch.nn.Tanh(),
        torch.nn.Linear(24, 12),
        torch.nn.GELU(),
        torch.nn.Linear(12, 3),
    )
    inputs = torch.randn(24, 10)
    report = vraven.explain(model, inputs, target=1, mode="causal", causal_top_k=3)
    output = Path("vraven_visual_gallery")
    output.mkdir(exist_ok=True)

    for visual, supported in report.capabilities().items():
        if not supported:
            continue
        for render_mode in ("interactive_dark", "publication_light"):
            figure = report.figure(visual, mode=render_mode)
            figure.write_html(
                output / f"{visual}_{render_mode}.html",
                include_plotlyjs=True,
            )

    methods = {
        "gradient": np.asarray(report.trace.inputs.attribution_values, dtype=float),
        "scaled_gradient": 0.9 * np.asarray(report.trace.inputs.attribution_values, dtype=float),
    }
    report.consistency_matrix(methods, mode="publication_light").write_html(
        output / "consistency_matrix_publication_light.html",
        include_plotlyjs=True,
    )


if __name__ == "__main__":
    main()
