"""Standalone figure export with journal-size presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

from vraven.core.errors import ExportError
from vraven.visualisation.palette import RenderMode


@dataclass(frozen=True, slots=True)
class PublicationPreset:
    name: str
    width_mm: float
    height_mm: float

    def pixels(self, dpi: int = 300) -> tuple[int, int]:
        if dpi <= 0:
            raise ValueError("dpi must be greater than zero")
        return round(self.width_mm / 25.4 * dpi), round(self.height_mm / 25.4 * dpi)


PUBLICATION_PRESETS = {
    "single_column": PublicationPreset("single_column", 89.0, 68.0),
    "double_column": PublicationPreset("double_column", 183.0, 105.0),
    "square": PublicationPreset("square", 140.0, 140.0),
    "wide": PublicationPreset("wide", 183.0, 86.0),
}


def export_figure(
    figure: go.Figure,
    path: str | Path,
    *,
    mode: RenderMode = "publication_light",
    preset: str = "double_column",
    dpi: int = 300,
    include_plotlyjs: str | bool = True,
) -> Path:
    """Export HTML, JSON, SVG, PDF, PNG, JPEG or WebP.

    Static image export uses Kaleido v1. Kaleido v1 also requires an installed
    Chrome or Chromium executable. HTML export is fully self-contained by default.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".html":
        figure.write_html(output, include_plotlyjs=include_plotlyjs, full_html=True)
        return output
    if suffix == ".json":
        output.write_text(figure.to_json(), encoding="utf-8")
        return output
    if suffix not in {".svg", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        raise ExportError(
            "Unsupported figure extension. Use HTML, JSON, SVG, PDF, PNG, JPEG or WebP."
        )
    if preset not in PUBLICATION_PRESETS:
        supported = ", ".join(sorted(PUBLICATION_PRESETS))
        raise ValueError(f"Unknown publication preset: {preset}. Choose from {supported}.")
    width, height = PUBLICATION_PRESETS[preset].pixels(dpi)
    try:
        figure.write_image(output, width=width, height=height, scale=1.0)
    except Exception as error:  # Plotly wraps missing Kaleido and Chrome differently by version.
        raise ExportError(
            "Static VRAVEN export requires Kaleido v1 and a compatible Chrome or Chromium "
            "installation. Install `vraven[publication]`, then run `plotly_get_chrome` if "
            "your system does not already provide Chrome. HTML export does not require Kaleido."
        ) from error
    return output
