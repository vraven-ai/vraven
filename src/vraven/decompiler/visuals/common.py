"""Shared scientific visual helpers for VRAVEN decompiler figures."""

from __future__ import annotations

import math
from collections.abc import Iterable

import plotly.graph_objects as go

from vraven.visualisation.base import VisualisationMetadata, finalise_figure
from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme


def rgba(hex_colour: str, alpha: float) -> str:
    value = hex_colour.lstrip("#")
    if len(value) != 6:
        return hex_colour
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{max(0.0, min(1.0, alpha)):.3f})"


def safe_width(value: float, *, minimum: float = 0.7, maximum: float = 8.0) -> float:
    return max(minimum, min(maximum, minimum + math.log1p(max(0.0, value)) * 1.7))


def normalise(values: Iterable[float]) -> list[float]:
    data = [float(value) for value in values]
    if not data:
        return []
    low, high = min(data), max(data)
    if math.isclose(low, high):
        return [0.5 for _ in data]
    return [(value - low) / (high - low) for value in data]


def scientific_figure(
    figure: go.Figure,
    *,
    mode: RenderMode,
    visual_id: str,
    question: str,
    metrics: tuple[str, ...],
    encodings: tuple[str, ...],
    limitations: tuple[str, ...],
    title: str,
    subtitle: str,
    width: int = 1180,
    height: int = 720,
    show_legend: bool = True,
    extra: dict | None = None,
) -> go.Figure:
    metadata = VisualisationMetadata(
        visual_id=visual_id,
        scientific_question=question,
        metric_names=metrics,
        limitations=limitations,
        extra={"visual_encodings": list(encodings), **(extra or {})},
    )
    finalise_figure(
        figure,
        metadata=metadata,
        mode=mode,
        title=title,
        subtitle=subtitle,
        width=width,
        height=height,
        show_legend=show_legend,
    )
    return figure


def add_coverage_badge(
    figure: go.Figure,
    *,
    x: float,
    y: float,
    label: str,
    value: float,
    mode: RenderMode,
    colour: str,
) -> None:
    """Place a compact metric badge in the reserved band just above the plot.

    Prefer ``y`` slightly above ``1.0`` (paper). ``apply_vraven_theme`` expands the
    top margin from the annotation ceiling so badges cannot collide with titles.
    """

    theme = get_theme(mode)
    figure.add_annotation(
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        text=f"<b>{label}</b><br>{100 * value:.1f}%",
        showarrow=False,
        align="center",
        xanchor="center",
        # Badge sits in the reserved band above the plot; text always uses the
        # primary theme ink (never the badge accent) so label contrast stays high.
        yanchor="bottom",
        yshift=6,
        bgcolor=theme.raised_panel if mode == "interactive_dark" else theme.panel,
        bordercolor=rgba(colour, 0.85),
        borderwidth=1.5,
        borderpad=7,
        font={"size": theme.note_size + 1, "color": theme.primary_text},
    )


def coverage_badge_band(
    figure: go.Figure,
    badges: list[tuple[str, float, str]],
    *,
    mode: RenderMode,
    y: float = 1.0,
    x_start: float = 0.02,
    x_step: float = 0.20,
) -> None:
    """Lay out coverage badges in a single non-overlapping horizontal band."""

    for index, (label, value, colour) in enumerate(badges):
        add_coverage_badge(
            figure,
            x=x_start + index * x_step,
            y=y,
            label=label,
            value=value,
            mode=mode,
            colour=colour,
        )
