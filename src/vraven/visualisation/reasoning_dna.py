"""VRAVEN Reasoning DNA matrix and compressed fingerprint ribbon."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.reasoning.fingerprint import reasoning_dna, reasoning_fingerprint
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, signed_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.reasoning_dna",
    research_question="What deterministic multimetric signature describes this inference trace?",
    required_metrics=(
        "activation statistics",
        "gradient statistics when available",
        "fixed-width reasoning fingerprint",
    ),
    data_domain="robustly scaled layer-by-metric trace",
    visual_encodings=(
        "upper matrix: x=layer, y=metric, colour=robust scaled value",
        "lower ribbon: x=fingerprint position, colour=compressed signed value",
    ),
    limitations=(
        "Reasoning DNA is a comparison signature, not a literal biological or causal code.",
        "Zero gradient fields in fast mode mean unavailable gradient metrics, not measured absence.",
    ),
    interpretation=(
        "The matrix exposes robust-scaled layer metrics and the ribbon provides a deterministic "
        "comparison fingerprint; neither assigns semantic identities to its coordinates."
    ),
    metric_definitions=(
        "scaled metric = (value - per-metric median) / per-metric IQR, clipped to [-4, 4]",
        "fingerprint = fixed-width linear resampling of the ordered scaled trace vector",
    ),
)


def _short_layer(name: str, *, limit: int = 16) -> str:
    cleaned = name.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip("._") + "…"


def reasoning_dna_figure(
    trace: NeuralTrace,
    *,
    dimensions: int = 128,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    dna = reasoning_dna(trace)
    if dna.normalised_matrix.size == 0:
        raise CapabilityError("Reasoning DNA requires at least one captured layer")
    fingerprint = reasoning_fingerprint(trace, dimensions=dimensions)
    peak_metric, peak_layer = np.unravel_index(
        int(np.argmax(np.abs(dna.normalised_matrix.T))), dna.normalised_matrix.T.shape
    )
    layer_labels = [_short_layer(name) for name in dna.layer_names]
    # Large vertical gap so rotated layer ticks never collide with the ribbon title.
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.70, 0.18],
        vertical_spacing=0.26,
        subplot_titles=("Layer metric signature", "Fixed-width fingerprint ribbon"),
    )
    figure.add_trace(
        go.Heatmap(
            z=dna.normalised_matrix.T,
            x=layer_labels,
            customdata=np.asarray(dna.layer_names)[None, :].repeat(len(dna.metric_names), axis=0),
            y=dna.metric_names,
            colorscale=signed_colour_scale(mode),
            zmid=0.0,
            zmin=-4.0,
            zmax=4.0,
            colorbar={"title": "Robust<br>scaled", "len": 0.55, "y": 0.72, "yanchor": "middle"},
            hovertemplate=(
                "Layer: %{customdata}<br>Metric: %{y}<br>Scaled value: %{z:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=fingerprint.reshape(1, -1),
            x=np.arange(dimensions),
            y=[" "],
            colorscale=signed_colour_scale(mode),
            zmid=0.0,
            zmin=-4.0,
            zmax=4.0,
            showscale=False,
            hovertemplate="Position: %{x}<br>Value: %{z:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # No x-axis title on the upper plot: subplot title already names it, and an
    # axis title here was landing on top of the ribbon.
    figure.update_xaxes(
        tickangle=-48,
        tickfont={"size": 10},
        automargin=False,
        title_text="",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Trace metric",
        title_standoff=28,
        automargin=True,
        tickfont={"size": 11},
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Deterministic fingerprint position",
        title_standoff=20,
        tickfont={"size": 11},
        automargin=True,
        row=2,
        col=1,
    )
    figure.update_yaxes(showticklabels=False, title_text="", row=2, col=1)

    # Keep subplot titles out of the tick band.
    figure.update_annotations(
        selector={"text": "Layer metric signature"},
        font={"size": 13},
        yshift=8,
    )
    figure.update_annotations(
        selector={"text": "Fixed-width fingerprint ribbon"},
        font={"size": 13},
        yshift=10,
    )

    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="per-metric median/IQR scaling clipped to [-4, 4]; linear resampling",
            key_results=(
                PublicationResult(
                    "Largest robust-scaled coordinate",
                    f"{dna.layer_names[peak_layer]} · {dna.metric_names[peak_metric]}",
                    f"Scaled value {float(dna.normalised_matrix.T[peak_metric, peak_layer]):.3f}.",
                ),
                PublicationResult(
                    "Trace matrix",
                    f"{len(dna.layer_names)} × {len(dna.metric_names)}",
                    "Captured layers by fixed ordered metric fields.",
                ),
                PublicationResult(
                    "Fingerprint width",
                    str(dimensions),
                    "Deterministic comparison coordinates retained in the lower ribbon.",
                    "positions",
                ),
            ),
            extra={
                "fingerprint_dimensions": dimensions,
                "layer_names": list(dna.layer_names),
            },
        ),
        mode=mode,
        title="VRAVEN Reasoning DNA",
        subtitle="A deterministic trace signature for comparison and drift analysis",
        width=1180 if mode == "interactive_dark" else 1040,
        height=820 if mode == "interactive_dark" else 740,
        show_legend=False,
        show_title=show_title,
    )
    # Extra bottom room for fingerprint axis title; keep theme top margin.
    top = figure.layout.margin.t if figure.layout.margin and figure.layout.margin.t is not None else 72
    left = figure.layout.margin.l if figure.layout.margin and figure.layout.margin.l is not None else 96
    right = figure.layout.margin.r if figure.layout.margin and figure.layout.margin.r is not None else 56
    figure.update_layout(margin={"l": max(left, 110), "r": right, "t": top, "b": 88})
    return figure
