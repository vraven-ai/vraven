"""Checkpoint evolution of internal trace fingerprints and output confidence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.visualisation.base import VisualContract, VisualisationMetadata, finalise_figure
from vraven.visualisation.palette import RenderMode, semantic_colour
from vraven.visualisation.theme import get_theme

CONTRACT = VisualContract(
    visual_id="vraven.model_evolution",
    research_question="How does one fixed input's reduced internal trace approach the final checkpoint, and how does output confidence change?",
    required_metrics=("fingerprint similarity to final checkpoint", "prediction confidence", "predicted class"),
    data_domain="one fixed input evaluated across ordered model checkpoints",
    visual_encodings=(
        "x=declared checkpoint order",
        "left y=cosine similarity to final checkpoint fingerprint",
        "right y=prediction confidence",
        "marker annotation=predicted class",
    ),
    limitations=(
        "Checkpoint order must be supplied in true training or version order.",
        "Similarity to the final checkpoint is descriptive and does not imply monotonic learning quality.",
        "Fingerprint comparison compresses potentially different internal architectures.",
    ),
)


def model_evolution_figure(
    *,
    checkpoint_names: Sequence[str],
    similarity_to_final: Sequence[float],
    confidences: Sequence[float],
    predictions: Sequence[int | None],
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    names = tuple(str(value) for value in checkpoint_names)
    similarity = np.asarray(similarity_to_final, dtype=np.float64)
    confidence = np.asarray(confidences, dtype=np.float64)
    if not (len(names) == similarity.size == confidence.size == len(predictions)):
        raise ValueError("Checkpoint evolution arrays must have equal lengths")
    index = np.arange(len(names))
    prediction_text = ["none" if value is None else str(value) for value in predictions]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    # Thick dotted purple (similarity) + thick dotted red (confidence) — distinct
    # from solid dual-axis clutter and readable on both render modes.
    purple = semantic_colour("latent", mode)
    red = semantic_colour("opposing", mode)
    figure.add_trace(
        go.Scatter(
            x=index,
            y=similarity,
            mode="lines+markers",
            name="Fingerprint similarity",
            line={"color": purple, "width": 5.0, "dash": "dot"},
            marker={"size": 12, "symbol": "diamond", "color": purple, "line": {"width": 0}},
            customdata=np.column_stack((names, prediction_text)),
            hovertemplate=(
                "Checkpoint: %{customdata[0]}<br>Similarity to final: %{y:.4f}<br>"
                "Prediction: %{customdata[1]}<extra>Fingerprint similarity</extra>"
            ),
        ),
        secondary_y=False,
    )
    finite = np.isfinite(confidence)
    figure.add_trace(
        go.Scatter(
            x=index[finite],
            y=confidence[finite],
            mode="lines+markers",
            name="Prediction confidence",
            line={"color": red, "width": 5.0, "dash": "dot"},
            marker={"size": 11, "symbol": "circle", "color": red, "line": {"width": 0}},
            customdata=np.column_stack(
                (
                    np.asarray(names, dtype=object)[finite],
                    np.asarray(prediction_text, dtype=object)[finite],
                )
            ),
            hovertemplate=(
                "Checkpoint: %{customdata[0]}<br>Confidence: %{y:.4f}<br>"
                "Prediction: %{customdata[1]}<extra>Prediction confidence</extra>"
            ),
        ),
        secondary_y=True,
    )
    figure.update_xaxes(
        title_text="Checkpoint order",
        tickmode="array",
        tickvals=index,
        ticktext=names,
        tickangle=-30,
        title_standoff=18,
        automargin=True,
    )
    figure.update_yaxes(
        title_text="Similarity to final fingerprint",
        range=[-1.05, 1.05],
        title_standoff=18,
        automargin=True,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Prediction confidence",
        range=[0.0, 1.05],
        title_standoff=18,
        automargin=True,
        secondary_y=True,
    )
    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="fixed-width robust-scaled fingerprint; cosine similarity to final checkpoint",
            extra={
                "checkpoint_count": len(names),
                "final_checkpoint": names[-1],
                "checkpoint_order_semantics": "user_declared_order",
            },
        ),
        mode=mode,
        title="VRAVEN Model Evolution",
        subtitle="Internal trace convergence and output confidence for one fixed input",
        width=max(1100, min(1600, 120 * len(names) + 520)),
        height=720,
        show_legend=True,
        # Dual-axis subplot title slot clips; brand heading is drawn below.
        show_title=False,
    )
    # Subplot dual-axis figures clip Plotly's default title slot; place the
    # brand heading as a paper annotation inside an oversized top margin.
    theme = get_theme(mode)
    show = True if show_title is None else bool(show_title)
    if show:
        figure.add_annotation(
            text=(
                "<b>VRAVEN Model Evolution</b><br>"
                f"<span style='font-size:{theme.note_size}px;color:{theme.secondary_text}'>"
                "Internal trace convergence and output confidence for one fixed input"
                "</span>"
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.18,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            font={"size": theme.title_size, "color": theme.primary_text},
        )
    figure.update_layout(
        margin={"l": 120, "r": 340, "t": 230, "b": 120},
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "traceorder": "normal",
            "itemsizing": "constant",
            "font": {"size": 12},
        },
    )
    return figure
