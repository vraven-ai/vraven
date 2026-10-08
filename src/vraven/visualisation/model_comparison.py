"""Matched-sample comparison of two models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.visualisation.base import VisualContract, VisualisationMetadata, finalise_figure
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.model_comparison",
    research_question="For the same samples, where do two models align internally and where do their outputs diverge?",
    required_metrics=(
        "paired reasoning-fingerprint cosine similarity",
        "prediction agreement",
        "confidence difference",
    ),
    data_domain="matched sample traces from two models",
    visual_encodings=(
        "upper y=paired fingerprint cosine similarity",
        "upper marker shape and colour=prediction agreement",
        "lower y=right-model confidence minus left-model confidence",
        "x=matched sample identifier",
    ),
    limitations=(
        "Fingerprint similarity depends on the selected VRAVEN metrics and fixed-width resampling.",
        "Different layer counts are compared through fingerprint normalisation, not direct layer alignment.",
        "Output agreement does not establish equal internal computation.",
    ),
)


def model_comparison_figure(
    *,
    sample_ids: Sequence[str],
    fingerprint_similarity: Sequence[float],
    prediction_agreement: Sequence[bool],
    confidence_delta: Sequence[float],
    left_name: str,
    right_name: str,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    identifiers = tuple(str(item) for item in sample_ids)
    similarity = np.asarray(fingerprint_similarity, dtype=np.float64)
    agreement = np.asarray(prediction_agreement, dtype=bool)
    delta = np.asarray(confidence_delta, dtype=np.float64)
    if not (len(identifiers) == similarity.size == agreement.size == delta.size):
        raise ValueError("All model comparison arrays must have equal lengths")

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.62, 0.38],
    )
    index = np.arange(len(identifiers))
    for value, label, role, symbol in (
        (True, "Prediction agrees", "stable", "circle"),
        (False, "Prediction differs", "opposing", "x"),
    ):
        mask = agreement == value
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=index[mask],
                y=similarity[mask],
                mode="markers",
                name=label,
                marker={
                    "size": 10,
                    "color": semantic_colour(role, mode),
                    "symbol": symbol,
                    "line": {"width": 0.8},
                },
                customdata=np.asarray([identifiers[item] for item in index[mask]])[:, None],
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>Fingerprint similarity: %{y:.4f}<br>"
                    f"{label}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    finite = np.isfinite(delta)
    figure.add_trace(
        go.Bar(
            x=index[finite],
            y=delta[finite],
            name=f"Confidence Δ ({right_name} − {left_name})",
            marker_color=[
                semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
                for value in delta[finite]
            ],
            customdata=np.asarray([identifiers[item] for item in index[finite]])[:, None],
            hovertemplate="Sample: %{customdata[0]}<br>Confidence difference: %{y:+.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0.0, line_dash="dot", line_width=0.8, row=2, col=1)
    figure.update_yaxes(title_text="Fingerprint cosine similarity", range=[-1.05, 1.05], row=1, col=1)
    figure.update_yaxes(title_text="Confidence Δ", row=2, col=1)
    figure.update_xaxes(
        title_text="Matched sample",
        tickmode="array",
        tickvals=index,
        ticktext=identifiers,
        tickangle=-45,
        row=2,
        col=1,
    )
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="per-model fixed-width robust-scaled reasoning fingerprints; paired cosine similarity",
            extra={
                "left_model": left_name,
                "right_model": right_name,
                "sample_count": len(identifiers),
                "confidence_delta_semantics": "right_minus_left",
            },
        ),
        mode=mode,
        title="VRAVEN Matched Model Comparison",
        subtitle=f"{left_name} versus {right_name} on identical samples",
        width=max(900, min(1600, 38 * len(identifiers) + 460)),
        height=720,
        show_legend=True,
        show_title=show_title,
    )
