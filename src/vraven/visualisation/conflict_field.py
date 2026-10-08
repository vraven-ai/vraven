"""Separated predictive and explanation-risk dimensions."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.metrics.uncertainty import conflict_profile
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.conflict_field",
    research_question="Which distinct uncertainty and reasoning-risk dimensions are elevated?",
    required_metrics=(
        "predictive_entropy",
        "prediction_margin",
        "internal_opposition",
        "pathway_concentration",
        "causal_sensitivity when available",
    ),
    data_domain="separate descriptive scores in [0, 1]",
    visual_encodings=("y=metric", "x=score", "colour=metric semantics"),
    limitations=(
        "The metrics are not combined into an unvalidated composite risk score.",
        "Pathway concentration describes dependence concentration rather than uncertainty.",
    ),
    interpretation=(
        "Each bar must be interpreted independently under its own definition. The largest "
        "dimension is a diagnostic priority, not a composite model-risk verdict."
    ),
    metric_definitions=(
        "predictive uncertainty derives from normalised predictive entropy",
        "margin uncertainty is the registered transform of target-output separation",
        "internal opposition and pathway concentration remain separate explanation diagnostics",
    ),
)


def conflict_field_figure(
    trace: NeuralTrace,
    *,
    sample_index: int = 0,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    profile = conflict_profile(trace, sample_index=sample_index)
    definitions = [
        ("Predictive uncertainty", profile.predictive_uncertainty, "uncertainty"),
        ("Margin uncertainty", profile.target_margin_uncertainty, "uncertainty"),
        ("Internal opposition", profile.internal_opposition, "opposing"),
        ("Pathway concentration", profile.pathway_concentration, "latent"),
        ("Causal sensitivity", profile.causal_sensitivity, "stable"),
    ]
    available = [(name, value, role) for name, value, role in definitions if value is not None]
    if not available:
        raise CapabilityError("No uncertainty or explanation-risk quantities are available")
    names = [item[0] for item in available]
    values = [float(item[1]) for item in available]
    largest_index = max(range(len(values)), key=values.__getitem__)
    colours = [semantic_colour(item[2], mode) for item in available]  # type: ignore[arg-type]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": colours, "line": {"width": 0}},
            text=[f"{value:.1%}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            textfont={"size": 12},
        )
    )
    # Extra headroom on x so outside bar labels (e.g. 90.7%) stay visible.
    figure.update_xaxes(
        title_text="Metric value",
        range=[0.0, 1.18],
        tickformat=".0%",
        automargin=True,
        title_standoff=18,
    )
    figure.update_yaxes(
        title_text=None,
        autorange="reversed",
        automargin=True,
        tickfont={"size": 12},
        ticksuffix="  ",
    )
    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="each metric retains its documented [0, 1] definition",
            key_results=(
                PublicationResult(
                    "Largest diagnostic dimension",
                    names[largest_index],
                    f"Registered value {values[largest_index]:.3f}; do not combine it into a risk score.",
                ),
                PublicationResult(
                    "Predictive uncertainty",
                    (
                        "Unavailable"
                        if profile.predictive_uncertainty is None
                        else f"{profile.predictive_uncertainty:.3f}"
                    ),
                    "Normalised output-distribution entropy for the selected sample.",
                ),
                PublicationResult(
                    "Dimensions available",
                    str(len(available)),
                    "Separately defined diagnostics rendered without aggregation.",
                ),
            ),
            extra={"sample_index": sample_index},
        ),
        mode=mode,
        title="VRAVEN Separated Uncertainty Diagnostics",
        subtitle="Separate uncertainty and reasoning-risk dimensions; no composite score",
        show_legend=False,
        show_title=show_title,
        width=1100 if mode == "interactive_dark" else 980,
        height=520 if mode == "interactive_dark" else 480,
    )
    top = figure.layout.margin.t if figure.layout.margin and figure.layout.margin.t is not None else 86
    # Wide left gutter: category labels like "Pathway concentration" must not clip.
    figure.update_layout(
        margin={"l": 210, "r": 72, "t": top, "b": 72},
        title={
            "text": figure.layout.title.text if figure.layout.title else None,
            "x": 0.0,
            "xref": "paper",
            "xanchor": "left",
            "pad": {"l": 4, "t": 2, "b": 12},
        }
        if figure.layout.title and figure.layout.title.text
        else None,
    )
    return figure
