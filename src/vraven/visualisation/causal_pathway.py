"""Intervention score changes for causally tested layers."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_names
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.causal_pathway",
    research_question="Which temporary layer interventions most change the selected target score?",
    required_metrics=("intervention score_delta",),
    data_domain="baseline target score minus intervened target score",
    visual_encodings=("y=intervened layer", "x=target-score drop", "colour=delta direction"),
    limitations=(
        "Single-layer ablation can create out-of-distribution internal states.",
        "Intervention sensitivity supports a causal role but does not identify a complete mechanism.",
    ),
    evidence_tier="interventional",
    interpretation=(
        "A positive score drop means the ablated layer supported the selected target under "
        "the declared intervention; a negative drop means the intervention increased it."
    ),
    metric_definitions=(
        "score_delta_l = baseline target score - target score after do(layer_l = baseline)",
    ),
)


def causal_pathway_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if not trace.causal_interventions:
        raise CapabilityError("Causal Pathway requires a trace generated with mode='causal'")
    ordered = sorted(trace.causal_interventions, key=lambda item: abs(item.score_delta))
    full_names = [item.layer_name for item in ordered]
    names = shorten_layer_names(full_names, max_length=24)
    values = [item.score_delta for item in ordered]
    colours = [
        semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
        for value in values
    ]
    customdata = [
        [full_name, item.intervened_target_score]
        for full_name, item in zip(full_names, ordered, strict=True)
    ]
    strongest_support_index = int(max(range(len(values)), key=values.__getitem__))
    strongest_inverse_index = int(min(range(len(values)), key=values.__getitem__))
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": colours},
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]}<br>Target-score drop: %{x:.5g}<br>"
                "Intervened target score: %{customdata[1]:.5g}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0.0, line_width=1.0, line_color=semantic_colour("neutral", mode))
    figure.update_xaxes(
        title_text="Baseline target score − intervened target score",
        title_standoff=20,
    )
    figure.update_yaxes(
        title_text="Intervened layer",
        title_standoff=24,
        tickfont={"size": 11},
        automargin=True,
    )
    figure.update_layout(margin={"l": 140, "r": 48, "b": 90, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="none",
            key_results=(
                PublicationResult(
                    "Largest target-score drop",
                    f"{values[strongest_support_index]:.5g}",
                    f"Measured after intervening on {full_names[strongest_support_index]}.",
                    "score units",
                ),
                PublicationResult(
                    (
                        "Most negative score drop"
                        if values[strongest_inverse_index] < 0
                        else "Smallest target-score drop"
                    ),
                    f"{values[strongest_inverse_index]:.5g}",
                    f"Measured after intervening on {full_names[strongest_inverse_index]}.",
                    "score units",
                ),
                PublicationResult(
                    "Interventions tested",
                    str(len(ordered)),
                    "Temporary single-layer interventions included in this figure.",
                    "layers",
                ),
            ),
            extra={"intervention_count": len(ordered)},
        ),
        mode=mode,
        title="VRAVEN Single-layer Intervention Effects",
        subtitle="Controlled temporary layer-output interventions",
        height=max(520, 70 * len(ordered) + 180),
        show_legend=False,
        show_title=show_title,
    )
