"""Standalone activation-energy profile."""

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
    visual_id="vraven.activation_energy",
    research_question="Where is activation magnitude concentrated across captured layers?",
    required_metrics=("activation_energy",),
    data_domain="non-negative layer magnitude",
    visual_encodings=("x=execution order", "y=sum of squared activations"),
    limitations=("Activation energy is observational magnitude, not causal importance.",),
    interpretation=(
        "Peak and concentration values describe where squared activation magnitude is stored; "
        "they do not establish target relevance or causal necessity."
    ),
    metric_definitions=(
        "activation energy_l = sum of squared captured activations in layer l",
        "peak share = maximum layer energy / total captured layer energy",
    ),
)


def activation_energy_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    values = [layer.activation_energy for layer in trace.layers]
    if not values or any(value is None for value in values):
        raise CapabilityError("Activation energy was not captured for every selected layer")
    full_names = [layer.name for layer in trace.layers]
    numeric_values = [float(value) for value in values if value is not None]
    peak_index = max(range(len(numeric_values)), key=numeric_values.__getitem__)
    total = sum(numeric_values)
    short_names = shorten_layer_names(full_names, max_length=22)
    figure = go.Figure(
        go.Bar(
            x=short_names,
            y=values,
            name="Activation energy",
            marker={"color": semantic_colour("supporting", mode), "line": {"width": 0}},
            customdata=full_names,
            hovertemplate=(
                "Layer: %{customdata}<br>Activation energy: %{y:.5g}<extra>VRAVEN</extra>"
            ),
        )
    )
    figure.update_xaxes(
        title_text="Captured layer",
        tickangle=-35,
        tickfont={"size": 10},
        title_standoff=28,
    )
    figure.update_yaxes(title_text="Activation energy")
    figure.update_layout(margin={"b": 130, "l": 90, "r": 40})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="none",
            key_results=(
                PublicationResult(
                    "Peak-energy layer",
                    full_names[peak_index],
                    f"Measured activation energy {numeric_values[peak_index]:.5g}.",
                ),
                PublicationResult(
                    "Peak share",
                    "Not estimable" if total <= 0 else f"{numeric_values[peak_index] / total:.1%}",
                    "Peak layer energy divided by total captured activation energy.",
                ),
                PublicationResult(
                    "Captured depth",
                    str(len(numeric_values)),
                    "Layers with a measured activation-energy summary.",
                    "layers",
                ),
            ),
        ),
        mode=mode,
        title="VRAVEN activation energy",
        subtitle="Reduced layer signal magnitude, not causal attribution",
        height=620,
        show_legend=False,
        show_title=show_title,
    )
