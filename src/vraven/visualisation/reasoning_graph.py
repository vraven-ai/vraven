"""Execution-summary reasoning graph."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.reasoning.graph import build_reasoning_graph
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_names
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.reasoning_graph",
    research_question="How are captured layer summaries ordered and where is target influence concentrated?",
    required_metrics=("execution order", "activation energy", "signed influence when available"),
    data_domain="sequential captured-module summary",
    visual_encodings=(
        "x=execution index",
        "y=signed influence or centred activation magnitude",
        "node area=activation energy",
        "edge=observed execution transition",
    ),
    limitations=(
        "Edges represent observed execution order, not neuron connectivity or causality.",
        "Fast-mode vertical positions are centred activation-magnitude proxies.",
    ),
    interpretation=(
        "The graph preserves execution order while marking where the selected vertical metric "
        "and activation energy are concentrated; its edges are not anatomical connectivity."
    ),
    metric_definitions=(
        "node area is proportional to captured activation energy after square-root diameter scaling",
        "vertical value is signed first-order influence when available, otherwise centred mean energy",
    ),
)


def reasoning_graph_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if not trace.layers:
        raise CapabilityError("Reasoning Graph requires captured layers")
    graph = build_reasoning_graph(trace)
    x = np.asarray([layer.execution_index for layer in trace.layers], dtype=float)
    has_gradient = any(layer.activation_gradient_signed is not None for layer in trace.layers)
    if has_gradient:
        y = np.asarray([float(layer.activation_gradient_signed or 0.0) for layer in trace.layers])
        y_label = "Signed first-order influence"
    else:
        energy = np.asarray([float(layer.mean_energy or 0.0) for layer in trace.layers])
        y = energy - np.median(energy)
        y_label = "Centred mean activation energy proxy"
    energies = np.asarray([float(layer.activation_energy or 0.0) for layer in trace.layers])
    maximum = max(float(energies.max()), 1e-12)
    sizes = 8.0 + 25.0 * np.sqrt(energies / maximum)
    dominant_vertical = int(np.argmax(np.abs(y)))
    dominant_energy = int(np.argmax(energies))
    colours = [
        semantic_colour("supporting", mode)
        if value > 0
        else semantic_colour("opposing", mode)
        if value < 0
        else semantic_colour("neutral", mode)
        for value in y
    ]
    full_names = [layer.name for layer in trace.layers]
    short_names = shorten_layer_names(full_names, max_length=18)
    # Alternate annotation side so adjacent long labels do not collide.
    text_positions = [
        "top center" if index % 2 == 0 else "bottom center" for index in range(len(short_names))
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": semantic_colour("neutral", mode), "width": 1.1},
            name="Execution transition",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=short_names,
            textposition=text_positions,
            textfont={"size": 9},
            marker={"size": sizes, "color": colours, "opacity": 0.85},
            customdata=np.column_stack(
                (energies, [layer.layer_type for layer in trace.layers], full_names)
            ),
            hovertemplate=(
                "Layer: %{customdata[2]}<br>Type: %{customdata[1]}<br>"
                "Vertical metric: %{y:.5g}<br>Activation energy: %{customdata[0]:.5g}<extra></extra>"
            ),
            name="Captured layer",
        )
    )
    figure.add_hline(y=0.0, line_width=0.8, line_dash="dot", line_color=semantic_colour("neutral", mode))
    figure.update_xaxes(
        title_text="Execution index",
        tickmode="array",
        tickvals=list(x),
        ticktext=[str(int(value)) for value in x],
        title_standoff=20,
    )
    figure.update_yaxes(title_text=y_label, title_standoff=16)
    figure.update_layout(margin={"b": 100, "t": 90, "l": 100, "r": 40})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="node diameter square-root transformed so area represents activation energy",
            key_results=(
                PublicationResult(
                    "Largest vertical signal",
                    trace.layers[dominant_vertical].name,
                    f"{y_label}: {float(y[dominant_vertical]):.5g}.",
                ),
                PublicationResult(
                    "Largest activation-energy node",
                    trace.layers[dominant_energy].name,
                    f"Captured activation energy {float(energies[dominant_energy]):.5g}.",
                ),
                PublicationResult(
                    "Execution coverage",
                    str(len(trace.layers)),
                    "Captured executions connected in observed order.",
                    "nodes",
                ),
            ),
            extra={"graph_semantics": graph.graph.get("semantics"), "causal": False},
        ),
        mode=mode,
        title="VRAVEN Reasoning Graph",
        subtitle="Sequential layer-summary graph; edges are execution transitions",
        width=max(980, 110 * len(trace.layers)),
        height=700,
        show_title=show_title,
    )
