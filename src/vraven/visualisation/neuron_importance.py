"""Top contributing neuron table for one layer (mockup-style importance view)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.core.errors import CapabilityError
from vraven.core.trace import LayerTrace, NeuralTrace
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_name
from vraven.visualisation.palette import RenderMode, semantic_colour
from vraven.visualisation.theme import get_theme

CONTRACT = VisualContract(
    visual_id="vraven.neuron_importance",
    research_question="Which compressed units contribute most for or against the selected target?",
    required_metrics=("unit_contributions", "unit activations when available"),
    data_domain="one captured layer",
    visual_encodings=(
        "table row=top absolute-contribution unit",
        "activation column=mean unit activation",
        "contribution column=signed contribution",
        "impact bars=absolute contribution",
    ),
    limitations=(
        "Compressed unit indices are not guaranteed to map 1:1 to architectural neurons.",
        "Values are first-order local contributions, not causal effects.",
    ),
    interpretation=(
        "The table preserves exact signed values while the paired bars rank their absolute "
        "magnitude; positive values support and negative values oppose the selected target."
    ),
    metric_definitions=(
        "unit contribution = reduced activation times target gradient",
        "displayed mass share = sum top-k |contribution| / sum all |contribution|",
    ),
)


def _resolve_layer(trace: NeuralTrace, layer: str | int | None) -> LayerTrace:
    candidates = [item for item in trace.layers if item.unit_contributions is not None]
    if not candidates:
        raise CapabilityError("Neuron Importance requires gradient or causal mode")
    if layer is None:
        return max(candidates, key=lambda item: float(item.activation_gradient_influence or 0.0))
    if isinstance(layer, int):
        return candidates[layer]
    for candidate in candidates:
        if candidate.name == layer:
            return candidate
    raise CapabilityError(f"No contribution layer named {layer!r}")


def neuron_importance_figure(
    trace: NeuralTrace,
    *,
    layer: str | int | None = None,
    top_k: int = 12,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    selected = _resolve_layer(trace, layer)
    contributions = np.asarray(selected.unit_contributions, dtype=np.float64).reshape(-1)
    activations = (
        np.asarray(selected.unit_activations, dtype=np.float64).reshape(-1)
        if selected.unit_activations is not None
        else np.full(contributions.shape, np.nan)
    )
    if activations.size != contributions.size:
        activations = np.full(contributions.shape, np.nan)

    order = np.argsort(np.abs(contributions))[::-1][: min(top_k, contributions.size)]
    selected_contrib = contributions[order]
    selected_act = activations[order]
    impact = np.abs(selected_contrib)
    neuron_ids = [f"Neuron {int(index)}" for index in order]
    # Table rows stay strongest-first; bars are drawn strongest at the top.
    bar_order = np.arange(order.size)[::-1]
    bar_labels = [neuron_ids[i] for i in bar_order]
    bar_values = selected_contrib[bar_order]
    bar_colours = [
        semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
        for value in bar_values
    ]

    theme = get_theme(mode)
    header_fill = theme.raised_panel
    cell_fill = theme.panel
    text_colour = theme.primary_text
    muted = theme.secondary_text
    contribution_colours = [
        semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
        for value in selected_contrib
    ]

    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.08,
        specs=[[{"type": "table"}, {"type": "xy"}]],
    )
    figure.add_trace(
        go.Table(
            columnwidth=[90, 80, 95],
            header={
                "values": ["Neuron ID", "Activation", "Contribution"],
                "fill_color": header_fill,
                "font": {"color": muted, "size": 12},
                "align": "left",
                "height": 34,
                "line": {"color": theme.border},
            },
            cells={
                "values": [
                    neuron_ids,
                    [f"{value:.3f}" if np.isfinite(value) else "—" for value in selected_act],
                    [
                        f"{value:+.3f}" if np.isfinite(value) else "—"
                        for value in selected_contrib
                    ],
                ],
                "fill_color": cell_fill,
                "font": {
                    "color": [text_colour, text_colour, contribution_colours],
                    "size": 12,
                },
                "align": "left",
                "height": 30,
                "line": {"color": theme.border},
            },
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=impact[bar_order],
            y=bar_labels,
            orientation="h",
            marker={"color": bar_colours, "line": {"width": 0}},
            customdata=bar_values,
            hovertemplate=(
                "%{y}<br>Contribution: %{customdata:+.5g}<br>"
                "Impact: %{x:.5g}<extra></extra>"
            ),
            name="Impact",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="Impact |contribution|",
        title_standoff=14,
        row=1,
        col=2,
    )
    figure.update_yaxes(automargin=True, row=1, col=2)
    figure.update_layout(margin={"l": 24, "r": 40, "b": 70, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="rows ordered by absolute contribution",
            key_results=(
                PublicationResult(
                    "Largest absolute contributor",
                    neuron_ids[0],
                    f"Signed contribution {float(selected_contrib[0]):+.5g}.",
                ),
                PublicationResult(
                    "Displayed contribution mass",
                    (
                        "Not estimable"
                        if float(np.sum(np.abs(contributions))) <= 0
                        else f"{float(np.sum(impact) / np.sum(np.abs(contributions))):.1%}"
                    ),
                    "Share of total compressed-unit absolute contribution retained by top-k.",
                ),
                PublicationResult(
                    "Selected layer",
                    selected.name,
                    f"{int(order.size)} of {int(contributions.size)} compressed units displayed.",
                ),
            ),
            extra={"layer": selected.name, "top_k": int(order.size)},
        ),
        mode=mode,
        title="VRAVEN Top Local Unit Contributions",
        subtitle=(
            f"Top contributing neurons · Layer {shorten_layer_name(selected.name, max_length=30)}"
        ),
        width=1180,
        height=max(560, 34 * int(order.size) + 220),
        show_legend=False,
        show_title=show_title,
    )
