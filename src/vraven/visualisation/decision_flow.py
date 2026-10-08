"""VRAVEN Decision Flow visualisation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.decision_flow",
    research_question="How do supporting and opposing first-order signals vary through model depth?",
    required_metrics=("unit_contributions", "activation_gradient_signed"),
    data_domain="signed target-specific first-order influence",
    visual_encodings=(
        "x=layer execution order",
        "upper ribbon=supporting influence",
        "lower ribbon=opposing influence",
        "line=net signed influence",
    ),
    limitations=(
        "Activation-gradient interaction is a local first-order approximation, not causal evidence.",
        "The ribbons aggregate units and do not claim neuron-to-neuron connectivity.",
    ),
    interpretation=(
        "Positive net values mark layers where measured local support exceeds opposition; "
        "ribbon area reports allocation of absolute first-order mass through depth."
    ),
    metric_definitions=(
        "support_l = sum of positive unit contributions at layer l / global absolute mass",
        "opposition_l = sum of absolute negative unit contributions at layer l / global absolute mass",
        "net_l = support_l - opposition_l",
    ),
)


def _rgba(hex_colour: str, opacity: float) -> str:
    value = hex_colour.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{opacity})"


def _signals(trace: NeuralTrace) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    support: list[float] = []
    opposition: list[float] = []
    for layer in trace.layers:
        if layer.unit_contributions is not None:
            units = np.asarray(layer.unit_contributions, dtype=np.float64)
            positive = float(np.clip(units, 0.0, None).sum())
            negative = float(np.clip(-units, 0.0, None).sum())
        elif layer.activation_gradient_signed is not None:
            signed = float(layer.activation_gradient_signed)
            positive = max(signed, 0.0)
            negative = max(-signed, 0.0)
        else:
            continue
        names.append(layer.name)
        support.append(positive)
        opposition.append(negative)
    if not names:
        raise CapabilityError("Decision Flow requires gradient or causal mode")
    support_array = np.asarray(support)
    opposition_array = np.asarray(opposition)
    denominator = float((support_array + opposition_array).sum())
    if denominator > 0:
        support_array /= denominator
        opposition_array /= denominator
    return names, support_array, opposition_array


def decision_flow_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    names, support, opposition = _signals(trace)
    net = support - opposition
    dominant_index = int(np.argmax(np.abs(net)))
    supporting_colour = semantic_colour("supporting", mode)
    opposing_colour = semantic_colour("opposing", mode)
    selected_colour = semantic_colour("selected", mode)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=names,
            y=support,
            mode="lines",
            line={"color": supporting_colour, "width": 1.8},
            fill="tozeroy",
            fillcolor=_rgba(supporting_colour, 0.36 if mode == "interactive_dark" else 0.20),
            name="Supporting signal",
            hovertemplate="%{x}<br>Supporting share: %{y:.3%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=names,
            y=-opposition,
            mode="lines",
            line={"color": opposing_colour, "width": 1.8},
            fill="tozeroy",
            fillcolor=_rgba(opposing_colour, 0.36 if mode == "interactive_dark" else 0.20),
            name="Opposing signal",
            customdata=opposition,
            hovertemplate="%{x}<br>Opposing share: %{customdata:.3%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=names,
            y=net,
            mode="lines+markers",
            line={"color": selected_colour, "width": 2.5},
            marker={"size": 6, "color": selected_colour},
            name="Net signal",
            hovertemplate="%{x}<br>Net signed share: %{y:.3%}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Model depth")
    figure.update_yaxes(title_text="Share of total absolute first-order influence", tickformat=".0%")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="global L1 normalisation across captured layers",
            key_results=(
                PublicationResult(
                    "Dominant net layer",
                    names[dominant_index],
                    f"Net signed share {float(net[dominant_index]):.3%}.",
                ),
                PublicationResult(
                    "Total supporting mass",
                    f"{float(np.sum(support)):.1%}",
                    "Share of global absolute first-order mass with positive sign.",
                ),
                PublicationResult(
                    "Total opposing mass",
                    f"{float(np.sum(opposition)):.1%}",
                    "Share of global absolute first-order mass with negative sign.",
                ),
            ),
        ),
        mode=mode,
        title="VRAVEN Decision Flow",
        subtitle="Target-specific supporting and opposing first-order signals",
        show_title=show_title,
    )
