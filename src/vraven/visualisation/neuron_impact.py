"""Signed unit-impact field for one selected layer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import LayerTrace, NeuralTrace
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.neuron_impact_field",
    research_question="Which compressed units support or oppose the target in one layer?",
    required_metrics=("unit_contributions",),
    data_domain="signed compressed-unit first-order influence",
    visual_encodings=(
        "x=compressed unit index",
        "y=signed influence",
        "marker area=absolute influence",
        "colour=signal direction",
    ),
    limitations=(
        "Compressed units may aggregate adjacent channels or features.",
        "Marker area, rather than radius, is proportional to absolute influence.",
    ),
    interpretation=(
        "Vertical sign separates local target support from opposition and marker area ranks "
        "absolute influence within the selected compressed layer."
    ),
    metric_definitions=(
        "signed unit influence = reduced activation times target gradient",
        "marker area is proportional to absolute signed unit influence",
    ),
)


def _resolve_layer(trace: NeuralTrace, layer: str | int | None) -> LayerTrace:
    candidates = [item for item in trace.layers if item.unit_contributions is not None]
    if not candidates:
        raise CapabilityError("Neuron Impact Field requires gradient or causal mode")
    if layer is None:
        return max(candidates, key=lambda item: float(item.activation_gradient_influence or 0.0))
    if isinstance(layer, int):
        try:
            return candidates[layer]
        except IndexError as error:
            raise ValueError("Layer index is outside the available contribution layers") from error
    for candidate in candidates:
        if candidate.name == layer or candidate.metadata.get("base_module_name") == layer:
            return candidate
    raise ValueError(f"No captured contribution layer named {layer!r}")


def neuron_impact_figure(
    trace: NeuralTrace,
    *,
    layer: str | int | None = None,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    selected = _resolve_layer(trace, layer)
    values = np.asarray(selected.unit_contributions, dtype=np.float64)
    magnitude = np.abs(values)
    maximum = max(float(magnitude.max()) if magnitude.size else 0.0, 1e-12)
    strongest_support = int(np.argmax(values))
    strongest_opposition = int(np.argmin(values))
    # Plotly marker size is diameter. Square root makes marker area proportional to magnitude.
    sizes = 5.0 + 27.0 * np.sqrt(magnitude / maximum)
    colours = [
        semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
        for value in values
    ]
    figure = go.Figure(
        go.Scatter(
            x=np.arange(values.size),
            y=values,
            mode="markers",
            marker={
                "size": sizes,
                "color": colours,
                "opacity": 0.78,
                "line": {"width": 0.45, "color": semantic_colour("neutral", mode)},
            },
            customdata=magnitude,
            hovertemplate=(
                "Unit: %{x}<br>Signed influence: %{y:.5g}<br>"
                "Absolute influence: %{customdata:.5g}<extra></extra>"
            ),
            name="Compressed units",
        )
    )
    figure.add_hline(y=0.0, line_width=1.0, line_color=semantic_colour("neutral", mode))
    # Size legend proxies for absolute contribution magnitude.
    for legend_value in (0.05, 0.10, 0.20, 0.40):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 5.0 + 27.0 * np.sqrt(legend_value),
                    "color": semantic_colour("neutral", mode),
                    "opacity": 0.5,
                },
                name=f"|contrib| {legend_value:.2f}",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    figure.update_xaxes(title_text="Neuron index", title_standoff=16)
    figure.update_yaxes(title_text="Contribution", title_standoff=20, automargin=True)
    figure.update_layout(
        margin={"l": 90, "r": 160, "b": 90, "t": 90},
        legend={
            "title": {"text": "|Contribution|"},
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
    )
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="none; marker diameter square-root transformed so area represents magnitude",
            key_results=(
                PublicationResult(
                    "Strongest supporting unit",
                    str(strongest_support),
                    f"Signed first-order influence {float(values[strongest_support]):.5g}.",
                ),
                PublicationResult(
                    "Strongest opposing unit",
                    str(strongest_opposition),
                    f"Signed first-order influence {float(values[strongest_opposition]):.5g}.",
                ),
                PublicationResult(
                    "Compressed units displayed",
                    str(values.size),
                    f"Selected layer: {selected.name}.",
                    "units",
                ),
            ),
            extra={"layer": selected.name},
        ),
        mode=mode,
        title="VRAVEN Compressed-unit Influence Field",
        subtitle=f"Layer: {selected.name}",
        width=1180,
        height=640,
        show_legend=True,
        show_title=show_title,
    )
