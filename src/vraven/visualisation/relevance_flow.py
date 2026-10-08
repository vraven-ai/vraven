"""Layer-wise LRP relevance flow across supported modules."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.attribution.lrp import LRPResult
from vraven.core.errors import CapabilityError
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_names
from vraven.visualisation.palette import RenderMode, relevance_colour_scale, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.relevance_flow",
    research_question="How does LRP relevance redistribute from the target score through supported layers?",
    required_metrics=("layer relevance from LRP",),
    data_domain="epsilon-rule LRP on Linear/ReLU/Embedding",
    visual_encodings=(
        "panel=layer or input",
        "heatmap colour=relevance magnitude",
        "arrow=execution order",
    ),
    limitations=(
        "Only Linear, ReLU and Embedding rules are implemented.",
        "Conservation is approximate when unsupported modules intervene.",
    ),
    interpretation=(
        "Shared-scale panels show how absolute epsilon-rule relevance is redistributed; "
        "the conservation ratio determines whether the rendering is confirmatory or exploratory."
    ),
    metric_definitions=(
        "conservation ratio = summed propagated relevance / selected target score",
        "panel colour = absolute epsilon-rule relevance with one shared scale",
    ),
)


def relevance_flow_figure(
    result: LRPResult,
    *,
    max_layers: int = 6,
    max_units: int = 48,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if not result.layers and result.input_relevance is None:
        raise ValueError("LRP result contains no relevance maps")

    panels: list[tuple[str, np.ndarray]] = []
    if result.input_relevance is not None:
        values = np.abs(np.asarray(result.input_relevance, dtype=np.float64)).reshape(-1)
        panels.append(("Input", values))
    selected_layers = result.layers
    if len(selected_layers) > max_layers:
        index = np.linspace(0, len(selected_layers) - 1, max_layers, dtype=int)
        selected_layers = [selected_layers[i] for i in index]
    for layer in selected_layers:
        values = np.abs(np.asarray(layer.relevance_values, dtype=np.float64)).reshape(-1)
        panels.append((layer.name, values))

    short = shorten_layer_names([name for name, _ in panels], max_length=18)
    column_count = len(panels)
    figure = make_subplots(
        rows=1,
        cols=column_count,
        subplot_titles=short,
        horizontal_spacing=0.04,
    )
    vmax = max(float(np.max(values)) if values.size else 0.0 for _, values in panels)
    vmax = max(vmax, 1e-12)
    colourscale = relevance_colour_scale(mode)
    for column, (_name, values) in enumerate(panels, start=1):
        width = min(max_units, max(8, int(np.ceil(np.sqrt(values.size)))))
        rows = int(np.ceil(values.size / width))
        grid = np.full(rows * width, np.nan)
        grid[: values.size] = values
        grid = grid.reshape(rows, width)
        figure.add_trace(
            go.Heatmap(
                z=grid,
                zmin=0.0,
                zmax=vmax,
                colorscale=colourscale,
                showscale=column == column_count,
                colorbar={"title": "Relevance", "thickness": 12} if column == column_count else None,
                hovertemplate="Row %{y}<br>Col %{x}<br>Relevance %{z:.4g}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(showticklabels=False, title_text=None, row=1, col=column)
        figure.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=column)

    for boundary in range(1, column_count):
        figure.add_annotation(
            x=boundary / column_count,
            y=1.08,
            xref="paper",
            yref="paper",
            text="→",
            showarrow=False,
            font={"size": 20, "color": semantic_colour("selected", mode)},
        )
    ratio = float(result.conservation_ratio)
    if not (0.01 <= ratio <= 100.0):
        raise CapabilityError(
            f"LRP conservation_ratio={ratio:.4g} is outside [0.01, 100.0]. "
            "The figure is suppressed because relevance loss is too severe for a "
            "scientifically useful rendering on this architecture."
        )
    if not (0.95 <= ratio <= 1.05):
        figure.add_annotation(
            x=0.5,
            y=-0.12,
            xref="paper",
            yref="paper",
            text=(
                f"Exploratory LRP: layer relevance conservation ratio {ratio:.4f}; "
                "normalized across layers for scientific inspection."
            ),
            showarrow=False,
            font={"size": 11, "color": semantic_colour("uncertainty", mode)},
        )

    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="absolute relevance; colour scale shared across panels",
            key_results=(
                PublicationResult(
                    "Relevance conservation",
                    f"{ratio:.4f}",
                    (
                        "Within the registered confirmatory interval [0.95, 1.05]."
                        if 0.95 <= ratio <= 1.05
                        else "Outside [0.95, 1.05]; interpret this rendering as exploratory."
                    ),
                    "ratio",
                ),
                PublicationResult(
                    "Target score",
                    f"{float(result.target_score):.5g}",
                    "Model target score used to initialise relevance propagation.",
                ),
                PublicationResult(
                    "Rendered path",
                    str(column_count),
                    "Input/layer panels sharing one absolute relevance scale.",
                    "panels",
                ),
            ),
            extra={
                "conservation_ratio": result.conservation_ratio,
                "epsilon": result.epsilon,
                "target_score": result.target_score,
                "panel_names": [name for name, _ in panels],
            },
        ),
        mode=mode,
        title="VRAVEN Relevance Flow (LRP)",
        subtitle="Relevance flow (input → supported layers)",
        width=max(960, 210 * column_count),
        height=520 if mode == "publication_light" else 600,
        show_legend=False,
        show_title=show_title,
    )
