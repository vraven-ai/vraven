"""Signed layer-by-unit spectral trace."""

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
from vraven.visualisation.palette import RenderMode, signed_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.spectral_trace",
    research_question="Where are signed target-specific unit signals concentrated across depth?",
    required_metrics=("unit_contributions",),
    data_domain="layer by compressed-unit signed influence",
    visual_encodings=("x=compressed unit", "y=layer", "colour=signed influence"),
    limitations=(
        "Units may be adaptively pooled when a layer exceeds the configured unit limit.",
        "First-order influence is not causal evidence.",
    ),
    evidence_tier="observational",
    interpretation=(
        "The extrema identify the compressed units with the strongest signed local influence; "
        "the heatmap reveals whether that evidence is concentrated or distributed through depth."
    ),
    metric_definitions=(
        "unit influence = reduced activation times target-gradient contribution",
        "colour limit = 99th percentile of finite absolute unit influence",
    ),
)


def spectral_trace_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    max_layers: int = 64,
    show_title: bool | None = None,
) -> go.Figure:
    selected = [layer for layer in trace.layers if layer.unit_contributions is not None]
    if not selected:
        raise CapabilityError("Spectral Trace requires unit contributions from gradient or causal mode")
    selected = selected[-max_layers:]
    width = max(len(layer.unit_contributions or []) for layer in selected)
    matrix = np.full((len(selected), width), np.nan, dtype=np.float64)
    for row, layer in enumerate(selected):
        values = np.asarray(layer.unit_contributions, dtype=np.float64)
        matrix[row, : values.size] = values
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    limit = max(limit, 1e-12)
    support_row, support_unit = np.unravel_index(int(np.nanargmax(matrix)), matrix.shape)
    opposing_row, opposing_unit = np.unravel_index(int(np.nanargmin(matrix)), matrix.shape)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(width),
            y=[layer.name for layer in selected],
            colorscale=signed_colour_scale(mode),
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Signed<br>influence"},
            hovertemplate="Layer: %{y}<br>Unit: %{x}<br>Influence: %{z:.5g}<extra></extra>",
            hoverongaps=False,
        )
    )
    figure.update_xaxes(title_text="Compressed unit index")
    figure.update_yaxes(title_text="Captured layer", autorange="reversed")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="colour clipped symmetrically at the 99th absolute percentile; data unchanged",
            key_results=(
                PublicationResult(
                    "Strongest supporting unit",
                    f"{selected[support_row].name}[{support_unit}]",
                    f"Signed first-order influence {float(matrix[support_row, support_unit]):.5g}.",
                ),
                PublicationResult(
                    "Strongest opposing unit",
                    f"{selected[opposing_row].name}[{opposing_unit}]",
                    f"Signed first-order influence {float(matrix[opposing_row, opposing_unit]):.5g}.",
                ),
                PublicationResult(
                    "Coverage",
                    f"{len(selected)} × {width}",
                    "Displayed layer-by-compressed-unit matrix shape; missing cells remain unmeasured.",
                ),
            ),
        ),
        mode=mode,
        title="VRAVEN Spectral Trace",
        subtitle="Signed activation-gradient influence across layers and compressed units",
        width=1100 if mode == "interactive_dark" else 980,
        height=max(480, min(1100, 120 + 24 * len(selected))),
        show_legend=False,
        show_title=show_title,
    )
