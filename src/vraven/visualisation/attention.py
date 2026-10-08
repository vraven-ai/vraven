"""Attention-weight structure visualisation."""

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
from vraven.visualisation.palette import RenderMode, sequential_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.attention_structure",
    research_question="How is attention weight distributed between query and key positions?",
    required_metrics=("attention",),
    data_domain="non-negative attention weight",
    visual_encodings=("x=key position", "y=query position", "colour=attention weight"),
    limitations=(
        "Raw attention weights are not causal importance.",
        "Some model implementations do not expose attention weights to module hooks.",
    ),
    interpretation=(
        "The peak identifies the largest exposed query-key weight in the selected head; "
        "attention concentration describes routing weights, not explanation importance."
    ),
    metric_definitions=(
        "peak attention = maximum model-exposed attention weight in the selected head",
        "mean row maximum = mean across queries of each query row's maximum key weight",
    ),
)


def _resolve_attention_layer(trace: NeuralTrace, layer: str | int | None) -> LayerTrace:
    candidates = [item for item in trace.layers if item.attention is not None]
    if not candidates:
        raise CapabilityError(
            "No attention matrix was exposed. Configure the model to return attention weights "
            "or use an architecture-specific adapter."
        )
    if layer is None:
        return candidates[0]
    if isinstance(layer, int):
        try:
            return candidates[layer]
        except IndexError as error:
            raise ValueError("Attention layer index is outside the captured range") from error
    for candidate in candidates:
        if candidate.name == layer or candidate.metadata.get("base_module_name") == layer:
            return candidate
    raise ValueError(f"No captured attention layer named {layer!r}")


def attention_structure_figure(
    trace: NeuralTrace,
    *,
    layer: str | int | None = None,
    head: int = 0,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    selected = _resolve_attention_layer(trace, layer)
    attention = np.asarray(selected.attention, dtype=np.float64)
    if attention.ndim != 3:
        raise CapabilityError("Captured attention must have [heads, query, key] shape")
    if head < 0 or head >= attention.shape[0]:
        raise ValueError(f"head must be between 0 and {attention.shape[0] - 1}")
    matrix = attention[head]
    peak_query, peak_key = np.unravel_index(int(np.nanargmax(matrix)), matrix.shape)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(matrix.shape[1]),
            y=np.arange(matrix.shape[0]),
            colorscale=sequential_colour_scale(mode),
            zmin=float(np.nanmin(matrix)),
            zmax=float(np.nanmax(matrix)),
            colorbar={"title": "Attention<br>weight"},
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.5g}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Key position")
    figure.update_yaxes(title_text="Query position", autorange="reversed")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="model-provided attention; spatial pooling only when configured limit is exceeded",
            key_results=(
                PublicationResult(
                    "Peak attention pair",
                    f"query {peak_query} → key {peak_key}",
                    f"Model-exposed attention weight {float(matrix[peak_query, peak_key]):.5g}.",
                ),
                PublicationResult(
                    "Mean row maximum",
                    f"{float(np.nanmean(np.nanmax(matrix, axis=1))):.5g}",
                    "Mean of the largest key weight for each query position.",
                ),
                PublicationResult(
                    "Selected head",
                    f"{selected.name} · head {head}",
                    f"Displayed attention matrix shape {matrix.shape[0]} × {matrix.shape[1]}.",
                ),
            ),
            extra={"layer": selected.name, "head": head},
        ),
        mode=mode,
        title="VRAVEN Attention Structure",
        subtitle=f"Layer: {selected.name}; head: {head}; attention weight is not causal importance",
        show_legend=False,
        show_title=show_title,
    )
