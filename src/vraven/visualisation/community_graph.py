"""Latent activation-community graph for one captured layer."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import LayerTrace, NeuralTrace
from vraven.reasoning.communities import ActivationCommunity, discover_activation_communities
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, semantic_colour, signed_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.community_graph",
    research_question=(
        "Which groups of units show similar activation patterns across the captured batch, "
        "and how do their target-specific contributions differ?"
    ),
    required_metrics=("unit activation matrix", "community membership", "mean unit contribution"),
    data_domain="one captured layer",
    visual_encodings=(
        "node=deterministically clustered latent activation community",
        "node area=number of units in the community",
        "node colour=mean target-specific first-order contribution",
        "edge width=positive cosine similarity between community centroids",
        "position=deterministic circular order, not learned geometry",
    ),
    limitations=(
        "Communities are statistical activation groups and are not automatically human concepts.",
        "Edges indicate positive centroid similarity, not anatomical connectivity or causality.",
        "The circular layout is deterministic presentation and carries no distance meaning.",
    ),
    interpretation=(
        "Node size reports community membership and edges report thresholded positive centroid "
        "similarity; neither the circular distance nor an edge implies causal connectivity."
    ),
    metric_definitions=(
        "community size = number of units assigned by deterministic farthest-first k-means",
        "edge similarity = cosine similarity between community activation centroids",
    ),
)


def _resolve_layer(trace: NeuralTrace, layer: str | int | None) -> LayerTrace:
    candidates = [candidate for candidate in trace.layers if candidate.unit_matrix]
    if not candidates:
        raise CapabilityError("Community Graph requires a captured batch unit matrix")
    if layer is None:
        return max(candidates, key=lambda item: len(item.unit_activations or ()))
    if isinstance(layer, int):
        try:
            return candidates[layer]
        except IndexError as error:
            raise CapabilityError(f"Community layer index {layer} is outside the captured range") from error
    for candidate in candidates:
        if candidate.name == layer:
            return candidate
    raise CapabilityError(f"No community-capable captured layer is named {layer!r}")


def _centroid_similarity(left: ActivationCommunity, right: ActivationCommunity) -> float:
    a = np.asarray(left.centroid, dtype=np.float64)
    b = np.asarray(right.centroid, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def community_graph_figure(
    trace: NeuralTrace,
    *,
    layer: str | int | None = None,
    n_clusters: int = 6,
    similarity_threshold: float = 0.20,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    """Render a deterministic graph of activation-pattern communities."""

    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than zero")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")

    selected = _resolve_layer(trace, layer)
    communities = discover_activation_communities(selected, n_clusters=n_clusters)
    if not communities:
        raise CapabilityError("No activation communities could be derived")

    count = len(communities)
    angles = np.linspace(math.pi / 2.0, math.pi / 2.0 + 2.0 * math.pi, count, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    figure = go.Figure()
    edge_records: list[tuple[int, int, float]] = []
    for first in range(count):
        for second in range(first + 1, count):
            similarity = _centroid_similarity(communities[first], communities[second])
            if similarity < similarity_threshold:
                continue
            edge_records.append((first, second, similarity))
            figure.add_trace(
                go.Scatter(
                    x=[x[first], x[second]],
                    y=[y[first], y[second]],
                    mode="lines",
                    line={
                        "color": semantic_colour("neutral", mode),
                        "width": 0.8 + 3.6 * similarity,
                    },
                    opacity=0.28 + 0.55 * similarity,
                    hovertemplate=(
                        f"Community {communities[first].community_id} ↔ "
                        f"Community {communities[second].community_id}<br>"
                        f"Centroid cosine similarity: {similarity:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
    # Annotate the strongest links so neuron-community relationships remain readable.
    for first, second, similarity in sorted(edge_records, key=lambda item: item[2], reverse=True)[:4]:
        figure.add_annotation(
            x=float((x[first] + x[second]) / 2.0),
            y=float((y[first] + y[second]) / 2.0),
            text=f"sim {similarity:.2f}",
            showarrow=False,
            font={"size": 9, "color": semantic_colour("selected", mode)},
            bgcolor="rgba(0,0,0,0)",
        )

    sizes = np.asarray([len(community.unit_indices) for community in communities], dtype=float)
    contribution_available = any(community.mean_contribution is not None for community in communities)
    contributions = np.asarray(
        [float(community.mean_contribution or 0.0) for community in communities], dtype=float
    )
    colour_limit = max(float(np.max(np.abs(contributions))), 1e-12)
    largest_community = int(np.argmax(sizes))
    strongest_edge = max(edge_records, key=lambda item: item[2]) if edge_records else None
    customdata = np.asarray(
        [
            [
                len(community.unit_indices),
                community.mean_activation,
                community.mean_contribution if community.mean_contribution is not None else np.nan,
                ", ".join(str(index) for index in community.unit_indices[:12]),
                community.community_id,
            ]
            for community in communities
        ],
        dtype=object,
    )

    marker: dict[str, object] = {
        "size": sizes,
        "sizemode": "area",
        "sizeref": max(2.0 * float(sizes.max()) / 58.0**2, 1e-6),
        "sizemin": 12,
        "line": {"color": semantic_colour("selected", mode), "width": 1.0},
        "opacity": 0.94,
    }
    if contribution_available:
        marker.update(
            {
                "color": contributions,
                "colorscale": signed_colour_scale(mode),
                "cmin": -colour_limit,
                "cmax": colour_limit,
                "cmid": 0.0,
                "colorbar": {
                    "title": "Mean<br>contribution",
                    "thickness": 12,
                    "len": 0.68,
                },
            }
        )
    else:
        marker["color"] = semantic_colour("latent", mode)

    node_labels = [
        f"C{community.community_id}<br>{len(community.unit_indices)}u"
        for community in communities
    ]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont={"size": 11},
            marker=marker,
            customdata=customdata,
            hovertemplate=(
                "Community C%{customdata[4]}<br>Units: %{customdata[0]}<br>"
                "Mean activation: %{customdata[1]:.4g}<br>"
                "Mean contribution: %{customdata[2]:.4g}<br>"
                "First unit indices: %{customdata[3]}<extra></extra>"
            ),
            name="Activation community",
        )
    )
    figure.update_xaxes(visible=False, range=[-1.45, 1.45], constrain="domain")
    figure.update_yaxes(visible=False, range=[-1.45, 1.45], scaleanchor="x", scaleratio=1)
    figure.update_layout(showlegend=False, margin={"l": 40, "r": 40, "t": 90, "b": 40})

    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation=(
                "Units z-normalised across the captured batch before deterministic farthest-first "
                "k-means; node marker uses Plotly area sizing"
            ),
            key_results=(
                PublicationResult(
                    "Largest latent community",
                    f"C{communities[largest_community].community_id}",
                    f"Contains {int(sizes[largest_community])} compressed units.",
                ),
                PublicationResult(
                    "Strongest retained similarity",
                    (
                        "No edge retained"
                        if strongest_edge is None
                        else f"{strongest_edge[2]:.3f}"
                    ),
                    (
                        f"No centroid pair exceeded threshold {similarity_threshold:.2f}."
                        if strongest_edge is None
                        else (
                            f"C{communities[strongest_edge[0]].community_id} ↔ "
                            f"C{communities[strongest_edge[1]].community_id}."
                        )
                    ),
                ),
                PublicationResult(
                    "Resolved graph",
                    f"{count} nodes · {len(edge_records)} edges",
                    f"Edges retain cosine similarity at or above {similarity_threshold:.2f}.",
                ),
            ),
            extra={
                "layer": selected.name,
                "layer_type": selected.layer_type,
                "requested_clusters": n_clusters,
                "resolved_clusters": count,
                "similarity_threshold": similarity_threshold,
                "rendered_edges": len(edge_records),
                "community_semantics": "latent_activation_pattern_group",
                "edge_semantics": "positive_centroid_cosine_similarity",
                "causal": False,
            },
        ),
        mode=mode,
        title="VRAVEN Activation Community Graph",
        subtitle=(
            f"{selected.name}: linked unit communities by activation-pattern similarity "
            f"({len(edge_records)} edges ≥ {similarity_threshold:.2f})"
        ),
        width=980,
        height=720,
        show_legend=False,
        show_title=show_title,
    )
