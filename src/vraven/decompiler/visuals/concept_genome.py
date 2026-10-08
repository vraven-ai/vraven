"""Cross-layer concept lineage visualisation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..evolution import EvolutionEvent
from ..feature_audit import reverse_feature_audit
from .common import rgba, scientific_figure


def concept_genome(
    result: DecompilationResult, *, mode: RenderMode = "interactive_dark"
) -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    atlas = result.atlas
    evolution = result.evolution
    audit = reverse_feature_audit(result, "concept_genome")
    positions = _lineage_layout(result)
    figure = go.Figure()

    event_colours = {
        EvolutionEvent.PERSIST: colours["stable"],
        EvolutionEvent.TRANSFORM: colours["latent"],
        EvolutionEvent.SPLIT: colours["uncertainty"],
        EvolutionEvent.MERGE: colours["selected"],
        EvolutionEvent.BIRTH: colours["strong_supporting"],
        EvolutionEvent.DEATH: colours["opposing"],
    }
    for edge in evolution.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        colour = event_colours[edge.event]
        figure.add_trace(
            go.Scatter(
                x=[x0, (x0 + x1) / 2, x1],
                y=[y0, y0 + (y1 - y0) * 0.18, y1],
                mode="lines",
                line={
                    "color": rgba(colour, 0.35 + 0.6 * min(1.0, edge.confidence)),
                    "width": 0.8 + 10.0 * edge.transport_mass,
                    "shape": "spline",
                    "smoothing": 1.05,
                    "dash": "solid" if edge.confidence >= 0.25 else "dot",
                },
                text=(
                    f"<b>{edge.event.value}</b><br>transport mass: {edge.transport_mass:.3f}"
                    f"<br>profile similarity: {edge.similarity:.3f}<br>confidence: {edge.confidence:.3f}"
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    all_concepts = atlas.concepts
    x = [positions[concept.id][0] for concept in all_concepts if concept.id in positions]
    y = [positions[concept.id][1] for concept in all_concepts if concept.id in positions]
    concepts = [concept for concept in all_concepts if concept.id in positions]
    sizes = [14 + 30 * min(1.0, np.sqrt(max(concept.score_variance, 0.0))) for concept in concepts]
    fills = [
        colours["selected"] if concept.semantically_grounded else colours["unknown"]
        for concept in concepts
    ]
    border = [
        colours["stable"] if concept.semantic_confidence >= 0.75 else colours["latent"]
        for concept in concepts
    ]
    text = [
        "<b>{}</b><br>{}<br>depth: {:.3f}<br>variance: {:.4f}<br>sparsity: {:.1%}"
        "<br>class effect: {}<br>semantic confidence: {:.2f}<br>top samples: {}".format(
            concept.id,
            concept.display_label,
            concept.depth,
            concept.score_variance,
            concept.activation_sparsity,
            f"{concept.class_effect_size:+.3f}"
            if concept.class_effect_size is not None
            else "not tested",
            concept.semantic_confidence,
            ", ".join(map(str, concept.top_samples[:6])),
        )
        for concept in concepts
    ]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[concept.id for concept in concepts],
            textposition="top center",
            textfont={"size": theme.note_size, "color": theme.secondary_text},
            marker={
                "size": sizes,
                "color": fills,
                "opacity": 0.94,
                "symbol": [
                    "circle" if concept.semantically_grounded else "circle-open"
                    for concept in concepts
                ],
                "line": {"color": border, "width": 2.0},
            },
            customdata=text,
            hovertemplate="%{customdata}<extra></extra>",
            name="Discovered internal concepts",
        )
    )

    for concept_set in sorted(atlas.concept_sets, key=lambda item: item.depth):
        figure.add_vline(
            x=concept_set.depth,
            line={"color": rgba(theme.border, 0.55), "width": 0.8, "dash": "dot"},
        )
        figure.add_annotation(
            x=concept_set.depth,
            y=1.0,
            xref="x",
            yref="paper",
            text=f"{concept_set.node_name}<br><span style='font-size:{theme.note_size}px'>"
            f"{concept_set.method} · EV {concept_set.explained_variance:.2f}</span>",
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            align="center",
            bgcolor=rgba(theme.panel, 0.92),
            bordercolor=rgba(theme.border, 0.55),
            borderwidth=1,
            borderpad=4,
            font={"size": theme.note_size + 1, "color": theme.secondary_text},
        )

    for event, colour in event_colours.items():
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": colour, "width": 4},
                name=event.value.replace("_", " ").title(),
            )
        )
    figure.update_xaxes(title="Normalised model depth", range=[-0.03, 1.03])
    figure.update_yaxes(
        title="Deterministic lineage position", showticklabels=False, zeroline=False
    )
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_concept_genome",
        question="Where do internal concepts appear, persist, split, merge or disappear through model depth?",
        metrics=(
            "optimal-transport mass",
            "activation-profile similarity",
            "concept variance",
            "semantic confidence",
        ),
        encodings=(
            "x = model depth",
            "line width = transport mass",
            "line colour = lineage event",
            "node area = concept variance",
            "open node = semantically unresolved",
        ),
        limitations=(
            "Optimal-transport lineage is a representation correspondence, not causal proof.",
            "Concept labels require separate grounding evidence.",
            "The y-axis is a deterministic layout coordinate, not a measured quantity.",
        ),
        title="Raven Flight · Concept genome",
        subtitle=f"{len(atlas.concepts)} concepts across {len(atlas.concept_sets)} internal states · {len(evolution.edges)} candidate lineage relations",
        width=1280,
        height=900,
        show_legend=True,
        extra={"feature_audit": audit.to_dict()},
    )


def _lineage_layout(result: DecompilationResult) -> dict[str, tuple[float, float]]:
    concept_sets = sorted(result.atlas.concept_sets, key=lambda item: item.depth)
    incoming: dict[str, list] = defaultdict(list)
    for edge in result.evolution.edges:
        incoming[edge.target].append(edge)
    positions: dict[str, tuple[float, float]] = {}
    for layer_index, concept_set in enumerate(concept_sets):
        concepts = list(concept_set.concepts)
        if layer_index == 0:
            ordering = sorted(concepts, key=lambda item: item.id)
        else:

            def parent_coordinate(concept):
                edges = incoming.get(concept.id, ())
                weighted = [
                    (positions[edge.source][1], edge.transport_mass)
                    for edge in edges
                    if edge.source in positions
                ]
                if not weighted:
                    return float("inf")
                total = sum(weight for _, weight in weighted)
                return sum(y * weight for y, weight in weighted) / max(total, 1e-12)

            ordering = sorted(concepts, key=lambda item: (parent_coordinate(item), item.id))
        count = max(1, len(ordering))
        for index, concept in enumerate(ordering):
            y = (count - 1) / 2.0 - index
            positions[concept.id] = (concept_set.depth, y)
    return positions
