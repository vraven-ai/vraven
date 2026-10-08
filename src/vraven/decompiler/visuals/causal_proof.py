"""Intervention-validated causal mechanism proof graph."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from .common import rgba, scientific_figure


def causal_proof(
    result: DecompilationResult, *, mode: RenderMode = "interactive_dark"
) -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    graph = result.causal_graph
    audit = reverse_feature_audit(result, "causal_proof")
    positions = _layout(graph.nodes)
    programme_concepts = set(result.programme.concept_ids)
    figure = go.Figure()

    for edge in graph.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        support = edge.effect >= 0
        colour = colours["supporting"] if support else colours["opposing"]
        validated = "intervention" in edge.evidence_tier
        width = 1.1 + 7.0 * min(1.0, abs(edge.effect) / max(_max_edge(graph.edges), 1e-12))
        figure.add_trace(
            go.Scatter(
                x=[x0, (x0 + x1) / 2, x1],
                y=[y0, (y0 + y1) / 2 + 0.10 * np.sign(y1 - y0), y1],
                mode="lines",
                line={
                    "color": rgba(colour, 0.35 + 0.55 * edge.sign_stability),
                    "width": width,
                    "dash": "solid" if validated else "dot",
                    "shape": "spline",
                    "smoothing": 0.9,
                },
                text=(
                    f"<b>{edge.source} → {edge.target}</b><br>intervention effect: {edge.effect:+.4f}"
                    f"<br>95% interval: [{edge.effect_ci[0]:+.4f}, {edge.effect_ci[1]:+.4f}]"
                    f"<br>derivative support: {edge.derivative_support if edge.derivative_support is not None else 'not available'}"
                    f"<br>derivative sign agreement: {edge.derivative_sign_agreement}"
                    f"<br>sign stability: {edge.sign_stability:.1%}"
                    f"<br>evidence tier: {edge.evidence_tier}"
                    f"<br>off-manifold distance: {edge.off_manifold_distance:.3f}"
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        if edge.derivative_support is not None:
            figure.add_trace(
                go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    mode="markers",
                    marker={
                        "size": 7,
                        "symbol": "diamond",
                        "color": colours["stable"]
                        if edge.derivative_sign_agreement
                        else colours["uncertainty"],
                        "line": {"width": 1, "color": theme.canvas},
                    },
                    hovertemplate=(
                        f"Derivative screen: {edge.derivative_support:+.4g}<br>"
                        f"Sign agrees with intervention: {edge.derivative_sign_agreement}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    nodes = graph.nodes
    effects = np.asarray([abs(node.output_effect) for node in nodes], dtype=float)
    max_effect = max(float(effects.max()) if effects.size else 0.0, 1e-12)
    x = [positions[node.concept_id][0] for node in nodes]
    y = [positions[node.concept_id][1] for node in nodes]
    sizes = [18 + 48 * np.sqrt(abs(node.output_effect) / max_effect) for node in nodes]
    fills = [
        colours["supporting"] if node.output_effect >= 0 else colours["opposing"] for node in nodes
    ]
    opacity = [0.92 if node.evidence_tier != "observed_only" else 0.42 for node in nodes]
    symbols = ["hexagon" if node.concept_id in programme_concepts else "circle" for node in nodes]
    text = [
        f"<b>{node.concept_id}</b><br>{node.label}<br>depth: {node.depth:.3f}<br>target-margin effect: {node.output_effect:+.4f}"
        f"<br>95% interval: [{node.output_effect_ci[0]:+.4f}, {node.output_effect_ci[1]:+.4f}]<br>responsibility: {node.responsibility:.1%}"
        f"<br>necessity: {node.necessity:.4f}<br>prediction flip: {node.prediction_flip_rate:.1%}<br>sign stability: {node.sign_stability:.1%}"
        f"<br>manifold distance: {node.manifold_distance:.3f}<br>evidence tier: {node.evidence_tier}<br>semantic confidence: {node.semantic_confidence:.2f}"
        for node in nodes
    ]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[node.concept_id for node in nodes],
            textposition="top center",
            textfont={"size": theme.note_size, "color": theme.secondary_text},
            marker={
                "size": sizes,
                "color": fills,
                "opacity": opacity,
                "symbol": symbols,
                "line": {
                    "color": [
                        colours["selected"]
                        if node.concept_id in programme_concepts
                        else colours["latent"]
                        for node in nodes
                    ],
                    "width": [
                        3.4 if node.concept_id in programme_concepts else 1.4 for node in nodes
                    ],
                },
            },
            customdata=text,
            hovertemplate="%{customdata}<extra></extra>",
            name="Concept mechanism",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"symbol": "hexagon", "size": 13, "color": colours["supporting"]},
            name="Retained in executable programme",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"width": 4, "color": colours["supporting"]},
            name="Intervention-supported transfer",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"width": 3, "dash": "dot", "color": colours["neutral"]},
            name="Association / weak evidence",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"symbol": "diamond", "size": 8, "color": colours["stable"]},
            name="Derivative screen agrees",
        )
    )

    figure.add_annotation(
        x=1.015,
        y=0.96,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Target</b><br>{result.class_names.get(result.target, result.target)}<br><br>"
            f"<b>Baseline margin</b><br>{float(np.median(graph.baseline_margin)):+.3f}<br><br>"
            f"<b>Validated edges</b><br>{len(graph.edges)}"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="left",
        bgcolor=rgba(theme.raised_panel, 0.95),
        bordercolor=theme.border,
        borderwidth=1.2,
        borderpad=10,
        font={"size": theme.note_size + 1, "color": theme.primary_text},
    )
    figure.update_xaxes(title="Normalised model depth", range=[-0.04, 1.05])
    figure.update_yaxes(title="Mechanism lane", showticklabels=False, zeroline=False)
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_causal_proof",
        question="Which discovered concepts have intervention-supported effects on downstream concepts and the selected decision?",
        metrics=(
            "target-margin effect",
            "bootstrap interval",
            "sign stability",
            "prediction flip rate",
            "derivative support",
            "off-manifold distance",
        ),
        encodings=(
            "node area = absolute output effect",
            "node colour = supports or opposes target",
            "hexagon = programme concept",
            "edge width = downstream intervention effect",
            "solid edge = intervention supported",
            "diamond = derivative screen",
        ),
        limitations=(
            "Effects are local to the tested cohort and intervention definition.",
            "Concept erasure can create off-manifold states.",
            "A graph may omit alternative mechanisms outside the search budget.",
        ),
        title="Raven Brain · Bidirectional causal proof graph",
        subtitle=f"{len(nodes)} tested concepts · {len(graph.edges)} downstream relations · target {result.class_names.get(result.target, result.target)}",
        width=1280,
        height=820,
        show_legend=True,
        extra={"feature_audit": audit.to_dict()},
    )


def _layout(nodes) -> dict[str, tuple[float, float]]:
    by_depth: dict[float, list] = defaultdict(list)
    for node in nodes:
        by_depth[round(node.depth, 6)].append(node)
    positions = {}
    for _depth, group in sorted(by_depth.items()):
        ordered = sorted(
            group, key=lambda item: (item.output_effect, item.concept_id), reverse=True
        )
        count = len(ordered)
        for index, node in enumerate(ordered):
            positions[node.concept_id] = (node.depth, (count - 1) / 2.0 - index)
    return positions


def _max_edge(edges) -> float:
    return max((abs(edge.effect) for edge in edges), default=1.0)
