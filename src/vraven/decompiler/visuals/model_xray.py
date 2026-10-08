"""Full-stack model X-ray for the captured VRAVEN mechanism IR."""

from __future__ import annotations

import math
from collections import defaultdict

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from ..ir import MechanismRole, NodeKind
from .common import coverage_badge_band, rgba, safe_width, scientific_figure

_ROLE_LANES = {
    MechanismRole.INPUT: 6.0,
    MechanismRole.RETRIEVE: 5.1,
    MechanismRole.DETECT: 4.2,
    MechanismRole.TRANSFORM: 4.2,
    MechanismRole.NORMALISE: 3.4,
    MechanismRole.GATE: 2.6,
    MechanismRole.SUPPRESS: 2.6,
    MechanismRole.POOL: 2.6,
    MechanismRole.MERGE: 1.8,
    MechanismRole.ROUTE: 1.8,
    MechanismRole.STORE: 1.0,
    MechanismRole.REPEAT: 1.0,
    MechanismRole.ATTEND: 1.0,
    MechanismRole.ACCUMULATE: 0.2,
    MechanismRole.COMPARE: 0.2,
    MechanismRole.READOUT: -0.6,
    MechanismRole.OUTPUT: -1.4,
    MechanismRole.SHAPE: 5.1,
    MechanismRole.RANDOM: 3.4,
    MechanismRole.UNKNOWN: -2.2,
}

_ROLE_LABELS = {
    6.0: "Input",
    5.1: "State / shape",
    4.2: "Feature transformation",
    3.4: "Normalisation",
    2.6: "Gating / pooling",
    1.8: "Merge / routing",
    1.0: "Memory / attention",
    0.2: "Evidence accumulation",
    -0.6: "Readout",
    -1.4: "Output",
    -2.2: "Unresolved",
}


def model_xray(result: DecompilationResult, *, mode: RenderMode = "interactive_dark") -> go.Figure:
    theme = get_theme(mode)
    ir = result.captured.ir
    colours = theme.semantic
    topology = ir.topological_index()
    denominator = max(1, len(ir.nodes) - 1)
    lane_counts: dict[float, int] = defaultdict(int)
    positions: dict[str, tuple[float, float]] = {}
    for node in ir.nodes:
        x = topology[node.id] / denominator
        lane = _ROLE_LANES.get(node.role, -2.2)
        offset = ((lane_counts[lane] % 5) - 2) * 0.08
        lane_counts[lane] += 1
        positions[node.id] = (x, lane + offset)

    figure = go.Figure()
    for edge in ir.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        volume = edge.tensor.numel if edge.tensor and edge.tensor.numel else 1
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={
                    "color": rgba(colours["neutral"], 0.22 if mode == "interactive_dark" else 0.33),
                    "width": safe_width(math.log1p(volume), minimum=0.45, maximum=3.4),
                    "shape": "spline",
                    "smoothing": 0.65,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for recognised, label in ((True, "Semantically recognised"), (False, "Unresolved computation")):
        nodes = [node for node in ir.nodes if node.semantically_recognised is recognised]
        if not nodes:
            continue
        x = [positions[node.id][0] for node in nodes]
        y = [positions[node.id][1] for node in nodes]
        sizes = [
            12
            + 5
            * math.log1p(
                max(
                    node.parameter_count,
                    node.tensor.numel if node.tensor and node.tensor.numel else 1,
                )
            )
            for node in nodes
        ]
        fill = [
            colours["unknown"] if not recognised else _role_colour(node.role, colours)
            for node in nodes
        ]
        border_width = [3.4 if node.intervention_capable else 1.0 for node in nodes]
        symbols = [_node_symbol(node.kind) for node in nodes]
        text = [
            "<b>{}</b><br>role: {}<br>operation: {}<br>shape: {}<br>parameters: {:,}<br>"
            "gradient: {}<br>intervention: {}<br>source: {}".format(
                node.name,
                node.role.value,
                node.target,
                tuple(node.tensor.shape) if node.tensor else "not recorded",
                node.parameter_count,
                "available" if node.differentiable else "not established",
                "available" if node.intervention_capable else "not available",
                node.source.module_path or node.source.operator_source or "unknown",
            )
            for node in nodes
        ]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=label,
                marker={
                    "size": sizes,
                    "sizemode": "diameter",
                    "color": fill,
                    "symbol": symbols,
                    "opacity": 0.92,
                    "line": {"color": colours["selected"], "width": border_width},
                },
                text=text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for lane, label in _ROLE_LABELS.items():
        figure.add_hline(
            y=lane,
            line={"color": rgba(theme.border, 0.52), "width": 0.7, "dash": "dot"},
        )
        # Place lane names in the left paper gutter (outside the plot domain).
        figure.add_annotation(
            x=0.155,
            y=lane,
            xref="paper",
            yref="y",
            text=label,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            align="right",
            bgcolor=rgba(theme.canvas, 0.96),
            borderpad=3,
            font={"size": theme.note_size + 1, "color": theme.secondary_text},
        )

    coverage = ir.coverage
    audit = reverse_feature_audit(result, "model_xray")
    coverage_badge_band(
        figure,
        [
            ("Operator semantics", coverage.operator_coverage, colours["latent"]),
            ("Intervention access", coverage.intervention_coverage, colours["selected"]),
            ("Gradient routes", coverage.gradient_coverage, colours["uncertainty"]),
            ("Parameters", coverage.parameter_coverage, colours["stable"]),
        ],
        mode=mode,
        y=1.0,
        x_start=0.20,
        x_step=0.19,
    )

    figure.update_xaxes(
        title="Normalised execution depth",
        range=[-0.02, 1.03],
        tickformat=".1f",
        domain=[0.17, 1.0],
    )
    figure.update_yaxes(
        showticklabels=False,
        title="",
        range=[-2.75, 6.55],
        gridcolor=rgba(theme.grid, 0.28),
        domain=[0.0, 1.0],
    )
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_model_xray",
        question="What computation was recovered from this previously unseen model, and what remains unresolved?",
        metrics=(
            "operator coverage",
            "gradient coverage",
            "intervention coverage",
            "tensor volume",
            "parameter count",
        ),
        encodings=(
            "horizontal position = execution depth",
            "vertical lane = mechanism role",
            "node area = logarithmic state size",
            "border width = intervention accessibility",
            "grey nodes = unresolved computation",
        ),
        limitations=(
            "Capture covers observed routes and declared guards only.",
            "Node-level semantics are operator semantics, not discovered domain concepts.",
            "FX graph layout is deterministic but not a physical brain topology.",
        ),
        title="Raven Eye · Full-stack model X-ray",
        subtitle=f"{ir.model_class} · {ir.capture_route.value} · {len(ir.nodes)} nodes · numerical equivalence: {coverage.numeric_equivalence_passed}",
        width=1360,
        height=940,
        show_legend=True,
        extra={"feature_audit": audit.to_dict()},
    )


def _role_colour(role: MechanismRole, colours: dict[str, str]) -> str:
    if role in {MechanismRole.DETECT, MechanismRole.TRANSFORM, MechanismRole.ATTEND}:
        return colours["latent"]
    if role in {MechanismRole.GATE, MechanismRole.SUPPRESS, MechanismRole.ROUTE}:
        return colours["uncertainty"]
    if role in {MechanismRole.MERGE, MechanismRole.ACCUMULATE, MechanismRole.COMPARE}:
        return colours["supporting"]
    if role in {MechanismRole.READOUT, MechanismRole.OUTPUT}:
        return colours["opposing"]
    if role in {MechanismRole.INPUT, MechanismRole.RETRIEVE, MechanismRole.SHAPE}:
        return colours["stable"]
    return colours["neutral"]


def _node_symbol(kind: NodeKind) -> str:
    return {
        NodeKind.INPUT: "diamond",
        NodeKind.PARAMETER: "hexagon",
        NodeKind.BUFFER: "hexagon2",
        NodeKind.CONTROL_FLOW: "triangle-right",
        NodeKind.OUTPUT: "star",
        NodeKind.UNRESOLVED: "x",
    }.get(kind, "circle")
