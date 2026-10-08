"""Contrastive causal programme view: why target rather than alternative."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from .common import rgba, scientific_figure


def _wrap_html(text: str, *, width: int = 96) -> str:
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if current and len(trial) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def _display_label(label: str, *, limit: int = 42) -> str:
    cleaned = " ".join(label.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def contrastive_proof(
    result: DecompilationResult, *, mode: RenderMode = "interactive_dark"
) -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    explanation = result.contrastive
    audit = reverse_feature_audit(result, "contrastive_proof")
    figure = go.Figure()
    supporting = explanation.supporting
    opposing = explanation.opposing
    rows = max(len(supporting), len(opposing), 1)

    # Central decision axis — keep clear of the narrative band below.
    figure.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0.35, rows + 1.15],
            mode="lines",
            line={"color": rgba(theme.border, 0.85), "width": 3.5},
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[0.55],
            mode="markers+text",
            marker={
                "size": 48,
                "symbol": "star",
                "color": colours["selected"],
                "line": {"color": theme.primary_text, "width": 1.5},
            },
            text=[explanation.target_label],
            textposition="top center",
            textfont={"size": theme.label_size + 1, "color": theme.primary_text},
            hovertemplate=(
                f"Reconstructed target margin: {explanation.programme_margin:+.4f}<extra></extra>"
            ),
            showlegend=False,
            cliponaxis=False,
        )
    )

    for index in range(rows):
        y = rows - index
        if index < len(supporting):
            item = supporting[index]
            width = min(4.2, 0.7 + abs(item.contribution) * 2.0)
            label = _display_label(item.label)
            figure.add_trace(
                go.Scatter(
                    x=[-0.25, -2.15, -4.6],
                    y=[y, y, y],
                    mode="lines+markers",
                    line={"color": colours["supporting"], "width": width, "shape": "spline"},
                    marker={
                        "size": [5, 18, 30],
                        "symbol": ["circle", "diamond", "hexagon"],
                        "color": colours["supporting"],
                        "line": {"color": theme.canvas, "width": 1.2},
                    },
                    hovertemplate=(
                        f"<b>{item.label}</b><br>supports {explanation.target_label}<br>"
                        f"programme contribution: {item.contribution:+.6f}<br>"
                        f"depth: {item.depth:.3f}<br>evidence tier: {item.evidence_tier}<extra></extra>"
                    ),
                    showlegend=False,
                    cliponaxis=False,
                )
            )
            figure.add_annotation(
                x=-2.15,
                y=y,
                text=f"{item.contribution:+.3f}",
                showarrow=False,
                yshift=14,
                font={"size": theme.note_size + 1, "color": theme.primary_text},
            )
            # Outer label: right edge at tip, text extends left into reserved side band.
            figure.add_annotation(
                x=-4.75,
                y=y,
                text=label,
                showarrow=False,
                xanchor="right",
                align="right",
                font={"size": theme.note_size + 1, "color": theme.primary_text},
            )
        if index < len(opposing):
            item = opposing[index]
            width = min(4.2, 0.7 + abs(item.contribution) * 2.0)
            label = _display_label(item.label)
            figure.add_trace(
                go.Scatter(
                    x=[0.25, 2.15, 4.6],
                    y=[y, y, y],
                    mode="lines+markers",
                    line={"color": colours["opposing"], "width": width, "shape": "spline"},
                    marker={
                        "size": [5, 18, 30],
                        "symbol": ["circle", "diamond", "hexagon"],
                        "color": colours["opposing"],
                        "line": {"color": theme.canvas, "width": 1.2},
                    },
                    hovertemplate=(
                        f"<b>{item.label}</b><br>supports {explanation.alternative_label}<br>"
                        f"programme contribution: {item.contribution:+.6f}<br>"
                        f"depth: {item.depth:.3f}<br>evidence tier: {item.evidence_tier}<extra></extra>"
                    ),
                    showlegend=False,
                    cliponaxis=False,
                )
            )
            figure.add_annotation(
                x=2.15,
                y=y,
                text=f"{item.contribution:+.3f}",
                showarrow=False,
                yshift=14,
                font={"size": theme.note_size + 1, "color": theme.primary_text},
            )
            # Outer label: left edge at tip, text extends right into reserved side band.
            figure.add_annotation(
                x=4.75,
                y=y,
                text=label,
                showarrow=False,
                xanchor="left",
                align="left",
                font={"size": theme.note_size + 1, "color": theme.primary_text},
            )

    figure.add_annotation(
        x=-4.75,
        y=rows + 0.85,
        text=(
            f"<b>Why {explanation.target_label}</b><br>"
            f"<span style='font-size:{theme.note_size}px'>supporting programme evidence</span>"
        ),
        showarrow=False,
        xanchor="left",
        align="left",
        font={"size": theme.label_size + 1, "color": colours["supporting"]},
    )
    figure.add_annotation(
        x=4.75,
        y=rows + 0.85,
        text=(
            f"<b>Why not {explanation.alternative_label}</b><br>"
            f"<span style='font-size:{theme.note_size}px'>opposing programme evidence</span>"
        ),
        showarrow=False,
        xanchor="right",
        align="right",
        font={"size": theme.label_size + 1, "color": colours["opposing"]},
    )
    figure.add_annotation(
        x=0,
        y=rows + 0.82,
        text=(
            f"<b>Contrastive margin</b><br>{explanation.programme_margin:+.4f}<br>"
            f"decisive depth "
            f"{explanation.decisive_depth if explanation.decisive_depth is not None else 'not established'}"
        ),
        showarrow=False,
        align="center",
        bgcolor=rgba(colours["selected"], 0.12),
        bordercolor=colours["selected"],
        borderwidth=1.4,
        borderpad=9,
        font={"size": theme.note_size + 1, "color": theme.primary_text},
    )
    # Narrative sits in paper space under the plot so it never collides with the star.
    figure.add_annotation(
        x=0.5,
        y=0.0,
        xref="paper",
        yref="paper",
        text=_wrap_html(explanation.plain_english, width=100),
        showarrow=False,
        align="left",
        xanchor="center",
        yanchor="top",
        yshift=-8,
        font={"size": theme.note_size + 1, "color": theme.secondary_text},
        bgcolor=rgba(theme.raised_panel, 0.92),
        bordercolor=theme.border,
        borderwidth=1,
        borderpad=10,
    )
    # Wide data range so outer labels sit fully inside the plot box.
    figure.update_xaxes(range=[-9.8, 9.8], visible=False, automargin=False)
    figure.update_yaxes(range=[0.2, rows + 1.45], visible=False, automargin=False)
    figure = scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_contrastive_proof",
        question="Why did the executable programme prefer the selected class over its strongest alternative?",
        metrics=("rule contribution", "programme margin", "decisive depth", "evidence tier"),
        encodings=(
            "left = evidence supporting selected class",
            "right = evidence supporting alternative",
            "line width = contribution magnitude",
            "central star = selected decision",
        ),
        limitations=(
            "Contrastive evidence is only as faithful as the extracted programme.",
            "Rule language uses grounded labels where available and technical identifiers otherwise.",
        ),
        title="Raven Contrast · Why this rather than that?",
        subtitle=f"{explanation.target_label} versus {explanation.alternative_label}",
        width=1560,
        height=max(820, 340 + 115 * rows),
        show_legend=False,
        extra={"feature_audit": audit.to_dict()},
    )
    top = (
        figure.layout.margin.t
        if figure.layout.margin and figure.layout.margin.t is not None
        else 96
    )
    # Side gutters + tall bottom band for the wrapped plain-English narrative.
    figure.update_layout(margin={"l": 36, "r": 36, "t": top, "b": 196})
    return figure
