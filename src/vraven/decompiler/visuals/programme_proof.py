"""Executable programme proof tree."""

from __future__ import annotations

import textwrap

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from .common import coverage_badge_band, rgba, scientific_figure


def programme_proof(
    result: DecompilationResult, *, mode: RenderMode = "interactive_dark"
) -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    program = result.programme
    audit = reverse_feature_audit(result, "programme_proof")
    contribution_map = {
        rule_id: float(values[0])
        for rule_id, values in program.contributions(result.focus_concept_values).items()
    }
    rules = sorted(
        program.rules, key=lambda rule: abs(contribution_map.get(rule.id, 0.0)), reverse=True
    )
    figure = go.Figure()
    y_positions = list(range(len(rules), 0, -1))
    cumulative = program.intercept

    # Main programme spine.
    figure.add_trace(
        go.Scatter(
            x=[8.55, 8.55],
            y=[0.4, len(rules) + 0.6],
            mode="lines",
            line={"color": rgba(colours["neutral"], 0.45), "width": 4},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for rule, y in zip(rules, y_positions, strict=True):
        contribution = contribution_map.get(rule.id, 0.0)
        before = cumulative
        cumulative += contribution
        colour = colours["supporting"] if contribution >= 0 else colours["opposing"]
        dash = "solid" if "intervention" in rule.evidence_tier else "dot"
        concept_labels = list(rule.labels)
        # Concept specimens.
        concept_xs = [1.1] if len(concept_labels) == 1 else [0.55, 1.65]
        for x, label in zip(concept_xs, concept_labels, strict=True):
            figure.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker={
                        "size": 25,
                        "symbol": "hexagon" if len(concept_labels) == 1 else "diamond",
                        "color": rgba(colours["latent"], 0.78),
                        "line": {"color": colours["selected"], "width": 1.8},
                    },
                    text=[_short(label, 24)],
                    textposition="bottom center",
                    textfont={"size": theme.note_size, "color": theme.secondary_text},
                    hovertemplate=f"<b>{label}</b><extra></extra>",
                    showlegend=False,
                )
            )
        # Operator gate.
        gate_label = "STANDARDISE" if rule.kind == "main" else "INTERACT"
        figure.add_shape(
            type="rect",
            x0=2.35,
            x1=4.05,
            y0=y - 0.28,
            y1=y + 0.28,
            fillcolor=rgba(colours["latent"], 0.12 if mode == "publication_light" else 0.19),
            line={"color": colours["latent"], "width": 1.4, "dash": dash},
        )
        figure.add_annotation(
            x=3.2,
            y=y,
            text=f"<b>{gate_label}</b><br><span style='font-size:{theme.note_size}px'>μ={rule.term_mean:.3g} · σ={rule.term_scale:.3g}</span>",
            showarrow=False,
            font={"size": theme.note_size + 1, "color": theme.primary_text},
        )
        for x in concept_xs:
            figure.add_trace(
                go.Scatter(
                    x=[x + 0.18, 2.35],
                    y=[y, y],
                    mode="lines",
                    line={"color": rgba(colours["latent"], 0.72), "width": 1.6, "dash": dash},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Coefficient multiplier.
        figure.add_trace(
            go.Scatter(
                x=[4.6],
                y=[y],
                mode="markers+text",
                marker={
                    "size": 34,
                    "symbol": "circle",
                    "color": rgba(colour, 0.18),
                    "line": {"color": colour, "width": 2.3},
                },
                text=[f"× {rule.coefficient:+.3f}"],
                textposition="middle right",
                textfont={"size": theme.note_size + 1, "color": theme.primary_text},
                hovertemplate=(
                    f"<b>{rule.id}</b><br>coefficient: {rule.coefficient:+.6f}<br>"
                    f"focus contribution: {contribution:+.6f}<br>evidence tier: {rule.evidence_tier}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[4.05, 4.42],
                y=[y, y],
                mode="lines",
                line={"color": colour, "width": 2.0, "dash": dash},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Margin increment capsule and programme spine connection.
        figure.add_shape(
            type="rect",
            x0=6.0,
            x1=7.55,
            y0=y - 0.27,
            y1=y + 0.27,
            fillcolor=rgba(colour, 0.15 if mode == "interactive_dark" else 0.08),
            line={"color": colour, "width": 1.7},
        )
        figure.add_annotation(
            x=6.775,
            y=y,
            text=f"<b>{contribution:+.4f}</b><br><span style='font-size:{theme.note_size}px'>{before:+.3f} → {cumulative:+.3f}</span>",
            showarrow=False,
            font={"size": theme.note_size + 1, "color": theme.primary_text},
        )
        figure.add_trace(
            go.Scatter(
                x=[5.1, 6.0],
                y=[y, y],
                mode="lines",
                line={"color": colour, "width": 2.2, "dash": dash},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[7.55, 8.55],
                y=[y, y],
                mode="lines",
                line={"color": colour, "width": 2.2},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Output decision node.
    final_colour = colours["supporting"] if cumulative > 0 else colours["opposing"]
    figure.add_trace(
        go.Scatter(
            x=[8.55],
            y=[0.15],
            mode="markers+text",
            marker={
                "size": 58,
                "symbol": "star",
                "color": rgba(final_colour, 0.88),
                "line": {"color": theme.primary_text, "width": 1.5},
            },
            text=[
                result.class_names.get(result.target, f"class {result.target}")
                if cumulative > 0
                else result.class_names.get(result.alternative, "alternative")
            ],
            textposition="bottom center",
            textfont={"size": theme.label_size + 1, "color": theme.primary_text},
            hovertemplate=f"Reconstructed final margin: {cumulative:+.6f}<extra></extra>",
            showlegend=False,
        )
    )

    fidelity = program.fidelity
    coverage_badge_band(
        figure,
        [
            ("Held-out R²", max(0.0, min(1.0, fidelity.validation_r2)), colours["latent"]),
            ("Decision agreement", fidelity.validation_decision_agreement, colours["stable"]),
            (
                "Intervention sign",
                fidelity.interventional_sign_agreement or 0.0,
                colours["uncertainty"],
            ),
            (
                "Rules retained",
                len(rules) / max(1, len(program.concept_ids) + len(result.causal_graph.edges)),
                colours["selected"],
            ),
        ],
        mode=mode,
        y=1.0,
        x_start=0.01,
        x_step=0.21,
    )

    figure.add_annotation(
        x=9.45,
        y=len(rules) + 0.25,
        text=(
            f"<b>Programme proof</b><br>intercept {program.intercept:+.4f}<br>"
            f"final margin {cumulative:+.4f}<br>rules {len(rules)}<br>"
            f"target {result.class_names.get(result.target, result.target)}"
        ),
        showarrow=False,
        align="left",
        bgcolor=rgba(theme.raised_panel, 0.95),
        bordercolor=theme.border,
        borderwidth=1.2,
        borderpad=10,
        font={"size": theme.note_size + 1, "color": theme.primary_text},
    )
    figure.update_xaxes(range=[-0.1, 10.4], visible=False)
    figure.update_yaxes(range=[-0.65, len(rules) + 1.1], visible=False)
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_programme_proof",
        question="Can a compact executable programme reproduce the model's target margin and tested interventions?",
        metrics=(
            "rule coefficient",
            "focus contribution",
            "observational fidelity",
            "interventional fidelity",
            "decision agreement",
        ),
        encodings=(
            "one horizontal proof line per executable rule",
            "cyan/red = supports/opposes target",
            "solid/dotted = intervention-supported/observational evidence",
            "right-hand spine = cumulative reconstructed margin",
        ),
        limitations=(
            "The programme is a sparse high-level surrogate, not the unique true mechanism.",
            "Interventional fidelity applies only to tested interventions.",
            "Rule semantics inherit uncertainty from concept grounding.",
        ),
        title="Raven Code · Executable decision programme",
        subtitle=f"{len(rules)} retained rules reconstruct target {result.class_names.get(result.target, result.target)}",
        width=1320,
        height=max(720, 130 + 90 * len(rules)),
        show_legend=False,
        extra={"feature_audit": audit.to_dict()},
    )


def _short(value: str, width: int) -> str:
    lines = textwrap.wrap(value, width=width)
    return "<br>".join(lines[:2]) + ("…" if len(lines) > 2 else "")
