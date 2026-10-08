"""Paired programme and concept view for counterfactual verification."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from .common import rgba, scientific_figure


def counterfactual_twin(
    result: DecompilationResult, *, mode: RenderMode = "interactive_dark"
) -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    proof = result.counterfactual
    audit = reverse_feature_audit(result, "counterfactual_twin")
    figure = go.Figure()

    if proof is None or proof.selected is None:
        _render_absent(figure, result, mode)
        return scientific_figure(
            figure,
            mode=mode,
            visual_id="vraven_counterfactual_twin",
            question="Was a programme-predicted concept change verified to alter the original model decision?",
            metrics=("programme candidates", "verified graph interventions", "search budget"),
            encodings=(
                "search stages shown as proof gates",
                "amber = incomplete proof",
                "grey = unverified domain validity",
            ),
            limitations=(
                "No result within a bounded search is not proof that no counterfactual exists.",
            ),
            title="Raven Shadow · Counterfactual proof",
            subtitle="No verified concept-level counterfactual was found within the declared search budget",
            width=1240,
            height=680,
            show_legend=False,
            extra={"feature_audit": audit.to_dict()},
        )

    candidate = proof.selected
    left_x, right_x = 1.6, 8.4
    centre_x = 5.0
    change_count = len(candidate.changes)
    y_positions = [change_count - index for index in range(change_count)]

    # Programme paths.
    for side_x, margin, prediction, label, colour in (
        (
            left_x,
            candidate.model_margin_before
            if candidate.model_margin_before is not None
            else candidate.programme_margin_before,
            candidate.model_prediction_before,
            "Original mechanism",
            colours["supporting"],
        ),
        (
            right_x,
            candidate.model_margin_after
            if candidate.model_margin_after is not None
            else candidate.programme_margin_after,
            candidate.model_prediction_after,
            "Counterfactual mechanism",
            colours["opposing"] if candidate.model_flip else colours["uncertainty"],
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=[side_x, side_x],
                y=[0.4, change_count + 1.15],
                mode="lines",
                line={"color": rgba(colour, 0.48), "width": 6},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[side_x],
                y=[0.15],
                mode="markers+text",
                marker={
                    "size": 58,
                    "symbol": "star",
                    "color": colour,
                    "line": {"color": theme.primary_text, "width": 1.4},
                },
                text=[
                    result.class_names.get(prediction, f"class {prediction}")
                    if prediction is not None
                    else "unknown"
                ],
                textposition="bottom center",
                textfont={"size": theme.label_size + 1, "color": theme.primary_text},
                hovertemplate=f"{label}<br>margin: {margin:+.4f}<extra></extra>",
                showlegend=False,
            )
        )
        figure.add_annotation(
            x=side_x,
            y=change_count + 1.35,
            text=f"<b>{label}</b><br>margin {margin:+.4f}",
            showarrow=False,
            font={"size": theme.label_size + 1, "color": theme.primary_text},
        )

    for change, y in zip(candidate.changes, y_positions, strict=True):
        # Original and replacement nodes.
        for x, value, colour, suffix in (
            (left_x, change.original_value, colours["latent"], "observed"),
            (right_x, change.replacement_value, colours["selected"], "intervened"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker={
                        "size": 42,
                        "symbol": "hexagon",
                        "color": rgba(colour, 0.86),
                        "line": {"color": theme.primary_text, "width": 1.2},
                    },
                    text=[f"{change.concept_id}<br>{value:+.3f}"],
                    textposition="middle center",
                    textfont={"size": theme.note_size, "color": theme.primary_text},
                    hovertemplate=(
                        f"<b>{change.label}</b><br>{suffix} concept value: {value:+.6f}<br>"
                        f"normalised cost: {change.normalised_cost:.4f}<br>"
                        f"programme contribution before: {change.programme_contribution_before:+.4f}<br>"
                        f"programme contribution after: {change.programme_contribution_after:+.4f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[y - 0.25, y - 0.55],
                    mode="lines",
                    line={"color": rgba(colour, 0.55), "width": 2.2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        # Change bridge.
        bridge_colour = (
            colours["uncertainty"] if change.semantic_status == "unresolved" else colours["stable"]
        )
        figure.add_trace(
            go.Scatter(
                x=[left_x + 0.45, centre_x, right_x - 0.45],
                y=[y, y + 0.16, y],
                mode="lines+markers",
                line={"color": bridge_colour, "width": 2.7, "dash": "dash", "shape": "spline"},
                marker={"size": [0, 9, 0], "symbol": "diamond", "color": bridge_colour},
                text=[
                    None,
                    f"Δ={change.replacement_value - change.original_value:+.3f}<br>cost={change.normalised_cost:.3f}",
                    None,
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        figure.add_annotation(
            x=centre_x,
            y=y - 0.34,
            text=change.label,
            showarrow=False,
            font={"size": theme.note_size, "color": theme.secondary_text},
        )

    status_colour = colours["stable"] if candidate.verified else colours["uncertainty"]
    status_bg = (
        rgba(colours["stable"], 0.15) if candidate.verified else rgba(colours["uncertainty"], 0.15)
    )

    # Plotly annotations only support a small HTML subset (<b>, <i>, <br>, <span>).
    # Full <div style=...> leaks CSS into the visible text.
    figure.add_annotation(
        x=centre_x,
        y=change_count + 1.45,
        text=(
            f"<b style='color:{status_colour}'>VERIFIED COUNTERFACTUAL PROOF</b><br>"
            f"programme flip: <b>{candidate.programme_flip}</b> · "
            f"model flip: <b>{candidate.model_flip}</b> · "
            f"domain valid: <b>{candidate.domain_valid}</b><br>"
            f"total normalised cost: <b>{candidate.total_cost:.3f}</b>"
        ),
        showarrow=False,
        align="center",
        bgcolor=status_bg,
        bordercolor=status_colour,
        borderwidth=1.8,
        borderpad=14,
        font={
            "size": theme.note_size + 1,
            "color": theme.primary_text,
            "family": "JetBrains Mono, monospace",
        },
    )
    figure.add_annotation(
        x=centre_x,
        y=-0.36,
        text=(
            f"<b>Behavioural proof:</b> prediction changed in the captured graph. · "
            f"<b>Domain proof:</b> {candidate.domain_validation_note}"
        ),
        showarrow=False,
        font={"size": theme.note_size + 1, "color": theme.secondary_text},
        align="center",
    )

    figure.update_xaxes(range=[0.1, 9.9], visible=False)
    figure.update_yaxes(range=[-0.7, change_count + 1.9], visible=False)
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_counterfactual_twin",
        question="What compact concept change reverses the decision, and does the original model change in the predicted place?",
        metrics=(
            "concept replacement",
            "normalised cost",
            "programme margin",
            "model margin",
            "prediction flip",
            "domain validity",
        ),
        encodings=(
            "left = observed mechanism",
            "right = intervened mechanism",
            "bridge = concept change",
            "star = final decision",
            "amber = unresolved semantic or validity evidence",
        ),
        limitations=(
            "Internal concept changes may not be realisable in raw input space.",
            "The search is bounded and does not prove global minimality.",
            "Domain validity must be supplied by the user or a separate domain model.",
        ),
        title="Raven Shadow · Counterfactual twin",
        subtitle=f"{len(candidate.changes)} concept changes · model flip {candidate.model_flip} · domain validity {candidate.domain_valid}",
        width=1240,
        height=max(680, 220 + 120 * change_count),
        show_legend=False,
        extra={"feature_audit": audit.to_dict()},
    )


def _render_absent(figure: go.Figure, result: DecompilationResult, mode: RenderMode) -> None:
    theme = get_theme(mode)
    colours = theme.semantic
    proof = result.counterfactual
    if proof is None:
        stages = [
            ("Programme screen", False, "Candidate screening against rules"),
            ("Graph intervention", False, "Causal do-intervention in graph"),
            ("Prediction flip", False, "Decision boundary reversal"),
            ("Domain validation", False, "Domain validity constraint check"),
        ]
        tested = flip_candidates = attempts = failures = verified = 0
    else:
        flip_candidates = getattr(proof, "programme_flip_candidates", 0)
        attempts = getattr(proof, "verification_attempts", 0)
        failures = getattr(proof, "verification_failures", 0)
        stages = [
            (
                "Programme screen",
                flip_candidates > 0,
                "Programme boundary-crossing candidate found",
            ),
            (
                "Graph intervention",
                proof.candidates_verified_in_model > 0,
                "Causal do-intervention in graph",
            ),
            ("Prediction flip", proof.behaviourally_verified, "Decision boundary reversal"),
            ("Domain validation", proof.domain_verified, "Domain validity constraint check"),
        ]
        tested = proof.candidates_tested_by_programme
        verified = proof.candidates_verified_in_model

    for index, (label, passed, desc) in enumerate(stages):
        x = 1.35 + index * 2.45
        colour = (
            colours["stable"]
            if passed
            else colours["selected"]
        )
        status_lbl = "✓ PASSED" if passed else "UNVERIFIED"
        status_colour = colours["stable"] if passed else colours["neutral"]
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[0.95],
                mode="markers+text",
                marker={
                    "size": 58,
                    "symbol": "hexagon",
                    "color": rgba(colour, 0.25 if mode == "interactive_dark" else 0.15),
                    "line": {"color": colour, "width": 2.5},
                },
                text=[f"<b>{label}</b><br><span style='color:{status_colour}'>{status_lbl}</span>"],
                textposition="bottom center",
                textfont={
                    "size": theme.label_size,
                    "color": theme.primary_text,
                    "family": "Plus Jakarta Sans, sans-serif",
                },
                hovertemplate=f"<b>{label}</b><br>{desc}<br>Status: {'Passed' if passed else 'Not Established'}<extra></extra>",
                showlegend=False,
            )
        )
        if index < len(stages) - 1:
            next_passed = stages[index + 1][1]
            line_colour = colours["stable"] if (passed and next_passed) else colours["selected"]
            figure.add_trace(
                go.Scatter(
                    x=[x + 0.55, x + 1.90],
                    y=[0.95, 0.95],
                    mode="lines",
                    line={"color": rgba(line_colour, 0.6), "width": 3.5, "dash": "dash"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    border_col = colours["selected"]
    bg_col = rgba(theme.raised_panel, 0.92 if mode == "interactive_dark" else 0.95)

    # Plotly annotations only support a small HTML subset (<b>, <i>, <br>, <span>).
    # Full <div style=...> leaks CSS into the visible text (see Decision Proof banner bug).
    figure.add_annotation(
        x=4.95,
        y=2.05,
        text=(
            f"<b style='color:{border_col}'>NO VERIFIED PROOF WITHIN SEARCH BUDGET</b><br>"
            f"programme evaluations: <b>{tested:,}</b> · boundary crossings: <b>{flip_candidates:,}</b><br>"
            f"graph attempts: <b>{attempts:,}</b> · successful executions: <b>{verified:,}</b> · failures: <b>{failures:,}</b><br>"
            f"<span style='color:{colours['uncertainty']}'>{proof.claim if proof is not None else 'Counterfactual proof unavailable'}</span>"
        ),
        showarrow=False,
        bgcolor=bg_col,
        bordercolor=border_col,
        borderwidth=1.8,
        borderpad=16,
        align="center",
        font={
            "size": theme.label_size + 1,
            "color": theme.primary_text,
            "family": "JetBrains Mono, monospace",
        },
    )
    figure.update_xaxes(range=[0, 10], visible=False)
    figure.update_yaxes(range=[-0.1, 2.75], visible=False)
