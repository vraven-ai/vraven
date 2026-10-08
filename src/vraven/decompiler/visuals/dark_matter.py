"""Multi-denominator explanation-coverage iris."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import get_theme

from ..decompiler import DecompilationResult
from ..feature_audit import reverse_feature_audit
from .common import rgba, scientific_figure


def dark_matter(result: DecompilationResult, *, mode: RenderMode = "interactive_dark") -> go.Figure:
    theme = get_theme(mode)
    colours = theme.semantic
    coverage = result.certificate.coverage
    fidelity = result.certificate.fidelity
    audit = reverse_feature_audit(result, "dark_matter")
    figure = go.Figure()

    rings = [
        (
            "Executed operator semantics",
            3.0,
            [coverage.operator_coverage, 1.0 - coverage.operator_coverage],
            [colours["latent"], colours["unknown"]],
            ["Recognised operators", "Unresolved operators"],
        ),
        (
            "Graph intervention access",
            2.0,
            [coverage.intervention_coverage, 1.0 - coverage.intervention_coverage],
            [colours["selected"], colours["neutral"]],
            ["Intervention-accessible", "Intervention-inaccessible"],
        ),
        (
            "Causal responsibility semantics",
            1.0,
            [
                coverage.causal_semantic_coverage,
                coverage.causal_semantic_association_coverage,
                coverage.unresolved_causal_responsibility,
            ],
            [colours["stable"], colours["latent"], colours["uncertainty"]],
            [
                "Causally grounded semantics",
                "Causally associated labels only",
                "Causally important but unresolved",
            ],
        ),
    ]
    for ring_name, radius, fractions, ring_colours, labels in rings:
        start = 0.0
        for fraction, colour, label in zip(fractions, ring_colours, labels, strict=True):
            fraction = max(0.0, min(1.0, float(fraction)))
            width = 360.0 * fraction
            if width <= 0:
                continue
            figure.add_trace(
                go.Barpolar(
                    r=[0.72],
                    theta=[start + width / 2.0],
                    width=[width],
                    base=[radius],
                    marker={"color": colour, "line": {"color": theme.canvas, "width": 2.0}},
                    opacity=0.94,
                    name=label,
                    text=[
                        f"<b>{ring_name}</b><br>{label}: {fraction:.1%}<br>"
                        f"denominator: {_denominator_text(ring_name)}"
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            start += width

    # Fidelity pupil. Independent dimensions are shown as separate spokes rather than averaged.
    fidelity_values = [
        max(0.0, min(1.0, fidelity.observational_decision_agreement)),
        max(0.0, min(1.0, (fidelity.observational_validation_r2 + 1.0) / 2.0)),
        max(0.0, min(1.0, fidelity.interventional_sign_agreement or 0.0)),
        1.0 if fidelity.counterfactual_model_flip else 0.0,
    ]
    fidelity_labels = [
        "Decision agreement",
        "Normalised held-out R²",
        "Intervention sign agreement",
        "Counterfactual model flip",
    ]
    angles = [45, 135, 225, 315]
    figure.add_trace(
        go.Scatterpolar(
            r=fidelity_values + [fidelity_values[0]],
            theta=angles + [angles[0]],
            fill="toself",
            fillcolor=rgba(colours["supporting"], 0.16),
            line={"color": colours["supporting"], "width": 2.2},
            marker={"size": 7, "color": colours["strong_supporting"]},
            text=[
                f"{label}: {value:.1%}"
                for label, value in zip(fidelity_labels, fidelity_values, strict=True)
            ]
            + [f"{fidelity_labels[0]}: {fidelity_values[0]:.1%}"],
            hovertemplate="%{text}<extra></extra>",
            name="Fidelity spokes",
            subplot="polar2",
        )
    )
    figure.update_layout(
        polar={
            "domain": {"x": [0.02, 0.68], "y": [0.02, 0.98]},
            "radialaxis": {"visible": False, "range": [0, 3.85]},
            "angularaxis": {"visible": False, "direction": "clockwise", "rotation": 90},
            "bgcolor": theme.canvas,
        },
        polar2={
            "domain": {"x": [0.72, 0.99], "y": [0.24, 0.76]},
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "tickvals": [0.25, 0.5, 0.75, 1.0],
                "tickformat": ".0%",
                "gridcolor": rgba(theme.grid, 0.65),
                "tickfont": {"size": theme.note_size, "color": theme.secondary_text},
            },
            "angularaxis": {
                "tickvals": angles,
                "ticktext": ["Agreement", "Held-out R²", "Intervention", "Counterfactual"],
                "tickfont": {"size": theme.note_size, "color": theme.secondary_text},
                "gridcolor": rgba(theme.grid, 0.35),
            },
            "bgcolor": theme.canvas,
        },
    )
    figure.add_annotation(
        x=0.35,
        y=0.51,
        xref="paper",
        yref="paper",
        text=(
            f"<b>VRAVEN dark matter</b><br>"
            f"{coverage.unresolved_causal_responsibility:.1%}<br>"
            f"of tested causal responsibility<br>remains semantically unresolved"
        ),
        showarrow=False,
        align="center",
        font={"size": theme.label_size + 2, "color": theme.primary_text},
    )
    figure.add_annotation(
        x=0.855,
        y=0.84,
        xref="paper",
        yref="paper",
        text=f"<b>Mechanism certificate</b><br><span style='font-size:{theme.note_size}px'>Independent fidelity dimensions</span>",
        showarrow=False,
        font={"size": theme.label_size + 1, "color": theme.primary_text},
    )
    figure.add_annotation(
        x=0.855,
        y=0.16,
        xref="paper",
        yref="paper",
        text=(
            f"Concepts {coverage.concepts_total}<br>grounded {coverage.concepts_semantically_grounded}<br>"
            f"associated only {coverage.concepts_semantically_associated}<br>"
            f"causal concepts {coverage.causal_concepts_total}<br>unsupported nodes {coverage.unsupported_nodes}"
        ),
        showarrow=False,
        align="center",
        bgcolor=rgba(theme.raised_panel, 0.9),
        bordercolor=theme.border,
        borderwidth=1,
        borderpad=8,
        font={"size": theme.note_size, "color": theme.secondary_text},
    )
    return scientific_figure(
        figure,
        mode=mode,
        visual_id="vraven_dark_matter",
        question="How much computation, intervention access and causally weighted semantic meaning has VRAVEN recovered, and what remains unexplained?",
        metrics=(
            "operator semantics coverage",
            "intervention coverage",
            "causal semantic coverage",
            "unresolved causal responsibility",
            "programme fidelity",
        ),
        encodings=(
            "outer ring = operator semantics",
            "middle ring = intervention access",
            "inner ring = causally weighted semantic coverage",
            "right iris = independent fidelity dimensions",
        ),
        limitations=(
            "Rings use different denominators and must not be added together.",
            "Semantic coverage depends on the supplied grounding evidence.",
            "Unresolved computation is exposed rather than interpreted through invented labels.",
        ),
        title="Raven Dark Matter · Explanation coverage iris",
        subtitle="Independent coverage dimensions; no opaque combined trust score",
        width=1260,
        height=760,
        show_legend=True,
        extra={"feature_audit": audit.to_dict()},
    )


def _denominator_text(ring_name: str) -> str:
    if ring_name.startswith("Executed"):
        return "executed mechanism operations"
    if ring_name.startswith("Graph"):
        return "candidate internal intervention points"
    return "absolute tested output effect"
