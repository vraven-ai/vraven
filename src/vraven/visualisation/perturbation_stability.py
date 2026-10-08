"""Perturbation stability curves."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.causal.perturbation import PerturbationResult
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.perturbation_stability",
    research_question="How stable is the target score under increasing input perturbations?",
    required_metrics=("baseline score", "perturbed score stability"),
    data_domain="controlled continuous-input perturbations",
    visual_encodings=(
        "x=perturbation level",
        "y=stability index in [0, 1]",
        "line colour=perturbation family",
    ),
    limitations=(
        "Stability is not calibrated explanation confidence.",
        "Noise, masking and shuffle are generic probes and may be task-invalid.",
    ),
    evidence_tier="behavioural",
    interpretation=(
        "Lower values mean a larger target-score change under the declared perturbation. "
        "The curves compare stress families but do not certify worst-case robustness."
    ),
    metric_definitions=(
        "stability(x') = 1 - |s_t(x') - s_t(x)| / (|s_t(x)| + epsilon)",
    ),
)

_KIND_COLOUR = {
    # Purple (latent) so noise is not a second red next to the Original baseline.
    "noise": "latent",
    "feature_mask": "stable",
    "shuffle": "supporting",
}
_KIND_LABEL = {
    "noise": "Random noise",
    "feature_mask": "Feature masking",
    "shuffle": "Shuffled inputs",
}
_KIND_WIDTH = {
    "noise": 3.4,
    "feature_mask": 2.4,
    "shuffle": 2.4,
}


def perturbation_stability_figure(
    result: PerturbationResult,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if not result.curves:
        raise ValueError("Perturbation result contains no curves")
    curve_results: list[PublicationResult] = []
    for curve in result.curves:
        minimum_index = min(range(len(curve.stability)), key=curve.stability.__getitem__)
        curve_results.append(
            PublicationResult(
                f"Minimum {_KIND_LABEL.get(curve.kind, curve.kind).lower()} stability",
                f"{float(curve.stability[minimum_index]):.3f}",
                f"Observed at perturbation level {float(curve.levels[minimum_index]):.1%}.",
            )
        )
    figure = go.Figure()
    # Baseline reference: thick red. Noise uses thick purple (see _KIND_COLOUR).
    figure.add_trace(
        go.Scatter(
            x=[0.0, 100.0],
            y=[1.0, 1.0],
            mode="lines",
            name="No score change",
            line={"color": semantic_colour("opposing", mode), "width": 3.6},
        )
    )
    for curve in result.curves:
        role = _KIND_COLOUR.get(curve.kind, "neutral")
        figure.add_trace(
            go.Scatter(
                x=[level * 100.0 for level in curve.levels],
                y=curve.stability,
                mode="lines+markers",
                name=_KIND_LABEL.get(curve.kind, curve.kind),
                line={
                    "color": semantic_colour(role, mode),
                    "width": _KIND_WIDTH.get(curve.kind, 2.4),
                },
                marker={"size": 7 if curve.kind == "noise" else 6},
                hovertemplate=(
                    "Level: %{x:.0f}%<br>Stability: %{y:.3f}<extra>"
                    + _KIND_LABEL.get(curve.kind, curve.kind)
                    + "</extra>"
                ),
            )
        )
    figure.update_xaxes(title_text="Perturbation level (%)", range=[-2, 102], title_standoff=18)
    figure.update_yaxes(
        title_text="Normalised target-score stability",
        range=[-0.05, 1.05],
        title_standoff=18,
    )
    figure.update_layout(
        margin={"b": 90, "l": 100, "r": 170, "t": 90},
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
    )
    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="stability = 1 - |score' - score| / (|score| + eps)",
            random_seed=result.metadata.get("random_seed")
            if isinstance(result.metadata, dict)
            else None,
            key_results=tuple(curve_results),
            extra={
                "baseline_score": result.baseline_score,
                "kinds": [curve.kind for curve in result.curves],
            },
        ),
        mode=mode,
        title="VRAVEN Perturbation Test",
        subtitle="Prediction stability under controlled input perturbations",
        width=1100,
        height=640,
        show_title=show_title,
    )
    # Theme defaults pin the legend under the x-axis; keep it on the right so
    # four series labels stay readable ("Shuffled inputs" must not truncate).
    figure.update_layout(
        margin={"l": 100, "r": 200, "t": figure.layout.margin.t or 150, "b": 90},
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "traceorder": "normal",
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "itemwidth": 30,
            "itemsizing": "constant",
        },
    )
    return figure
