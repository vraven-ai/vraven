"""Cumulative evidence trajectory across model depth."""

from __future__ import annotations

import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.metrics.influence import cumulative_evidence
from vraven.reasoning.pathways import decision_transition_layer
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_name, shorten_layer_names
from vraven.visualisation.palette import RenderMode, semantic_colour
from vraven.visualisation.theme import get_theme

CONTRACT = VisualContract(
    visual_id="vraven.evidence_trajectory",
    research_question="Where does cumulative first-order support begin to dominate opposition?",
    required_metrics=("activation_gradient_signed",),
    data_domain="cumulative normalised first-order influence",
    visual_encodings=(
        "x=layer",
        "support line=cumulative positive influence",
        "opposition line=cumulative negative influence",
        "net line=support minus opposition",
    ),
    limitations=(
        "This is evidence trajectory, not layer confidence.",
        "A valid probability trajectory requires trained intermediate probes and is not inferred here.",
    ),
    evidence_tier="observational",
    interpretation=(
        "The crossing and final net value locate where the measured first-order evidence "
        "becomes support-dominant; they do not measure hidden-layer confidence."
    ),
    metric_definitions=(
        "support_l = cumulative positive activation-gradient mass through layer l",
        "opposition_l = cumulative absolute negative activation-gradient mass through layer l",
        "net_l = support_l - opposition_l after global L1 normalisation",
    ),
)


def evidence_trajectory_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    names, support, opposition = cumulative_evidence(trace)
    if not names:
        raise CapabilityError("Evidence Trajectory requires gradient or causal mode")
    net = support - opposition
    # Shorter ticks when the depth axis is dense.
    label_len = 10 if len(names) > 18 else 14 if len(names) > 12 else 20
    short_names = shorten_layer_names(names, max_length=label_len)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=short_names,
            y=support,
            mode="lines+markers",
            name="Support",
            line={"color": semantic_colour("supporting", mode), "width": 2.6},
            marker={"size": 6},
            customdata=list(names),
            hovertemplate="%{customdata}<br>Cumulative support: %{y:.3%}<extra>Support</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=short_names,
            y=opposition,
            mode="lines+markers",
            name="Opposition",
            line={
                "color": semantic_colour("opposing", mode),
                "width": 3.2,
                "dash": "dot",
            },
            marker={"size": 6},
            customdata=list(names),
            hovertemplate="%{customdata}<br>Cumulative opposition: %{y:.3%}<extra>Opposition</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=short_names,
            y=net,
            mode="lines+markers",
            name="Net signal",
            line={
                "color": semantic_colour("latent", mode),
                "width": 3.2,
                "dash": "dot",
            },
            marker={"size": 7},
            customdata=list(names),
            hovertemplate="%{customdata}<br>Net signal: %{y:.3%}<extra>Net signal</extra>",
        )
    )
    transition = decision_transition_layer(trace)
    if transition in names:
        short_transition = shorten_layer_name(transition, max_length=label_len)
        figure.add_shape(
            type="line",
            x0=short_transition,
            x1=short_transition,
            y0=0.0,
            y1=1.0,
            xref="x",
            yref="paper",
            line={
                "width": 1.2,
                "dash": "dash",
                "color": semantic_colour("stable", mode),
            },
        )
        # Keep this label inside the plot (not above paper y=1) so theme
        # does not open a large overhead badge band.
        figure.add_annotation(
            x=short_transition,
            y=0.96,
            xref="x",
            yref="paper",
            text="stable net support",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=10,
            font={"color": semantic_colour("stable", mode), "size": 10},
        )

    dense = len(names) > 14
    figure.update_xaxes(
        title_text="Model depth",
        tickangle=-90 if dense else -40,
        tickfont={"size": 9 if dense else 10},
        title_standoff=12,
        automargin=False,
        dtick=1 if not dense else None,
    )
    if dense:
        # Show every other tick label to avoid a packed diagonal band.
        figure.update_xaxes(
            tickmode="array",
            tickvals=short_names[::2],
            ticktext=short_names[::2],
        )
    figure.update_yaxes(
        title_text="Cumulative share of absolute first-order influence",
        tickformat=".0%",
        title_standoff=22,
        automargin=True,
    )

    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="global L1 normalisation before cumulative summation",
            key_results=(
                PublicationResult(
                    "Stable support transition",
                    transition or "Not detected",
                    "First layer after which the registered net-support rule remains satisfied.",
                ),
                PublicationResult(
                    "Final net evidence",
                    f"{float(net[-1]):.3%}",
                    "Final cumulative supporting mass minus opposing mass.",
                ),
                PublicationResult(
                    "Measured depth",
                    str(len(names)),
                    "Captured layers contributing a comparable signed first-order value.",
                    "layers",
                ),
            ),
            extra={"transition_layer": transition},
        ),
        mode=mode,
        title="VRAVEN Evidence Trajectory",
        subtitle="Cumulative target-specific support and opposition, not hidden-layer confidence",
        width=1200 if mode == "interactive_dark" else 1040,
        height=620 if mode == "interactive_dark" else 560,
        # Default Plotly title + automargin leaves a large empty band; draw
        # the brand heading as a tight paper annotation instead.
        show_title=False,
        show_legend=True,
    )
    theme = get_theme(mode)
    show = True if show_title is None else bool(show_title)
    if show:
        figure.add_annotation(
            text=(
                "<b>VRAVEN Evidence Trajectory</b><br>"
                f"<span style='font-size:{theme.note_size}px;color:{theme.secondary_text}'>"
                "Cumulative target-specific support and opposition, not hidden-layer confidence"
                "</span>"
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            font={"size": theme.title_size, "color": theme.primary_text},
        )
    # Legend on the right so series labels never collide with each other.
    figure.update_layout(
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"size": 12},
            "traceorder": "normal",
            "itemsizing": "constant",
            "itemwidth": 36,
            "tracegroupgap": 8,
        },
        margin={
            "l": 120,
            "r": 320,
            "t": 78 if show else 40,
            "b": 150 if dense else 110,
        },
    )
    return figure
