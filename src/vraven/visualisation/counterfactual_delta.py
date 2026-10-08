"""Counterfactual explanation UI: summary cards, change table and value diff."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.causal.counterfactual import CounterfactualResult
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import readable_feature_name
from vraven.visualisation.palette import RenderMode, semantic_colour
from vraven.visualisation.theme import get_theme

CONTRACT = VisualContract(
    visual_id="vraven.counterfactual_delta",
    research_question="Which input values changed most in a constrained target counterfactual?",
    required_metrics=("original value", "counterfactual value", "delta"),
    data_domain="top absolute continuous-input changes",
    visual_encodings=(
        "summary cards=original and target predictions",
        "table rows=largest changed features",
        "dumbbell=original-to-counterfactual values",
    ),
    limitations=(
        "Generic continuous counterfactuals do not guarantee domain validity.",
        "Only the largest absolute changes are displayed.",
    ),
    evidence_tier="behavioural",
    interpretation=(
        "A successful result is a model-verified input change under the optimiser constraints; "
        "it is not automatically feasible, semantic or safe in the application domain."
    ),
    metric_definitions=(
        "delta = counterfactual input - original input on displayed coordinates",
        "displayed L1 = sum of absolute displayed deltas",
        "displayed L2 = square root of the sum of squared displayed deltas",
    ),
)


def counterfactual_delta_figure(
    result: CounterfactualResult,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    theme = get_theme(mode)
    indices = np.asarray(result.top_changed_indices)
    original = np.asarray(result.original_values, dtype=np.float64)
    counterfactual = np.asarray(result.counterfactual_values, dtype=np.float64)
    delta = np.asarray(result.delta_values, dtype=np.float64)
    order = np.argsort(np.abs(delta))
    indices, original, counterfactual, delta = (
        indices[order],
        original[order],
        counterfactual[order],
        delta[order],
    )

    def _name(index: int) -> str:
        if feature_names is not None and 0 <= index < len(feature_names):
            return readable_feature_name(feature_names[index], max_length=34)
        return f"Feature {index}"

    labels = [_name(int(index)) for index in indices]
    original_class = (
        class_names[result.original_prediction]
        if class_names and 0 <= result.original_prediction < len(class_names)
        else str(result.original_prediction)
    )
    target_class = (
        class_names[result.target_class]
        if class_names and 0 <= result.target_class < len(class_names)
        else str(result.target_class)
    )
    cf_class = (
        class_names[result.counterfactual_prediction]
        if class_names and 0 <= result.counterfactual_prediction < len(class_names)
        else str(result.counterfactual_prediction)
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.14,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
    )
    figure.add_trace(
        go.Table(
            header={
                "values": [
                    "Original prediction",
                    "Counterfactual prediction",
                    "Target class",
                    "Success",
                ],
                "fill_color": semantic_colour("neutral", mode),
                "font": {"color": "white", "size": 12},
                "align": "left",
            },
            cells={
                "values": [
                    [f"{original_class} (score {result.original_target_score:.4g})"],
                    [f"{cf_class} (score {result.counterfactual_target_score:.4g})"],
                    [target_class],
                    ["yes" if result.success else "no"],
                ],
                "fill_color": [
                    [semantic_colour("opposing", mode)],
                    [semantic_colour("supporting", mode)],
                    [theme.raised_panel],
                    [theme.raised_panel],
                ],
                "font": {"color": theme.primary_text, "size": 12},
                "align": "left",
                "height": 34,
            },
        ),
        row=1,
        col=1,
    )

    for label, first, second in zip(labels, original, counterfactual, strict=True):
        figure.add_shape(
            type="line",
            x0=float(first),
            x1=float(second),
            y0=label,
            y1=label,
            xref="x",
            yref="y",
            line={"color": semantic_colour("neutral", mode), "width": 1.2},
            layer="below",
        )
    figure.add_trace(
        go.Scatter(
            x=original,
            y=labels,
            mode="markers",
            name="Original",
            marker={"size": 8, "color": semantic_colour("opposing", mode), "symbol": "circle-open"},
            hovertemplate="%{y}<br>Original: %{x:.5g}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=counterfactual,
            y=labels,
            mode="markers",
            name="Counterfactual",
            marker={"size": 9, "color": semantic_colour("supporting", mode), "symbol": "diamond"},
            customdata=delta,
            hovertemplate="%{y}<br>Counterfactual: %{x:.5g}<br>Delta: %{customdata:.5g}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Input value", row=2, col=1, title_standoff=16)
    figure.update_yaxes(title_text="Top changes", row=2, col=1, automargin=True)
    figure.update_layout(margin={"l": 170, "r": 40, "b": 80, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="none",
            key_results=(
                PublicationResult(
                    "Target reached",
                    "Yes" if result.success else "No",
                    f"Model verification after {result.steps_completed} optimisation steps.",
                ),
                PublicationResult(
                    "Displayed L1 change",
                    f"{float(np.sum(np.abs(delta))):.5g}",
                    "Sum of absolute changes over the displayed top coordinates only.",
                    "input units",
                ),
                PublicationResult(
                    "Displayed L2 change",
                    f"{float(np.linalg.norm(delta)):.5g}",
                    "Euclidean norm over the displayed top coordinates only.",
                    "input units",
                ),
            ),
            extra={
                "target_class": result.target_class,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "original_prediction": result.original_prediction,
                "counterfactual_prediction": result.counterfactual_prediction,
            },
        ),
        mode=mode,
        title="VRAVEN Counterfactual Explanation",
        subtitle=(
            f"Target {target_class}; "
            f"{'target reached' if result.success else 'target not reached'} "
            f"in {result.steps_completed} steps"
        ),
        height=max(640, 220 + 30 * len(labels)),
        show_title=show_title,
    )
