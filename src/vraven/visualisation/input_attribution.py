"""Signed input-space attribution view for continuous inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import readable_feature_name
from vraven.visualisation.palette import RenderMode, semantic_colour, signed_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.input_attribution",
    research_question="Where is signed input-times-gradient influence concentrated in the input space?",
    required_metrics=("input_times_target_gradient",),
    data_domain="signed first-order input attribution",
    visual_encodings=(
        "tabular: bar length=signed or absolute attribution; label=feature name",
        "dense: cell position=input position; colour=signed input-times-gradient value",
    ),
    limitations=(
        "Input-times-gradient is local and may be sensitive to scaling and saturation.",
        "Large inputs are deterministically sampled and displayed in sampled order.",
        "Feature bars summarise the stored attribution vector and are not global SHAP values.",
    ),
    evidence_tier="observational",
    interpretation=(
        "Sign identifies the local direction of the input-times-target-gradient term and "
        "absolute magnitude ranks concentration at this input."
    ),
    metric_definitions=(
        "attribution_i = x_i times partial target_score / partial x_i",
        "displayed mass share = sum displayed |attribution| / sum stored |attribution|",
    ),
)


def _matrix(trace: NeuralTrace) -> tuple[np.ndarray, str]:
    inputs = trace.inputs
    if inputs is None or not inputs.attribution_available or inputs.attribution_values is None:
        raise CapabilityError("Input Attribution requires gradient or causal mode and a floating input")
    values = np.asarray(inputs.attribution_values, dtype=np.float64)
    shape = inputs.attribution_shape or (values.size,)
    if int(np.prod(shape)) == values.size:
        reshaped = values.reshape(shape)
        if reshaped.ndim == 1:
            return reshaped.reshape(1, -1), "exact one-dimensional input"
        if reshaped.ndim == 2:
            return reshaped, "exact two-dimensional input"
        if reshaped.ndim == 3 and reshaped.shape[0] <= 4:
            return reshaped.mean(axis=0), "channel-mean signed attribution"
    width = min(128, max(16, int(math.ceil(math.sqrt(values.size)))))
    rows = int(math.ceil(values.size / width))
    matrix = np.full(rows * width, np.nan)
    matrix[: values.size] = values
    return matrix.reshape(rows, width), "deterministically sampled flattened input"


def _resolve_feature_names(
    trace: NeuralTrace,
    feature_names: Sequence[str] | None,
    width: int,
) -> list[str] | None:
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if len(names) == width:
            return names
    metadata = trace.metadata if isinstance(trace.metadata, dict) else {}
    user = metadata.get("user_metadata")
    if isinstance(user, dict):
        candidate = user.get("feature_names")
        if isinstance(candidate, (list, tuple)) and len(candidate) == width:
            return [str(name) for name in candidate]
    direct = metadata.get("feature_names")
    if isinstance(direct, (list, tuple)) and len(direct) == width:
        return [str(name) for name in direct]
    return None


def _feature_importance_figure(
    values: np.ndarray,
    feature_names: Sequence[str],
    *,
    mode: RenderMode,
    top_k: int | None,
    mapping: str,
    show_title: bool | None,
) -> go.Figure:
    signed = np.asarray(values, dtype=np.float64).reshape(-1)
    if signed.size != len(feature_names):
        raise CapabilityError("feature_names length must match the attribution vector width")
    order = np.argsort(np.abs(signed))[::-1]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        order = order[: min(top_k, order.size)]
    # Smallest abs at the bottom so the strongest features sit at the top.
    order = order[::-1]
    selected = signed[order]
    labels = [readable_feature_name(feature_names[index]) for index in order]
    colours = [
        semantic_colour("supporting", mode) if value >= 0 else semantic_colour("opposing", mode)
        for value in selected
    ]
    strongest = int(np.argmax(np.abs(selected)))
    total_mass = float(np.sum(np.abs(signed)))
    shown_mass = float(np.sum(np.abs(selected)))
    figure = go.Figure(
        go.Bar(
            x=selected,
            y=labels,
            orientation="h",
            marker={"color": colours},
            customdata=[feature_names[index] for index in order],
            hovertemplate=(
                "Feature: %{customdata}<br>"
                "Signed input×gradient: %{x:.5g}<extra></extra>"
            ),
            name="Feature attribution",
        )
    )
    figure.add_vline(x=0.0, line_width=1.0, line_color=semantic_colour("neutral", mode))
    figure.update_xaxes(title_text="Signed input × target-gradient", title_standoff=18)
    figure.update_yaxes(title_text="Input feature", title_standoff=18, automargin=True)
    figure.update_layout(margin={"l": 180, "r": 40, "b": 80, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="bars show stored signed attribution; ordering by absolute magnitude",
            key_results=(
                PublicationResult(
                    "Strongest displayed feature",
                    labels[strongest],
                    f"Signed local attribution {float(selected[strongest]):.5g}.",
                ),
                PublicationResult(
                    "Displayed attribution mass",
                    "Not estimable" if total_mass <= 0 else f"{shown_mass / total_mass:.1%}",
                    "Share of total stored absolute attribution retained by the displayed features.",
                ),
                PublicationResult(
                    "Features displayed",
                    str(int(order.size)),
                    "Features selected by absolute local attribution magnitude.",
                    "features",
                ),
            ),
            extra={
                "display_mapping": mapping,
                "presentation": "named_feature_bars",
                "features_shown": int(order.size),
            },
        ),
        mode=mode,
        title="VRAVEN Local Feature Attribution",
        subtitle="Named signed input-times-gradient attribution for tabular features",
        height=max(560, 26 * int(order.size) + 180),
        width=1100,
        show_legend=False,
        show_title=show_title,
    )


def input_attribution_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
    feature_names: Sequence[str] | None = None,
    top_k: int | None = 25,
) -> go.Figure:
    matrix, mapping = _matrix(trace)
    # Prefer a named feature-importance chart for one-row / one-dimensional tabular inputs.
    if matrix.ndim == 2 and matrix.shape[0] == 1:
        names = _resolve_feature_names(trace, feature_names, matrix.shape[1])
        if names is not None:
            return _feature_importance_figure(
                matrix[0],
                names,
                mode=mode,
                top_k=top_k,
                mapping=mapping,
                show_title=show_title,
            )

    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    limit = max(limit, 1e-12)
    finite_matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    peak_row, peak_column = np.unravel_index(
        int(np.argmax(np.abs(finite_matrix))), finite_matrix.shape
    )
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            colorscale=signed_colour_scale(mode),
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Signed<br>attribution"},
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Attribution: %{z:.5g}<extra></extra>",
            hoverongaps=False,
        )
    )
    figure.update_xaxes(title_text="Input position or sampled position", title_standoff=18)
    figure.update_yaxes(title_text="Input row", autorange="reversed", title_standoff=18)
    figure.update_layout(margin={"b": 90, "l": 90, "r": 80})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="colour clipped symmetrically at the 99th absolute percentile; data unchanged",
            key_results=(
                PublicationResult(
                    "Strongest displayed position",
                    f"row {peak_row}, column {peak_column}",
                    f"Signed local attribution {float(finite_matrix[peak_row, peak_column]):.5g}.",
                ),
                PublicationResult(
                    "Positive attribution mass",
                    f"{float(np.sum(np.clip(finite_matrix, 0.0, None))):.5g}",
                    "Sum of positive stored input-times-gradient terms.",
                ),
                PublicationResult(
                    "Opposing attribution mass",
                    f"{float(np.sum(np.abs(np.clip(finite_matrix, None, 0.0)))):.5g}",
                    "Absolute sum of negative stored input-times-gradient terms.",
                ),
            ),
            extra={"display_mapping": mapping, "presentation": "heatmap"},
        ),
        mode=mode,
        title="VRAVEN Input Attribution",
        subtitle=f"Signed input-times-gradient; {mapping}",
        show_legend=False,
        show_title=show_title,
    )
