"""Feature × sample/sequence attribution heatmap."""

from __future__ import annotations

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
from vraven.visualisation.palette import RenderMode, signed_colour_scale

CONTRACT = VisualContract(
    visual_id="vraven.feature_sequence_heatmap",
    research_question="How does signed input attribution vary across features and samples?",
    required_metrics=("per-sample input attribution matrix",),
    data_domain="batch feature attributions or supplied matrix",
    visual_encodings=(
        "y=feature name",
        "x=sample or sequence index",
        "colour=signed attribution",
    ),
    limitations=(
        "When derived from a single stored attribution vector, values are repeated across samples.",
        "When a caller-supplied [N, F] matrix is used, each row is an independent attribution vector.",
        "Sequence semantics depend on the caller-supplied sample ordering.",
    ),
    interpretation=(
        "Rows rank features by mean absolute local attribution and columns retain declared "
        "sample order; repeated-vector mode is a display limitation, not cohort evidence."
    ),
    metric_definitions=(
        "feature importance_j = mean_i |attribution_ij|",
        "colour limit = 99th percentile of displayed absolute attribution",
    ),
)


def feature_sequence_heatmap_figure(
    matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    *,
    trace: NeuralTrace | None = None,
    feature_names: Sequence[str] | None = None,
    sample_labels: Sequence[str] | None = None,
    top_features: int = 16,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    values: np.ndarray
    source = "caller_matrix"
    if matrix is not None:
        values = np.asarray(matrix, dtype=np.float64)
    elif trace is not None and trace.inputs and trace.inputs.attribution_values:
        vector = np.asarray(trace.inputs.attribution_values, dtype=np.float64).reshape(-1)
        batch = int(trace.inputs.batch_size or 1)
        values = np.tile(vector.reshape(1, -1), (batch, 1))
        source = "repeated_single_vector"
    else:
        raise CapabilityError(
            "Feature sequence heatmap requires an attribution matrix or a gradient trace"
        )
    if values.ndim != 2:
        raise ValueError("Attribution matrix must be [samples, features]")
    feature_count = values.shape[1]
    names = (
        [str(name) for name in feature_names]
        if feature_names is not None and len(feature_names) == feature_count
        else [f"f{index}" for index in range(feature_count)]
    )
    importance = np.mean(np.abs(values), axis=0)
    keep = np.argsort(importance)[::-1][: min(top_features, feature_count)]
    selected = values[:, keep].T  # [features, samples]
    y_labels = [readable_feature_name(names[index], max_length=36) for index in keep]
    x_labels = (
        [str(label) for label in sample_labels]
        if sample_labels is not None and len(sample_labels) == values.shape[0]
        else [str(index) for index in range(values.shape[0])]
    )
    finite = np.abs(selected[np.isfinite(selected)])
    limit = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    limit = max(limit, 1e-12)
    peak_feature_index = int(np.argmax(np.mean(np.abs(selected), axis=1)))
    figure = go.Figure(
        go.Heatmap(
            z=selected,
            x=x_labels,
            y=y_labels,
            colorscale=signed_colour_scale(mode),
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Impact", "thickness": 12},
            hovertemplate="Feature: %{y}<br>Sample: %{x}<br>Impact: %{z:.5g}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Time / sequence", title_standoff=18, tickangle=-30)
    figure.update_yaxes(title_text="Feature", automargin=True, title_standoff=16)
    figure.update_layout(margin={"l": 180, "r": 70, "b": 100, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="colour clipped at 99th absolute percentile",
            key_results=(
                PublicationResult(
                    "Highest mean-|attribution| feature",
                    y_labels[peak_feature_index],
                    f"Mean absolute displayed attribution {float(np.mean(np.abs(selected[peak_feature_index]))):.5g}.",
                ),
                PublicationResult(
                    "Displayed matrix",
                    f"{int(keep.size)} × {int(values.shape[0])}",
                    "Selected features by ordered samples/sequence positions.",
                ),
                PublicationResult(
                    "Attribution source",
                    source.replace("_", " "),
                    (
                        "Independent caller-provided rows."
                        if source == "caller_matrix"
                        else "One stored vector repeated; no cross-sample inference is valid."
                    ),
                ),
            ),
            extra={
                "features_shown": int(keep.size),
                "sample_count": int(values.shape[0]),
                "attribution_source": source,
            },
        ),
        mode=mode,
        title="VRAVEN Feature Attribution Heatmap",
        subtitle=(
            "Independent per-sample attributions"
            if source == "caller_matrix"
            else "WARNING: single attribution vector repeated across samples"
        ),
        width=max(900, 36 * values.shape[0] + 320),
        height=max(560, 24 * int(keep.size) + 200),
        show_legend=False,
        show_title=show_title,
    )
