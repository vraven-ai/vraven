"""VRAVEN Latent Flight shared-projection / t-SNE small multiples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.core.trace import NeuralTrace
from vraven.reasoning.projection import TSNE_CAVEAT, layer_tsne, shared_pca
from vraven.visualisation.base import VisualContract, VisualisationMetadata, finalise_figure
from vraven.visualisation.labels import shorten_layer_names
from vraven.visualisation.palette import (
    CLASS_COLOURS,
    CLASS_SYMBOLS,
    RenderMode,
    semantic_colour,
)
from vraven.visualisation.theme import get_theme

ProjectionMethod = Literal["pca", "tsne"]

CONTRACT = VisualContract(
    visual_id="vraven.latent_flight",
    research_question="How does batch representation geometry evolve through selected layers?",
    required_metrics=("representation",),
    data_domain="shared PCA or per-layer t-SNE coordinates",
    visual_encodings=(
        "panel=selected model depth",
        "x/y=projected coordinates",
        "colour and marker shape=class identity",
        "arrow=model order",
    ),
    limitations=(
        "PCA preserves linear variance rather than all neighbourhood relationships.",
        TSNE_CAVEAT,
        "Class separation in projection is descriptive and does not establish causal reasoning.",
    ),
)


def _class_name(trace: NeuralTrace, label: Any) -> str:
    names = trace.metadata.get("class_names", {})
    if isinstance(names, list):
        try:
            return str(names[int(label)])
        except Exception:
            return str(label)
    if isinstance(names, dict):
        return str(names.get(label, names.get(str(label), label)))
    user = trace.metadata.get("user_metadata")
    if isinstance(user, dict) and isinstance(user.get("class_names"), list):
        try:
            return str(user["class_names"][int(label)])
        except Exception:
            return str(label)
    return str(label)


def _class_colour(name: str, label_index: int, mode: RenderMode) -> str:
    """Map common security class names to semantic colours when possible."""

    key = str(name).strip().lower()
    if any(token in key for token in ("malware", "injection", "attack", "anomaly", "malicious")):
        return semantic_colour("opposing", mode)
    if any(token in key for token in ("benign", "normal", "clean", "goodware")):
        return semantic_colour("stable", mode)
    if "unknown" in key:
        return semantic_colour("unknown", mode)
    return CLASS_COLOURS[label_index % len(CLASS_COLOURS)]


def latent_flight_figure(
    trace: NeuralTrace,
    *,
    labels: Sequence[Any] | None = None,
    layer_names: Sequence[str] | None = None,
    max_layers: int = 4,
    method: ProjectionMethod = "pca",
    random_seed: int = 17,
    perplexity: float = 30.0,
    align: bool = True,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if method == "pca":
        projection = shared_pca(
            trace,
            layer_names=tuple(layer_names) if layer_names is not None else None,
            max_layers=max_layers,
        )
    elif method == "tsne":
        projection = layer_tsne(
            trace,
            layer_names=tuple(layer_names) if layer_names is not None else None,
            max_layers=max_layers,
            random_seed=random_seed,
            perplexity=perplexity,
            align=align,
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    if labels is None:
        predicted = trace.prediction.predicted_classes if trace.prediction else None
        labels = (
            predicted
            if predicted and len(predicted) == projection.sample_count
            else [0] * projection.sample_count
        )
    labels_array = np.asarray(list(labels), dtype=object)
    if labels_array.size != projection.sample_count:
        raise ValueError(
            f"Expected {projection.sample_count} labels but received {labels_array.size}"
        )
    unique_labels = sorted(set(labels_array.tolist()), key=str)
    column_count = len(projection.layer_names)
    titles = shorten_layer_names(projection.layer_names, max_length=20)
    figure = make_subplots(
        rows=1,
        cols=column_count,
        subplot_titles=titles,
        horizontal_spacing=0.07,
    )
    all_coordinates = np.vstack(projection.coordinates)
    x_padding = max(float(np.ptp(all_coordinates[:, 0])) * 0.10, 0.25)
    y_padding = max(float(np.ptp(all_coordinates[:, 1])) * 0.10, 0.25)
    x_range = [
        float(all_coordinates[:, 0].min() - x_padding),
        float(all_coordinates[:, 0].max() + x_padding),
    ]
    y_range = [
        float(all_coordinates[:, 1].min() - y_padding),
        float(all_coordinates[:, 1].max() + y_padding),
    ]
    theme = get_theme(mode)
    marker_line = {
        "width": 0.6 if mode == "interactive_dark" else 0.4,
        "color": theme.border,
    }

    for column, coordinates in enumerate(projection.coordinates, start=1):
        for label_index, label in enumerate(unique_labels):
            mask = labels_array == label
            class_name = _class_name(trace, label)
            figure.add_trace(
                go.Scatter(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    mode="markers",
                    marker={
                        "size": 8 if mode == "interactive_dark" else 7,
                        "color": _class_colour(class_name, label_index, mode),
                        "symbol": CLASS_SYMBOLS[label_index % len(CLASS_SYMBOLS)],
                        "opacity": 0.86,
                        "line": marker_line,
                    },
                    name=class_name,
                    legendgroup=str(label),
                    showlegend=column == 1,
                    customdata=np.flatnonzero(mask),
                    hovertemplate=(
                        f"{class_name}<br>Sample: %{{customdata}}<br>"
                        "Comp. 1: %{x:.4f}<br>Comp. 2: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(range=x_range, row=1, col=column)
        figure.update_yaxes(range=y_range, row=1, col=column)
        if column > 1:
            figure.update_yaxes(showticklabels=False, title_text=None, row=1, col=column)

    if method == "pca":
        x_label = f"Comp. 1 ({projection.explained_variance_ratio[0]:.1%})"
        y_label = f"Comp. 2 ({projection.explained_variance_ratio[1]:.1%})"
        subtitle = "Representation landscape across model depth (shared PCA)"
    else:
        x_label = "Comp. 1"
        y_label = "Comp. 2"
        subtitle = (
            f"Representation landscape (t-SNE) · seed={projection.random_seed}"
        )
    for column in range(1, column_count + 1):
        figure.update_xaxes(title_text=x_label, title_standoff=12, row=1, col=column)
    figure.update_yaxes(title_text=y_label, title_standoff=12, row=1, col=1)

    for boundary in range(1, column_count):
        figure.add_annotation(
            x=(boundary / column_count) - 0.01,
            y=0.52,
            xref="paper",
            yref="paper",
            text="→",
            showarrow=False,
            font={"size": 26, "color": semantic_colour("selected", mode)},
        )
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.22,
            "yanchor": "top",
            "title": {"text": "Class"},
        },
        margin={"b": 130, "t": 100, "l": 70, "r": 40},
    )
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation=(
                "one PCA basis fitted to the union of selected layer representations"
                if method == "pca"
                else "per-layer t-SNE with optional Procrustes alignment"
            ),
            random_seed=projection.random_seed
            or (
                trace.metadata.get("capture_configuration", {}).get("random_seed")
                if isinstance(trace.metadata.get("capture_configuration"), dict)
                else None
            ),
            extra={
                "selected_layers": list(projection.layer_names),
                "explained_variance_ratio": list(projection.explained_variance_ratio),
                "projection": projection.projection,
                "method": method,
                "projection_metadata": projection.metadata or {},
                "caveat": TSNE_CAVEAT if method == "tsne" else None,
            },
        ),
        mode=mode,
        title="VRAVEN Representation Landscape",
        subtitle=subtitle,
        width=max(980, 340 * column_count),
        height=620 if mode == "publication_light" else 700,
        show_title=show_title,
    )
