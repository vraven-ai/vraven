"""Dataset-level reasoning-fingerprint visualisations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.visualisation.base import VisualContract, VisualisationMetadata, finalise_figure
from vraven.visualisation.palette import (
    CLASS_COLOURS,
    CLASS_SYMBOLS,
    RenderMode,
    semantic_colour,
    signed_colour_scale,
)

SIMILARITY_CONTRACT = VisualContract(
    visual_id="vraven.reasoning_similarity_matrix",
    research_question="Which samples have similar reduced internal trace fingerprints?",
    required_metrics=("fixed-width reasoning fingerprint", "cosine similarity"),
    data_domain="independent sample traces",
    visual_encodings=(
        "row and column=sample identifier",
        "cell colour=cosine similarity between reduced trace fingerprints",
    ),
    limitations=(
        "Similarity reflects the selected VRAVEN fingerprint metrics and dimensions.",
        "Similarity does not prove semantic or causal equivalence between samples.",
    ),
)

LANDSCAPE_CONTRACT = VisualContract(
    visual_id="vraven.reasoning_landscape",
    research_question="How are sample-level reduced internal trace fingerprints arranged in one shared low-dimensional space?",
    required_metrics=("fixed-width reasoning fingerprint", "shared principal-component projection"),
    data_domain="independent sample traces",
    visual_encodings=(
        "x=first principal component of fingerprints",
        "y=second principal component of fingerprints",
        "colour and marker shape=declared grouping variable",
        "point=one independently explained sample",
    ),
    limitations=(
        "Distances are a two-dimensional approximation of fingerprint geometry.",
        "Clusters do not establish human-readable concepts or causal mechanisms.",
    ),
)


def _fingerprint_matrix(fingerprints: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(fingerprints, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise CapabilityError("Dataset reasoning views require a non-empty [samples, dimensions] matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Fingerprint matrix contains non-finite values")
    return matrix


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = matrix / np.where(norms > 1e-12, norms, 1.0)
    similarity = normalised @ normalised.T
    for index, norm in enumerate(norms.reshape(-1)):
        if norm <= 1e-12:
            similarity[index, index] = 1.0
    return np.clip(similarity, -1.0, 1.0)


def reasoning_similarity_matrix_figure(
    fingerprints: np.ndarray | Sequence[Sequence[float]],
    *,
    sample_ids: Sequence[str] | None = None,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    matrix = _fingerprint_matrix(fingerprints)
    count = matrix.shape[0]
    identifiers = tuple(sample_ids or [f"sample_{index:05d}" for index in range(count)])
    if len(identifiers) != count:
        raise ValueError("sample_ids must match the fingerprint row count")
    similarity = _cosine_matrix(matrix)
    figure = go.Figure(
        go.Heatmap(
            z=similarity,
            x=identifiers,
            y=identifiers,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale=signed_colour_scale(mode),
            colorbar={"title": "Cosine<br>similarity", "thickness": 12, "len": 0.76},
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Similarity: %{z:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Sample", tickangle=-45)
    figure.update_yaxes(title_text="Sample", autorange="reversed")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            SIMILARITY_CONTRACT,
            normalisation="L2-normalised fingerprints followed by cosine inner product",
            extra={
                "sample_count": count,
                "fingerprint_dimensions": matrix.shape[1],
                "similarity_range": [-1.0, 1.0],
            },
        ),
        mode=mode,
        title="VRAVEN Reasoning Similarity Matrix",
        subtitle="Pairwise similarity of reduced internal trace fingerprints",
        width=max(760, min(1500, 38 * count + 360)),
        height=max(620, min(1400, 34 * count + 300)),
        show_legend=False,
        show_title=show_title,
    )


def _shared_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular, vectors = np.linalg.svd(centred, full_matrices=False)
    retained = min(2, vectors.shape[0])
    coordinates = centred @ vectors[:retained].T
    if retained < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - retained)))
    total = np.square(singular).sum()
    ratio = np.square(singular[:retained]) / total if total > 0 else np.zeros(retained)
    ratio = np.pad(ratio, (0, 2 - retained))
    return coordinates, ratio


def reasoning_landscape_figure(
    fingerprints: np.ndarray | Sequence[Sequence[float]],
    *,
    sample_ids: Sequence[str] | None = None,
    groups: Sequence[object] | None = None,
    group_name: str = "group",
    method: str = "pca",
    highlight_index: int | None = None,
    random_seed: int = 17,
    perplexity: float = 30.0,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    matrix = _fingerprint_matrix(fingerprints)
    count = matrix.shape[0]
    if count < 2:
        raise CapabilityError("Reasoning Landscape requires at least two samples")
    identifiers = tuple(sample_ids or [f"sample_{index:05d}" for index in range(count)])
    if len(identifiers) != count:
        raise ValueError("sample_ids must match the fingerprint row count")
    resolved_groups = tuple(str(value) if value is not None else "unlabelled" for value in (groups or ["all"] * count))
    if len(resolved_groups) != count:
        raise ValueError("groups must match the fingerprint row count")

    projection_meta: dict[str, object] = {"projection": "shared_pca"}
    if method == "pca":
        coordinates, ratio = _shared_pca(matrix)
        x_title = f"Fingerprint PC1 ({ratio[0] * 100:.1f}% variance)"
        y_title = f"Fingerprint PC2 ({ratio[1] * 100:.1f}% variance)"
        subtitle = "Shared PCA projection of independent sample trace fingerprints"
    elif method == "tsne":
        from vraven.reasoning.projection import TSNE_CAVEAT, dataset_tsne

        coordinates, projection_meta = dataset_tsne(
            matrix,
            random_seed=random_seed,
            perplexity=perplexity,
        )
        ratio = np.asarray([0.0, 0.0])
        x_title = "t-SNE dimension 1"
        y_title = "t-SNE dimension 2"
        subtitle = f"Dataset t-SNE of reasoning fingerprints; seed={random_seed}"
        projection_meta = {**projection_meta, "caveat": TSNE_CAVEAT}
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    unique_groups = tuple(dict.fromkeys(resolved_groups))
    figure = go.Figure()
    for group_index, group in enumerate(unique_groups):
        indices = np.asarray([index for index, value in enumerate(resolved_groups) if value == group])
        figure.add_trace(
            go.Scatter(
                x=coordinates[indices, 0],
                y=coordinates[indices, 1],
                mode="markers",
                name=f"{group_name}: {group}",
                marker={
                    "size": 10,
                    "color": CLASS_COLOURS[group_index % len(CLASS_COLOURS)],
                    "symbol": CLASS_SYMBOLS[group_index % len(CLASS_SYMBOLS)],
                    "opacity": 0.82,
                    "line": {"width": 0.7},
                },
                customdata=np.asarray([identifiers[index] for index in indices])[:, None],
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>"
                    "Dim1: %{x:.4g}<br>Dim2: %{y:.4g}<br>"
                    f"{group_name.title()}: {group}<extra></extra>"
                ),
            )
        )
    if highlight_index is not None:
        if not 0 <= highlight_index < count:
            raise ValueError("highlight_index is out of range")
        figure.add_trace(
            go.Scatter(
                x=[coordinates[highlight_index, 0]],
                y=[coordinates[highlight_index, 1]],
                mode="markers+text",
                name="This sample",
                text=["This sample"],
                textposition="top center",
                marker={
                    "size": 16,
                    "color": semantic_colour("selected", mode),
                    "symbol": "circle-open",
                    "line": {"width": 2.5, "color": semantic_colour("selected", mode)},
                },
                hovertemplate="Highlighted sample: %{text}<extra></extra>",
            )
        )
        figure.add_annotation(
            x=float(coordinates[highlight_index, 0]),
            y=float(coordinates[highlight_index, 1]),
            text="This sample",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font={"color": semantic_colour("selected", mode)},
        )
    figure.update_xaxes(title_text=x_title, title_standoff=16)
    figure.update_yaxes(title_text=y_title, title_standoff=16)
    figure.update_layout(margin={"l": 90, "r": 40, "b": 100, "t": 90})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            LANDSCAPE_CONTRACT,
            normalisation=(
                "one deterministic PCA basis fitted to the complete fingerprint matrix"
                if method == "pca"
                else "single dataset t-SNE embedding"
            ),
            random_seed=random_seed if method == "tsne" else None,
            extra={
                "sample_count": count,
                "fingerprint_dimensions": matrix.shape[1],
                "explained_variance_ratio": ratio.tolist(),
                "group_name": group_name,
                "group_values": list(unique_groups),
                "method": method,
                "highlight_index": highlight_index,
                **projection_meta,
            },
        ),
        mode=mode,
        title="VRAVEN Internal Representation",
        subtitle=subtitle,
        width=920,
        height=650,
        show_legend=True,
        show_title=show_title,
    )

OUTLIER_CONTRACT = VisualContract(
    visual_id="vraven.reasoning_outlier",
    research_question="Which bounded dataset samples have fingerprints that are unusual relative to their nearest neighbours?",
    required_metrics=("fixed-width reasoning fingerprint", "nearest-neighbour cosine similarity"),
    data_domain="independent sample traces",
    visual_encodings=(
        "x=sample identifier",
        "y=one minus mean nearest-neighbour cosine similarity",
        "marker emphasis=highest observed outlier scores",
    ),
    limitations=(
        "An unusual trace is a review candidate, not proof of an adversarial sample, drift, backdoor or model error.",
        "Scores depend on cohort composition, fingerprint definition and neighbourhood size.",
    ),
)


def reasoning_outlier_figure(
    fingerprints: np.ndarray | Sequence[Sequence[float]],
    *,
    sample_ids: Sequence[str] | None = None,
    neighbours: int = 5,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    matrix = _fingerprint_matrix(fingerprints)
    count = matrix.shape[0]
    if count < 2:
        raise CapabilityError("Reasoning Outlier View requires at least two samples")
    if neighbours <= 0:
        raise ValueError("neighbours must be greater than zero")
    identifiers = tuple(sample_ids or [f"sample_{index:05d}" for index in range(count)])
    if len(identifiers) != count:
        raise ValueError("sample_ids must match the fingerprint row count")
    similarity = _cosine_matrix(matrix)
    np.fill_diagonal(similarity, -np.inf)
    resolved_neighbours = min(neighbours, count - 1)
    nearest = np.partition(similarity, -resolved_neighbours, axis=1)[:, -resolved_neighbours:]
    scores = 1.0 - nearest.mean(axis=1)
    rank = np.argsort(np.argsort(scores))
    highlighted = rank >= max(0, count - max(1, int(np.ceil(count * 0.10))))
    colours = [
        semantic_colour("opposing", mode) if flag else semantic_colour("latent", mode)
        for flag in highlighted
    ]
    figure = go.Figure(
        go.Bar(
            x=identifiers,
            y=scores,
            marker={"color": colours},
            customdata=np.column_stack((nearest.mean(axis=1), highlighted)),
            hovertemplate=(
                "Sample: %{x}<br>Outlier score: %{y:.4f}<br>"
                "Mean neighbour similarity: %{customdata[0]:.4f}<br>"
                "Top decile in this cohort: %{customdata[1]}<extra></extra>"
            ),
            name="Reasoning-fingerprint outlier score",
        )
    )
    figure.update_xaxes(title_text="Sample", tickangle=-45)
    figure.update_yaxes(title_text="1 − mean nearest-neighbour cosine similarity")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            OUTLIER_CONTRACT,
            normalisation="L2 fingerprint normalisation followed by cosine nearest neighbours",
            extra={
                "sample_count": count,
                "fingerprint_dimensions": matrix.shape[1],
                "requested_neighbours": neighbours,
                "resolved_neighbours": resolved_neighbours,
                "highlight_rule": "highest ten percent within the displayed cohort",
            },
        ),
        mode=mode,
        title="VRAVEN Dataset Reasoning Outliers",
        subtitle="Candidates with unusual reduced internal trace fingerprints",
        width=max(820, min(1600, 42 * count + 360)),
        height=590,
        show_legend=False,
        show_title=show_title,
    )
