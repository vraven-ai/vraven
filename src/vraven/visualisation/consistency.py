"""Multi-method explanation agreement matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from vraven.metrics.agreement import agreement_matrices
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.palette import RenderMode, sequential_colour_scale, signed_colour_scale

AgreementMetric = Literal["rank_correlation", "sign_agreement", "topk_overlap"]

CONTRACT = VisualContract(
    visual_id="vraven.consistency_matrix",
    research_question="How consistently do independent explanation vectors rank and sign features?",
    required_metrics=("rank_correlation", "sign_agreement", "topk_overlap"),
    data_domain="pairwise method agreement",
    visual_encodings=("row/column=method", "colour and text=agreement"),
    limitations=(
        "Agreement does not prove that any method is correct.",
        "Compared explanation vectors must address the same sample, target and feature space.",
    ),
    evidence_tier="comparative",
    interpretation=(
        "Off-diagonal cells quantify method agreement under one registered metric. High "
        "agreement is reproducibility evidence, not evidence that the shared explanation is correct."
    ),
    metric_definitions=(
        "rank_correlation = Spearman correlation of paired explanation vectors",
        "sign_agreement = fraction of coordinates with the same sign",
        "topk_overlap = Jaccard overlap of the absolute top-k coordinate sets",
    ),
)


def consistency_matrix_figure(
    explanations: Mapping[str, Sequence[float]],
    *,
    metric: AgreementMetric = "rank_correlation",
    top_k: int | None = None,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    result = agreement_matrices(explanations, top_k=top_k)
    matrix = getattr(result, metric)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if off_diagonal.size:
        minimum = float(np.min(off_diagonal))
        median = float(np.median(off_diagonal))
    else:
        minimum = median = float("nan")
    signed = metric == "rank_correlation"
    zmin, zmax = (-1.0, 1.0) if signed else (0.0, 1.0)
    text = np.vectorize(lambda value: f"{value:.2f}")(matrix)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=result.labels,
            y=result.labels,
            zmin=zmin,
            zmax=zmax,
            zmid=0.0 if signed else None,
            colorscale=signed_colour_scale(mode) if signed else sequential_colour_scale(mode),
            text=text,
            texttemplate="%{text}",
            colorbar={"title": metric.replace("_", " ")},
            hovertemplate="%{y} vs %{x}<br>Agreement: %{z:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Explanation method", side="bottom")
    figure.update_yaxes(title_text="Explanation method", autorange="reversed")
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="metric-defined range",
            key_results=(
                PublicationResult(
                    "Median pairwise agreement",
                    "Not estimable" if not np.isfinite(median) else f"{median:.3f}",
                    f"Median off-diagonal {metric.replace('_', ' ')} across methods.",
                ),
                PublicationResult(
                    "Weakest method pair",
                    "Not estimable" if not np.isfinite(minimum) else f"{minimum:.3f}",
                    "Minimum off-diagonal agreement; inspect the labelled cell for the pair.",
                ),
                PublicationResult(
                    "Methods compared",
                    str(len(result.labels)),
                    "Explanation vectors registered on the same coordinate space.",
                    "methods",
                ),
            ),
            extra={"displayed_metric": metric, "top_k": top_k},
        ),
        mode=mode,
        title="VRAVEN Consistency Matrix",
        subtitle=metric.replace("_", " ").title(),
        show_legend=False,
        show_title=show_title,
    )
