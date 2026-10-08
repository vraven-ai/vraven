"""Observational--causal layer concordance visualisation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.metrics.evidence_audit import observational_causal_concordance
from vraven.visualisation.base import (
    PublicationResult,
    VisualContract,
    VisualisationMetadata,
    finalise_figure,
)
from vraven.visualisation.labels import shorten_layer_names
from vraven.visualisation.palette import RenderMode, semantic_colour

CONTRACT = VisualContract(
    visual_id="vraven.mechanism_concordance",
    research_question=(
        "Does local activation-gradient evidence predict the direction and relative magnitude "
        "of measured zero-ablation effects across matched layers?"
    ),
    required_metrics=("activation_gradient_signed", "zero-ablation score_delta"),
    data_domain="paired signed layer effects after separate L1 normalisation",
    visual_encodings=(
        "x=L1-normalised first-order activation-gradient interaction",
        "y=L1-normalised zero-ablation target-score drop",
        "point=one unambiguously matched layer",
        "shape and colour=sign agreement",
        "diagonal=perfect signed magnitude agreement",
    ),
    limitations=(
        "Zero ablation can create out-of-distribution internal states.",
        "Agreement supports local concordance, not recovery of a unique causal mechanism.",
        "Reused modules are excluded when one intervention maps to multiple executions.",
    ),
    evidence_tier="observational_interventional",
    interpretation=(
        "Agreement indicates that local first-order allocation tracks measured zero-ablation "
        "effects across matched layers; disagreement identifies where the observational proxy fails."
    ),
    metric_definitions=(
        "rho = Spearman rank correlation of matched signed layer effects",
        "sign agreement = matched layers with equal non-zero effect direction / matched layers",
        "L1 discrepancy = sum of absolute differences between separately L1-normalised effects",
    ),
)


def mechanism_concordance_figure(
    trace: NeuralTrace,
    *,
    mode: RenderMode = "publication_light",
    random_seed: int = 17,
    max_permutations: int = 4096,
    show_title: bool | None = None,
) -> go.Figure:
    """Plot matched observational and zero-ablation layer effects.

    Required trace data are signed activation-gradient interactions and explicit
    zero-output interventions.  Each axis is normalised independently by its L1
    mass, so the comparison tests direction and relative allocation rather than
    assuming the two estimators share an absolute scale.
    """

    audit = observational_causal_concordance(
        trace,
        random_seed=random_seed,
        max_permutations=max_permutations,
    )
    if not audit.layer_names:
        raise CapabilityError(
            "Mechanism Concordance requires unambiguously matched gradient layers and "
            "strategy='zero' causal interventions"
        )
    observed = np.asarray(audit.observational_l1, dtype=np.float64)
    causal = np.asarray(audit.causal_l1, dtype=np.float64)
    agreed = np.sign(observed) == np.sign(causal)
    full_names = list(audit.layer_names)
    labels = shorten_layer_names(full_names, max_length=22)
    colours = [
        semantic_colour("stable", mode) if match else semantic_colour("opposing", mode)
        for match in agreed
    ]
    symbols = ["circle" if match else "x" for match in agreed]
    figure = go.Figure()
    bound = max(float(np.max(np.abs(np.concatenate((observed, causal))))), 0.05) * 1.12
    figure.add_trace(
        go.Scatter(
            x=[-bound, bound],
            y=[-bound, bound],
            mode="lines",
            name="Perfect concordance",
            line={"color": semantic_colour("neutral", mode), "width": 1.3, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=observed,
            y=causal,
            mode="markers+text" if len(labels) <= 8 else "markers",
            name="Matched layers",
            text=labels,
            textposition="top center",
            textfont={"size": 10},
            marker={
                "size": 13 if mode == "interactive_dark" else 11,
                "color": colours,
                "symbol": symbols,
                "line": {"width": 1.1, "color": semantic_colour("neutral", mode)},
            },
            customdata=np.column_stack(
                (
                    np.asarray(full_names, dtype=object),
                    np.asarray(audit.observational_raw, dtype=np.float64),
                    np.asarray(audit.causal_raw, dtype=np.float64),
                    np.where(agreed, "same sign", "opposite sign"),
                )
            ),
            hovertemplate=(
                "%{customdata[0]}<br>First-order raw: %{customdata[1]:.5g}"
                "<br>Zero-ablation raw: %{customdata[2]:.5g}"
                "<br>First-order L1 share: %{x:.3f}"
                "<br>Zero-ablation L1 share: %{y:.3f}"
                "<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0.0, line_width=0.8, line_color=semantic_colour("neutral", mode))
    figure.add_vline(x=0.0, line_width=0.8, line_color=semantic_colour("neutral", mode))
    figure.update_xaxes(
        title_text="Observational first-order interaction (signed L1 share)",
        range=[-bound, bound],
        zeroline=False,
        title_standoff=18,
    )
    figure.update_yaxes(
        title_text="Measured zero-ablation score drop (signed L1 share)",
        range=[-bound, bound],
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        title_standoff=18,
    )
    statistic_parts = []
    if audit.spearman_rho is not None:
        statistic_parts.append(f"Spearman ρ={audit.spearman_rho:.3f}")
    if audit.sign_agreement is not None:
        statistic_parts.append(f"sign agreement={audit.sign_agreement:.1%}")
    if audit.normalised_l1_discrepancy is not None:
        statistic_parts.append(f"L1 discrepancy={audit.normalised_l1_discrepancy:.3f}")
    if audit.permutation_p_value is not None:
        statistic_parts.append(f"permutation p={audit.permutation_p_value:.4f}")
    subtitle = "; ".join(statistic_parts) or "Matched zero-ablation diagnostic"
    figure = finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="each signed vector divided by its own total absolute mass",
            random_seed=random_seed if audit.permutation_method else None,
            key_results=(
                PublicationResult(
                    "Rank concordance",
                    "Not estimable" if audit.spearman_rho is None else f"{audit.spearman_rho:.3f}",
                    "Spearman correlation across uniquely matched layers.",
                    "rho",
                ),
                PublicationResult(
                    "Sign agreement",
                    "Not estimable"
                    if audit.sign_agreement is None
                    else f"{audit.sign_agreement:.1%}",
                    "Share of matched layers with the same observational and ablation direction.",
                ),
                PublicationResult(
                    "Permutation evidence",
                    "Not estimable"
                    if audit.permutation_p_value is None
                    else f"p={audit.permutation_p_value:.4f}",
                    audit.permutation_method or "Insufficient matched layers for a permutation test.",
                ),
            ),
            extra={
                "matched_layers": list(audit.layer_names),
                "spearman_rho": audit.spearman_rho,
                "sign_agreement": audit.sign_agreement,
                "cosine_similarity": audit.cosine_similarity,
                "normalised_l1_discrepancy": audit.normalised_l1_discrepancy,
                "permutation_p_value": audit.permutation_p_value,
                "permutation_method": audit.permutation_method,
                "excluded_nonzero_interventions": list(audit.excluded_nonzero_interventions),
                "excluded_ambiguous_layers": list(audit.excluded_ambiguous_layers),
            },
        ),
        mode=mode,
        title="VRAVEN Mechanism Concordance",
        subtitle=subtitle,
        width=900,
        height=680,
        show_legend=False,
        show_title=show_title,
    )
    figure.update_layout(margin={"l": 110, "r": 50, "t": 100, "b": 100})
    return figure
