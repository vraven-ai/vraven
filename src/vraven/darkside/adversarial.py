"""Internal rerouting caused by adversarial, evasive or counterfactual inputs."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import cosine_similarity, linear_cka, neighbourhood_overlap


def analyse_adversarial_rerouting(
    clean: TraceDataset,
    altered: TraceDataset,
    *,
    input_l0: np.ndarray | None = None,
    input_l2: np.ndarray | None = None,
    neighbours: int = 10,
) -> AnalysisResult:
    """Measure where small input changes redirect internal evidence pathways."""

    if clean.sample_ids != altered.sample_ids:
        raise ValueError("Clean and altered datasets must contain the same ordered samples")
    rows = []
    for layer in clean.layers:
        if layer not in altered.representations:
            continue
        left = clean.representations[layer]
        right = altered.representations[layer]
        if left.shape != right.shape:
            raise ValueError(f"Aligned representation shape changed for {layer}")
        paired = np.asarray(
            [cosine_similarity(x, y) for x, y in zip(left, right, strict=True)]
        )
        influence_left = clean.unit_influences.get(layer)
        influence_right = altered.unit_influences.get(layer)
        sign_flip = None
        influence_cosine = None
        if influence_left is not None and influence_right is not None:
            width = min(influence_left.shape[1], influence_right.shape[1])
            a = influence_left[:, :width]
            b = influence_right[:, :width]
            sign_flip = float(np.mean(np.sign(a) != np.sign(b)))
            influence_cosine = float(
                np.mean([cosine_similarity(x, y) for x, y in zip(a, b, strict=True)])
            )
        reroute = 1.0 - float(np.mean(paired))
        rows.append(
            {
                "layer": layer,
                "representation_cka": linear_cka(left, right),
                "mean_paired_cosine": float(np.mean(paired)),
                "neighbourhood_overlap": neighbourhood_overlap(
                    left, right, neighbours=neighbours
                ),
                "unit_sign_flip_fraction": sign_flip,
                "unit_influence_cosine": influence_cosine,
                "reroute_index": reroute,
            }
        )
    if not rows:
        raise ValueError("No shared representation layers were found")
    score_delta = altered.target_scores - clean.target_scores
    prediction_flip = altered.predicted_labels != clean.predicted_labels
    critical = max(rows, key=lambda row: row["reroute_index"])
    metrics = {
        "samples": len(clean.sample_ids),
        "prediction_flip_rate": float(np.mean(prediction_flip)),
        "mean_target_score_change": float(np.mean(score_delta)),
        "critical_reroute_layer": critical["layer"],
        "maximum_reroute_index": critical["reroute_index"],
    }
    if input_l0 is not None:
        l0 = np.asarray(input_l0, dtype=np.float64).reshape(-1)
        if l0.size != len(clean.sample_ids) or not np.isfinite(l0).all():
            raise ValueError("input_l0 must contain one finite value per sample")
        metrics["mean_input_l0"] = float(np.mean(l0))
    if input_l2 is not None:
        l2 = np.asarray(input_l2, dtype=np.float64).reshape(-1)
        if l2.size != len(clean.sample_ids) or not np.isfinite(l2).all():
            raise ValueError("input_l2 must contain one finite value per sample")
        metrics["mean_input_l2"] = float(np.mean(l2))
    return AnalysisResult(
        analysis="adversarial_rerouting",
        metrics=metrics,
        tables={"layers": rows, "per_sample_score_change": score_delta.tolist()},
        metadata={
            "claim_type": "paired clean-versus-altered internal path diagnostic",
            "neighbours": neighbours,
        },
        limitations=[
            "Rerouting does not by itself prove malicious intent or adversarial construction.",
            "The audit requires exact clean-to-altered sample pairing.",
        ],
    )
