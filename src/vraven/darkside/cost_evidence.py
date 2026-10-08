"""Computation cost versus measured evidence contribution."""

from __future__ import annotations

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import pareto_frontier


def analyse_cost_evidence_tradeoff(
    dataset: TraceDataset,
    *,
    evidence: str = "absolute_influence",
) -> AnalysisResult:
    """Locate layers that consume runtime while adding little measured evidence."""

    if evidence not in {"absolute_influence", "activation_rms"}:
        raise ValueError("evidence must be 'absolute_influence' or 'activation_rms'")
    rows = []
    zero_timing_layers: list[str] = []
    for layer in dataset.layers:
        if layer not in dataset.timing_ms:
            raise ValueError(f"Timing data are unavailable for layer {layer!r}")
        timings = np.asarray(dataset.timing_ms[layer], dtype=np.float64).reshape(-1)
        if np.any(timings < 0):
            raise ValueError(f"Timing data contain negative values for layer {layer!r}")
        latency = float(np.mean(timings))
        if latency <= 0:
            zero_timing_layers.append(layer)
        if evidence == "absolute_influence":
            if layer not in dataset.unit_influences:
                raise ValueError(f"Unit influence data are unavailable for layer {layer!r}")
            benefit = float(np.mean(np.sum(np.abs(dataset.unit_influences[layer]), axis=1)))
        else:
            if layer not in dataset.activation_rms:
                raise ValueError(f"Activation RMS data are unavailable for layer {layer!r}")
            benefit = float(np.mean(dataset.activation_rms[layer]))
        rows.append(
            {
                "layer": layer,
                "mean_latency_ms": latency,
                "mean_evidence": benefit,
                "evidence_per_ms": benefit / max(latency, 1e-12),
            }
        )
    costs = np.asarray([row["mean_latency_ms"] for row in rows])
    benefits = np.asarray([row["mean_evidence"] for row in rows])
    frontier = pareto_frontier(costs, benefits)
    total_cost = float(costs.sum())
    total_evidence = float(benefits.sum())
    cumulative_cost = 0.0
    cumulative_evidence = 0.0
    for index, row in enumerate(rows):
        cumulative_cost += row["mean_latency_ms"]
        cumulative_evidence += row["mean_evidence"]
        row["pareto_efficient"] = bool(frontier[index])
        row["cumulative_cost_fraction"] = cumulative_cost / max(total_cost, 1e-12)
        row["cumulative_evidence_fraction"] = cumulative_evidence / max(total_evidence, 1e-12)
        row["evidence_debt"] = (
            row["cumulative_cost_fraction"] - row["cumulative_evidence_fraction"]
        )
    least = min(rows, key=lambda row: row["evidence_per_ms"])
    return AnalysisResult(
        analysis="cost_evidence_tradeoff",
        metrics={
            "layers": len(rows),
            "total_mean_latency_ms": total_cost,
            "total_evidence": total_evidence,
            "pareto_layers": int(np.sum(frontier)),
            "lowest_efficiency_layer": least["layer"],
            "lowest_evidence_per_ms": least["evidence_per_ms"],
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "performance and explanatory-signal diagnostic",
            "evidence_metric": evidence,
        },
        warnings=[
            "Low measured evidence does not prove that a layer can be removed without retraining or quality loss.",
            *(
                [
                    "Some layers have zero recorded timing; evidence-per-millisecond is then numerically unstable: "
                    + ", ".join(zero_timing_layers)
                ]
                if zero_timing_layers
                else []
            ),
        ],
        limitations=[
            "Hook timing can include framework synchronisation and measurement overhead.",
            "Absolute influence is not an additive conservation law across layers.",
        ],
    )
