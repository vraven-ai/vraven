"""Scientifically mapped Plotly figures for every DarkSide analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vraven.visualisation.labels import readable_unit_group_label, shorten_layer_name
from vraven.visualisation.theme import apply_vraven_theme, get_theme

from ..contracts import AnalysisResult


def _finish(
    figure: go.Figure,
    result: AnalysisResult,
    *,
    mode: str,
    title: str,
    subtitle: str,
    height: int | None = None,
) -> go.Figure:
    figure.update_layout(
        meta={
            "vraven_darkside": {
                "analysis": result.analysis,
                "schema_version": result.schema_version,
                "claim_type": result.metadata.get("claim_type"),
            }
        }
    )
    return apply_vraven_theme(
        figure,
        mode=mode,
        title=title,
        subtitle=subtitle,
        height=height,
    )


def _decision_circuit(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["groups"]
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    y_map = {layer: index for index, layer in enumerate(layers)}
    short_layers = [shorten_layer_name(layer.split("#", 1)[0], max_length=22) for layer in layers]
    figure = go.Figure()
    for retained, name, colour, symbol in (
        (True, "Retained sparse core", theme.semantic["selected"], "diamond"),
        (False, "Removed while sufficient", theme.semantic["neutral"], "circle-open"),
    ):
        selected = [row for row in rows if bool(row["retained"]) is retained]
        figure.add_trace(
            go.Scatter(
                x=[float(np.mean(row["indices"])) for row in selected],
                y=[y_map[row["layer"]] for row in selected],
                mode="markers",
                name=name,
                marker={
                    "size": [16.0 + 384.0 * min(abs(row["individual_score_drop"]), 1.0) for row in selected],
                    "sizemode": "area",
                    "sizeref": 1.0,
                    "color": colour,
                    "symbol": symbol,
                    "line": {"width": 1, "color": theme.border},
                },
                customdata=[
                    [
                        readable_unit_group_label(
                            row["group"], indices=row["indices"], compact=False
                        ),
                        row["individual_score_drop"],
                        shorten_layer_name(str(row["layer"]).split("#", 1)[0], max_length=28),
                        row["indices"],
                    ]
                    for row in selected
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>Layer: %{customdata[2]}"
                    "<br>Individual score drop: %{customdata[1]:.5g}"
                    "<br>Indices: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(y_map.values()),
        ticktext=short_layers,
        title="Layer",
    )
    figure.update_xaxes(title="Unit index within selected output axis")
    return _finish(
        figure,
        result,
        mode=mode,
        title="Approximate sparse decision circuitry",
        subtitle="Marker area reflects individual intervention effect; filled diamonds remain in the ε-sufficient core",
    )



def _circuit_multiplicity(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = sorted(
        result.tables["components"],
        key=lambda row: (row["inclusion_probability"], -row["inclusion_entropy_bits"]),
        reverse=True,
    )
    figure = go.Figure()
    for stable, name, role, symbol in (
        (True, "Empirical stable core", "selected", "diamond"),
        (False, "Variable or absent component", "uncertainty", "circle"),
    ):
        selected = [row for row in rows if bool(row["stable_core"]) is stable]
        if not selected:
            continue
        figure.add_trace(
            go.Scatter(
                x=[row["inclusion_probability"] for row in selected],
                y=[row["component"] for row in selected],
                mode="markers",
                name=name,
                marker={
                    "size": [49.0 + 196.0 * row["inclusion_entropy_bits"] for row in selected],
                    "sizemode": "area",
                    "sizeref": 1.0,
                    "symbol": symbol,
                    "color": theme.semantic[role],
                    "line": {"width": 1, "color": theme.border},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [row["wilson_high"] - row["inclusion_probability"] for row in selected],
                    "arrayminus": [row["inclusion_probability"] - row["wilson_low"] for row in selected],
                },
                customdata=[
                    [
                        row["included_runs"],
                        row["inclusion_entropy_bits"],
                        row["threshold_ambiguous"],
                    ]
                    for row in selected
                ],
                hovertemplate=(
                    "%{y}<br>Inclusion probability: %{x:.3f}"
                    "<br>Included runs: %{customdata[0]}"
                    "<br>Inclusion entropy: %{customdata[1]:.3f} bits"
                    "<br>Threshold ambiguous: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    threshold = float(result.metadata["stable_threshold"])
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=theme.semantic["opposing"],
        annotation_text=f"stable-core lower-bound rule: {threshold:.0%}",
    )
    figure.update_xaxes(title="Empirical component inclusion probability", range=[0, 1.02])
    figure.update_yaxes(title="Aligned component", autorange="reversed")
    return _finish(
        figure,
        result,
        mode=mode,
        title="Circuit multiplicity and empirical stable core",
        subtitle="Horizontal intervals are Wilson bounds; marker area represents inclusion entropy",
        height=max(520, min(1100, 120 + 28 * len(rows))),
    )

def _shortcut(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables.get("groups") or result.tables.get("layers")
    if result.analysis == "layerwise_shortcut_probe":
        figure = go.Figure(
            go.Scatter(
                x=[row["layer"] for row in rows],
                y=[row["probe_auc_mean"] for row in rows],
                error_y={"type": "data", "array": [row["probe_auc_std"] for row in rows]},
                mode="lines+markers",
                marker={"color": theme.semantic["uncertainty"], "size": 8},
                line={"color": theme.semantic["uncertainty"], "width": 2.2},
                name="Shortcut probe AUC",
            )
        )
        figure.add_hline(y=0.5, line_dash="dot", line_color=theme.semantic["neutral"])
        figure.update_yaxes(title="Cross-validated probe AUC", range=[0.45, 1.01])
        figure.update_xaxes(title="Layer")
        return _finish(
            figure, result, mode=mode, title="Layerwise shortcut decodability",
            subtitle="Decodability is observational and does not prove causal use",
        )
    figure = go.Figure(
        go.Scatter(
            x=[row["input_fraction"] for row in rows],
            y=[abs(row["relative_score_drop"]) for row in rows],
            mode="markers+text",
            text=[row["group"] for row in rows],
            textposition="top center",
            marker={
                "size": [64.0 + 576.0 * min(abs(row.get("attribution_dominance", 0.0)) if np.isfinite(row.get("attribution_dominance", np.nan)) else 0.0, 1.0) for row in rows],
                "sizemode": "area",
                "sizeref": 1.0,
                "color": [theme.semantic["contradiction"] if row["prediction_flipped"] else theme.semantic["uncertainty"] for row in rows],
                "line": {"width": 1, "color": theme.border},
            },
            customdata=[[row["score_drop"], row.get("attribution_dominance"), row["prediction_flipped"]] for row in rows],
            hovertemplate=(
                "Input fraction: %{x:.3g}<br>|Relative score drop|: %{y:.3g}"
                "<br>Raw score drop: %{customdata[0]:.5g}<br>Attribution dominance: %{customdata[1]:.3g}"
                "<br>Prediction flipped: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title="Fraction of input coordinates changed", type="log")
    figure.update_yaxes(title="Absolute relative target-score drop")
    return _finish(
        figure, result, mode=mode, title="Shortcut reliance audit",
        subtitle="Upper-left groups are narrow but locally dominant; red indicates a prediction flip",
    )


def _class_conditional(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=[row["layer"] for row in rows],
            y=[row["energy_distance"] for row in rows],
            name="Energy distance",
            marker_color=theme.semantic["latent"],
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=[row["layer"] for row in rows],
            y=[row["maximum_absolute_hedges_g"] for row in rows],
            name="Maximum |Hedges g|",
            mode="lines+markers",
            line={"color": theme.semantic["opposing"], "width": 2.2},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title="Distribution separation", secondary_y=False)
    figure.update_yaxes(title="Maximum standardised unit difference", secondary_y=True)
    figure.update_xaxes(title="Layer")
    return _finish(
        figure, result, mode=mode, title="Class-conditional internal separation",
        subtitle="Distribution distance and strongest unit-level class flip are shown on separate scales",
    )


def _drift(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure()
    series = (
        ("1 − CKA", [1.0 - row["cka_similarity"] for row in rows], "opposing"),
        ("Procrustes distance", [row["procrustes_distance"] for row in rows], "latent"),
        ("Bounded MMD²", [row["mmd2_rbf_bounded"] for row in rows], "uncertainty"),
        ("Bounded relative sliced Wasserstein", [row["sliced_wasserstein_bounded"] for row in rows], "selected"),
    )
    for name, values, role in series:
        figure.add_trace(
            go.Scatter(
                x=[row["layer"] for row in rows], y=values, mode="lines+markers",
                name=name, line={"color": theme.semantic[role], "width": 2.0},
            )
        )
    figure.update_xaxes(title="Layer")
    figure.update_yaxes(title="Dimensionless drift component", range=[0, 1.02])
    return _finish(
        figure, result, mode=mode, title="Representation drift profile",
        subtitle="All displayed components are bounded to [0, 1]; the composite remains an uncalibrated ranking index",
    )


def _failure(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(x=[row["layer"] for row in rows], y=[row["energy_distance"] for row in rows], name="Energy distance", marker_color=theme.semantic["opposing"]),
        secondary_y=False,
    )
    if any(row.get("failure_probe_auc") is not None for row in rows):
        figure.add_trace(
            go.Scatter(x=[row["layer"] for row in rows], y=[row.get("failure_probe_auc") for row in rows], name="Failure probe AUC", mode="lines+markers", line={"color": theme.semantic["uncertainty"], "width": 2.2}),
            secondary_y=True,
        )
        figure.add_hline(y=0.5, line_dash="dot", line_color=theme.semantic["neutral"], secondary_y=True)
    figure.update_yaxes(title="Cohort distribution distance", secondary_y=False)
    figure.update_yaxes(title="Cross-validated failure decodability", secondary_y=True)
    figure.update_xaxes(title="Layer")
    return _finish(
        figure, result, mode=mode, title="Failure anatomy across depth",
        subtitle=f"{result.metrics['failure_cohort'].upper()} compared with {result.metrics['reference_cohort'].upper()}",
    )


def _stability(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure()
    for name, key, role in (
        ("Composite stability", "stability_index", "stable"),
        ("CKA", "cka_similarity", "supporting"),
        ("Neighbourhood overlap", "neighbourhood_overlap", "selected"),
        ("Mean paired cosine", "mean_paired_cosine", "latent"),
    ):
        figure.add_trace(go.Scatter(x=[row["layer"] for row in rows], y=[row[key] for row in rows], name=name, mode="lines+markers", line={"color": theme.semantic[role], "width": 2.0}))
    figure.add_hline(y=result.metrics["collapse_threshold"], line_dash="dash", line_color=theme.semantic["opposing"], annotation_text="declared collapse threshold")
    figure.update_xaxes(title="Layer")
    figure.update_yaxes(title="Stability", range=[-0.05, 1.05])
    return _finish(
        figure, result, mode=mode, title="Internal-path stability geometry",
        subtitle="The first layer below the declared composite threshold marks a diagnostic collapse point",
    )



def _prediction_depth(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=[row["layer"] for row in rows],
            y=[row["samples_first_stable_here"] for row in rows],
            name="Samples first stable here",
            marker_color=theme.semantic["latent"],
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=[row["layer"] for row in rows],
            y=[row["probe_final_agreement"] for row in rows],
            name="Probe agreement with final prediction",
            mode="lines+markers",
            line={"color": theme.semantic["selected"], "width": 2.2},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title="Samples", secondary_y=False)
    figure.update_yaxes(title="Agreement", range=[0, 1.02], secondary_y=True)
    figure.update_xaxes(title="Layer")
    return _finish(
        figure, result, mode=mode, title="Prediction depth",
        subtitle="Earliest layer after which local probe predictions remain consistent with the model output",
    )


def _matrix(result: AnalysisResult, mode: str, *, key: str, title: str, subtitle: str) -> go.Figure:
    theme = get_theme(mode)
    names = result.tables.get("method_names") or result.tables.get("condition_names") or result.tables.get("group_names")
    matrix = result.tables.get("matrices", {}).get(key, result.tables.get("interaction_matrix"))
    values = np.asarray(matrix, dtype=float)
    maximum = float(np.max(np.abs(values))) or 1.0
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=names,
            y=names,
            zmid=0 if np.min(values) < 0 else None,
            zmin=-maximum if np.min(values) < 0 else 0,
            zmax=maximum if np.min(values) < 0 else 1,
            colorscale=(
                [[0.0, theme.semantic["opposing"]], [0.5, theme.panel], [1.0, theme.semantic["supporting"]]]
                if np.min(values) < 0
                else [[0.0, theme.panel], [1.0, theme.semantic["supporting"]]]
            ),
            colorbar={"title": key.replace("_", " ")},
            hovertemplate="%{y} × %{x}<br>%{z:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title="")
    figure.update_yaxes(title="", autorange="reversed")
    return _finish(figure, result, mode=mode, title=title, subtitle=subtitle, height=650)


def _capacity(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["units"]
    roles = {
        "dead_or_near_dead": "neutral",
        "dormant_sensitive": "uncertainty",
        "critical_active": "opposing",
        "active_low_impact": "stable",
    }
    figure = go.Figure()
    for category, role in roles.items():
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        figure.add_trace(
            go.Scatter(
                x=[row["active_fraction"] for row in selected],
                y=[max(row["mean_absolute_influence"], 1e-12) for row in selected],
                mode="markers",
                name=category.replace("_", " ").title(),
                marker={"color": theme.semantic[role], "size": 8, "opacity": 0.78},
                customdata=[[row["layer"], row["unit"], row["mean_energy"]] for row in selected],
                hovertemplate="%{customdata[0]} unit %{customdata[1]}<br>Active fraction: %{x:.3f}<br>Mean |influence|: %{y:.4g}<br>Mean energy: %{customdata[2]:.4g}<extra></extra>",
            )
        )
    figure.update_xaxes(title="Fraction of audit samples above activity threshold")
    figure.update_yaxes(title="Mean absolute target-specific influence", type="log")
    return _finish(
        figure, result, mode=mode, title="Unused capacity and dormant sensitivity",
        subtitle="Dormant-sensitive units are rarely active but influential when active",
    )


def _adversarial(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure()
    for name, values, role in (
        ("Reroute index", [row["reroute_index"] for row in rows], "opposing"),
        ("1 − representation CKA", [1.0 - row["representation_cka"] for row in rows], "uncertainty"),
        ("1 − neighbourhood overlap", [1.0 - row["neighbourhood_overlap"] for row in rows], "latent"),
    ):
        figure.add_trace(go.Scatter(x=[row["layer"] for row in rows], y=values, name=name, mode="lines+markers", line={"color": theme.semantic[role], "width": 2.1}))
    figure.update_xaxes(title="Layer")
    figure.update_yaxes(title="Internal rerouting magnitude")
    return _finish(
        figure, result, mode=mode, title="Adversarial or evasion rerouting",
        subtitle="Paired clean and altered samples reveal where evidence paths diverge first",
    )


def _cost(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure()
    for efficient, name, role, symbol in (
        (True, "Pareto-efficient", "selected", "diamond"),
        (False, "Dominated", "neutral", "circle"),
    ):
        selected = [row for row in rows if bool(row["pareto_efficient"]) is efficient]
        figure.add_trace(
            go.Scatter(
                x=[row["mean_latency_ms"] for row in selected],
                y=[row["mean_evidence"] for row in selected],
                mode="markers+text",
                text=[row["layer"] for row in selected],
                textposition="top center",
                name=name,
                marker={"color": theme.semantic[role], "symbol": symbol, "size": 11},
                customdata=[[row["evidence_per_ms"], row["evidence_debt"]] for row in selected],
                hovertemplate="Cost: %{x:.4g} ms<br>Evidence: %{y:.4g}<br>Evidence/ms: %{customdata[0]:.4g}<br>Evidence debt: %{customdata[1]:.4g}<extra></extra>",
            )
        )
    figure.update_xaxes(title="Mean measured layer latency (ms)")
    figure.update_yaxes(title="Mean measured evidence signal")
    return _finish(
        figure, result, mode=mode, title="Cost–evidence trade-off",
        subtitle="Pareto points are not dominated by a lower-cost, higher-evidence layer",
    )


def _mediation(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=[row["layer"] for row in rows], y=[row["clean_to_corrupt_recovery"] for row in rows], name="Clean → corrupted recovery", marker_color=theme.semantic["supporting"]))
    if any("corrupt_to_clean_damage" in row for row in rows):
        figure.add_trace(go.Bar(x=[row["layer"] for row in rows], y=[-row.get("corrupt_to_clean_damage", 0.0) for row in rows], name="Corrupted → clean damage", marker_color=theme.semantic["opposing"]))
    figure.add_hline(y=0, line_color=theme.border)
    figure.update_xaxes(title="Patched layer")
    figure.update_yaxes(title="Target-score recovery or damage")
    return _finish(
        figure, result, mode=mode, title="Activation mediation and repair",
        subtitle="Positive bars show score recovered by inserting the clean activation into the corrupted run",
    )


def _topology(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["layers"]
    figure = go.Figure(go.Scatter(x=[row["layer"] for row in rows], y=[row["h0_wasserstein_1"] for row in rows], mode="lines+markers", name="H0 persistence distance", line={"color": theme.semantic["latent"], "width": 2.2}, marker={"size": 8}))
    if any("h1_lifetime_quantile_distance" in row for row in rows):
        figure.add_trace(go.Scatter(x=[row["layer"] for row in rows], y=[row.get("h1_lifetime_quantile_distance") for row in rows], mode="lines+markers", name="H1 lifetime-distribution distance", line={"color": theme.semantic["uncertainty"], "width": 2.2}))
    figure.update_xaxes(title="Layer")
    figure.update_yaxes(title="Persistence summary distance")
    return _finish(
        figure, result, mode=mode, title="Topological representation shift",
        subtitle="H0 is computed exactly from minimum-spanning-tree death times",
    )


def _uncertainty(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = sorted(result.tables["elements"], key=lambda row: row["topk_selection_probability"], reverse=True)[:30]
    figure = go.Figure(
        go.Scatter(
            x=[row["bootstrap_centre"] for row in rows],
            y=[row["label"] for row in rows],
            mode="markers",
            marker={"size": [64.0 + 260.0 * row["topk_selection_probability"] for row in rows], "sizemode": "area", "sizeref": 1.0, "color": [row["topk_selection_probability"] for row in rows], "colorscale": [[0, theme.semantic["neutral"]], [1, theme.semantic["selected"]]], "showscale": True, "colorbar": {"title": "Top-k probability"}},
            error_x={"type": "data", "symmetric": False, "array": [row["bootstrap_high"] - row["bootstrap_centre"] for row in rows], "arrayminus": [row["bootstrap_centre"] - row["bootstrap_low"] for row in rows]},
            customdata=[[row["rank_standard_deviation"], row["positive_probability"], row["negative_probability"]] for row in rows],
            hovertemplate="%{y}<br>Mean: %{x:.4g}<br>Rank SD: %{customdata[0]:.3g}<br>P(+): %{customdata[1]:.3f}<br>P(−): %{customdata[2]:.3f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=theme.border)
    figure.update_xaxes(title="Attribution with bootstrap interval")
    figure.update_yaxes(title="Element", autorange="reversed")
    return _finish(
        figure, result, mode=mode, title="Explanation uncertainty",
        subtitle="Marker area and colour encode empirical top-k selection probability across repeated explanations",
        height=760,
    )


def _ood(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = result.tables["queries"]
    colours = [theme.semantic["contradiction"] if row["high_confidence_internal_anomaly"] else theme.semantic["uncertainty"] if row["reference_percentile"] >= 0.95 else theme.semantic["stable"] for row in rows]
    figure = go.Figure(go.Bar(x=[row["query_index"] for row in rows], y=[row["reference_percentile"] for row in rows], marker_color=colours, customdata=[[row["confidence"], row["mahalanobis_distance"]] for row in rows], hovertemplate="Query %{x}<br>Reference percentile: %{y:.3f}<br>Confidence: %{customdata[0]}<br>Mahalanobis: %{customdata[1]:.4g}<extra></extra>"))
    threshold = float(result.metadata.get("anomaly_percentile_threshold", 0.99))
    figure.add_hline(y=threshold, line_dash="dash", line_color=theme.semantic["opposing"], annotation_text=f"{threshold:.1%} boundary")
    figure.update_xaxes(title="Query sample")
    figure.update_yaxes(title="Internal anomaly percentile", range=[0, 1.02])
    return _finish(
        figure,
        result,
        mode=mode,
        title="High-confidence internal anomaly screening",
        subtitle="Review signal only — not proof of adversarial or out-of-distribution inputs",
    )


def _circuit_wiring(result: AnalysisResult, mode: str):
    """Render measured inter-group couplings with the package's Plotly backend."""

    theme = get_theme(mode)
    nodes = list(result.tables.get("nodes") or [])
    edges = list(result.tables.get("edges") or [])
    layers = list(result.tables.get("layer_order") or [])
    if not layers:
        layers = list(dict.fromkeys(str(node["layer"]) for node in nodes))
    x_map = {layer: index for index, layer in enumerate(layers)}
    y_map: dict[str, float] = {}
    for layer in layers:
        selected = sorted(
            (node for node in nodes if str(node["layer"]) == layer),
            key=lambda row: (not bool(row["retained"]), row.get("mean_unit", 0.0), row["id"]),
        )
        for index, node in enumerate(selected):
            denominator = max(len(selected) - 1, 1)
            y_map[str(node["id"])] = 0.0 if len(selected) == 1 else 1.0 - 2.0 * index / denominator

    evidence_colours = {
        "structural": theme.semantic["neutral"],
        "active": theme.semantic["latent"],
        "sensitive": theme.semantic["uncertainty"],
        "intervention_tested": theme.semantic["selected"],
        "unstable": theme.semantic["opposing"],
        "causal_support": theme.semantic["supporting"],
        "semantically_grounded": theme.semantic["stable"],
    }
    figure = go.Figure()
    shown_levels: set[str] = set()
    for edge in sorted(edges, key=lambda row: float(row.get("mass_normalised", 0.0))):
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in y_map or target not in y_map:
            continue
        level = str(edge.get("evidence_level") or "structural")
        mass = float(edge.get("mass_normalised", 0.0))
        flow = float(edge.get("activation_transfer", edge.get("mass", 0.0)))
        structural = float(edge.get("structural_mass", flow))
        source_label = readable_unit_group_label(source, compact=False)
        target_label = readable_unit_group_label(target, compact=False)
        x0 = float(x_map[str(edge["source_layer"])])
        x1 = float(x_map[str(edge["target_layer"])])
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y_map[source], y_map[target]],
                mode="lines",
                name=level.replace("_", " ").title(),
                legendgroup=level,
                showlegend=level not in shown_levels,
                line={
                    "color": evidence_colours.get(level, theme.semantic["neutral"]),
                    "width": 1.0 + 5.0 * mass,
                },
                opacity=0.35 + 0.6 * mass,
                customdata=[[source_label, target_label, structural, flow, level]] * 2,
                hovertemplate=(
                    "%{customdata[0]} → %{customdata[1]}"
                    "<br>Evidence: %{customdata[4]}"
                    "<br>Structural mass: %{customdata[2]:.4g}"
                    "<br>Activation transfer: %{customdata[3]:.4g}<extra></extra>"
                ),
            )
        )
        shown_levels.add(level)

    for retained, label, colour, symbol in (
        (True, "Retained core group", theme.semantic["selected"], "diamond"),
        (False, "Optional or removed group", theme.semantic["neutral"], "circle-open"),
    ):
        selected = [node for node in nodes if bool(node["retained"]) is retained and str(node["id"]) in y_map]
        if not selected:
            continue
        figure.add_trace(
            go.Scatter(
                x=[float(x_map[str(node["layer"])]) for node in selected],
                y=[y_map[str(node["id"])] for node in selected],
                mode="markers",
                name=label,
                marker={
                    "size": [12.0 + 18.0 * min(abs(float(node.get("score_drop", 0.0))), 1.0) for node in selected],
                    "color": colour,
                    "symbol": symbol,
                    "line": {"width": 1.2, "color": theme.border},
                },
                customdata=[
                    [
                        readable_unit_group_label(str(node["id"]), indices=list(node.get("indices") or []), compact=False),
                        shorten_layer_name(str(node["layer"]), max_length=28),
                        float(node.get("score_drop", 0.0)),
                    ]
                    for node in selected
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>Layer: %{customdata[1]}"
                    "<br>Score drop: %{customdata[2]:.4g}<extra></extra>"
                ),
            )
        )

    figure.update_xaxes(
        title="Layer (forward →)",
        tickmode="array",
        tickvals=list(range(len(layers))),
        ticktext=[shorten_layer_name(layer, max_length=16) for layer in layers],
    )
    figure.update_yaxes(title="Neuron group", showticklabels=False, zeroline=False)
    return _finish(
        figure,
        result,
        mode=mode,
        title="VRAVEN mechanism microscope",
        subtitle="Measured inter-group couplings; edge colour identifies evidence level and width reflects normalised mass",
        height=max(560, min(900, 180 + 36 * max(len(nodes), 4))),
    )


def _internal_anatomy(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    palette = [
        theme.semantic["selected"],
        theme.semantic["latent"],
        theme.semantic["uncertainty"],
        theme.semantic["opposing"],
        theme.semantic["supporting"],
        theme.semantic["stable"],
        theme.semantic["neutral"],
    ]
    units = list(result.tables["units"])
    layers = list(dict.fromkeys(row["layer"] for row in units))
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Input contribution proxy", "Output pre-activation |z|"),
        horizontal_spacing=0.12,
    )
    input_rows = [row for row in units if row["side"] == "input"]
    output_rows = [row for row in units if row["side"] == "output"]
    for layer_index, layer in enumerate(layers):
        colour = palette[layer_index % len(palette)]
        selected_in = [row for row in input_rows if row["layer"] == layer]
        selected_out = [row for row in output_rows if row["layer"] == layer]
        if selected_in:
            figure.add_trace(
                go.Bar(
                    x=[f"{layer}:in[{row['unit']}]" for row in selected_in],
                    y=[row["contribution"] for row in selected_in],
                    name=f"{layer} input",
                    marker_color=colour,
                    hovertemplate=(
                        "%{x}<br>contribution=%{y:.4g}"
                        "<br>a=%{customdata[0]:.4g}"
                        "<br>||W[:,j]||=%{customdata[1]:.4g}<extra></extra>"
                    ),
                    customdata=[
                        [row["activation"], row["outgoing_weight_l2"]] for row in selected_in
                    ],
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if selected_out:
            figure.add_trace(
                go.Bar(
                    x=[f"{layer}:out[{row['unit']}]" for row in selected_out],
                    y=[abs(row["preactivation"]) for row in selected_out],
                    name=f"{layer} |z|",
                    marker_color=colour,
                    hovertemplate=(
                        "%{x}<br>|z|=%{y:.4g}"
                        "<br>z=%{customdata[0]:.4g}"
                        "<br>||W[j,:]||=%{customdata[1]:.4g}<extra></extra>"
                    ),
                    customdata=[
                        [row["preactivation"], row["incoming_weight_l2"]] for row in selected_out
                    ],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
    figure.update_xaxes(title="Unit", tickangle=-35, row=1, col=1)
    figure.update_xaxes(title="Unit", tickangle=-35, row=1, col=2)
    figure.update_yaxes(title="a ⊙ ∂||z||²/∂a", row=1, col=1)
    figure.update_yaxes(title="|W a + b|", row=1, col=2)
    return _finish(
        figure,
        result,
        mode=mode,
        title="Internal Linear anatomy",
        subtitle="Live W×a anatomy per Linear module; left is local energy routing, right is pre-activation magnitude",
        height=640,
    )


def _class_conditional_unit_field(result: AnalysisResult, mode: str) -> go.Figure:
    theme = get_theme(mode)
    rows = list(result.tables["field"])
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    units = sorted({int(row["unit"]) for row in rows})
    value_map = {(row["layer"], int(row["unit"])): float(row["hedges_g"]) for row in rows}
    matrix = np.array(
        [[value_map.get((layer, unit), np.nan) for unit in units] for layer in layers],
        dtype=float,
    )
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = max(limit, 1e-12)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(unit) for unit in units],
            y=layers,
            colorscale=[
                [0.0, theme.semantic["opposing"]],
                [0.5, theme.raised_panel],
                [1.0, theme.semantic["selected"]],
            ],
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Hedges g"},
            hovertemplate="Layer %{y}<br>Unit %{x}<br>Hedges g=%{z:.4g}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Unit index")
    figure.update_yaxes(title="Layer", autorange="reversed")
    return _finish(
        figure,
        result,
        mode=mode,
        title="Class-conditional unit field",
        subtitle=(
            f"Signed standardised contrast ({result.metrics['class_a']} → "
            f"{result.metrics['class_b']}); observational, not causal"
        ),
        height=max(480, min(900, 140 + 48 * len(layers))),
    )


def figure_for_result(result: AnalysisResult, *, mode: str = "publication_light"):
    """Render the scientifically matched figure for a DarkSide result."""

    if mode not in {"interactive_dark", "publication_light"}:
        raise ValueError("mode must be 'interactive_dark' or 'publication_light'")
    dispatch = {
        "decision_circuitry": _decision_circuit,
        "circuit_multiplicity": _circuit_multiplicity,
        "shortcut_detection": _shortcut,
        "layerwise_shortcut_probe": _shortcut,
        "class_conditional_internals": _class_conditional,
        "class_conditional_unit_field": _class_conditional_unit_field,
        "circuit_wiring": _circuit_wiring,
        "internal_anatomy": _internal_anatomy,
        "representation_drift": _drift,
        "failure_anatomy": _failure,
        "stability_geometry": _stability,
        "prediction_depth": _prediction_depth,
        "unused_capacity": _capacity,
        "adversarial_rerouting": _adversarial,
        "cost_evidence_tradeoff": _cost,
        "activation_mediation": _mediation,
        "topological_shift": _topology,
        "explanation_uncertainty": _uncertainty,
        "high_confidence_ood": _ood,
    }
    if result.analysis == "cross_method_disagreement":
        return _matrix(result, mode, key="topk_jaccard", title="Cross-method disagreement", subtitle="Top-k overlap on the same indexed explanatory surface")
    if result.analysis == "causal_synergy":
        return _matrix(result, mode, key="interaction", title="Causal synergy and redundancy", subtitle="Positive values are super-additive; negative values indicate redundancy or compensation")
    if result.analysis == "mechanism_transfer":
        return _matrix(result, mode, key="signed_cosine", title="Mechanism transfer", subtitle="Signed cosine similarity of aligned intervention-effect maps")
    try:
        return dispatch[result.analysis](result, mode)
    except KeyError as exc:
        raise KeyError(f"No DarkSide figure is registered for {result.analysis!r}") from exc
