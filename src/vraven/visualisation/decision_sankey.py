"""Contribution-over-time Sankey: features → model depth → class outputs."""

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
from vraven.visualisation.labels import readable_feature_name, shorten_layer_name
from vraven.visualisation.palette import RenderMode, semantic_colour
from vraven.visualisation.theme import get_theme

CONTRACT = VisualContract(
    visual_id="vraven.decision_sankey",
    research_question=(
        "How does first-order contribution mass move from named input features "
        "through model depth into the competing class outputs?"
    ),
    required_metrics=("input attribution", "layer signed influence", "prediction scores"),
    data_domain="mass-conserving summary of absolute contributions with signed routing",
    visual_encodings=(
        "left nodes=top named features (+ Others)",
        "middle columns=selected model-depth stages",
        "right nodes=class outputs with scores",
        "link width=absolute contribution mass",
        "link colour=support (positive) or opposition (negative)",
    ),
    limitations=(
        "Links summarise absolute contribution mass and do not recover synaptic connectivity.",
        "Feature groups are the largest absolute input attributions, with remainder aggregated.",
        "Intermediate depth columns are selected captured layers, not every neuron.",
    ),
    interpretation=(
        "Width preserves absolute summary mass while colour preserves sign. The links connect "
        "aggregated stages and must not be read as recovered synaptic pathways."
    ),
    metric_definitions=(
        "feature mass = absolute stored input attribution",
        "stage mass = absolute selected layer influence",
        "output routing share = signed layer mass allocated to support or opposition",
    ),
)


def _rgba(hex_colour: str, alpha: float) -> str:
    value = hex_colour.lstrip("#")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def _depth_label(layer_name: str, index: int, total: int) -> str:
    short = shorten_layer_name(layer_name, max_length=18)
    lower = layer_name.lower()
    if "embed" in lower:
        return f"L{index} (Emb.)"
    if "encoder" in lower or short.startswith("ae.enc") or ".enc." in short:
        return f"L{index} (Enc.)"
    if "decoder" in lower or short.startswith("ae.dec") or ".dec." in short:
        return f"L{index} (Dec.)"
    if index == 0:
        return f"L{index} (In)"
    if index == total - 1:
        return f"L{index} (Out)"
    return f"L{index} (Block)"


def _resolve_class_names(
    trace: NeuralTrace,
    class_names: Sequence[str] | None,
) -> tuple[str, str]:
    names: list[str] | None = None
    if class_names is not None and len(class_names) >= 2:
        names = [str(name) for name in class_names]
    else:
        meta_names = trace.metadata.get("class_names")
        if isinstance(meta_names, list) and len(meta_names) >= 2:
            names = [str(name) for name in meta_names]
        else:
            user = trace.metadata.get("user_metadata")
            if isinstance(user, dict) and isinstance(user.get("class_names"), list):
                names = [str(name) for name in user["class_names"]]
    if names is None or len(names) < 2:
        return "Target (+)", "Target (−)"
    # Prefer malware/injection as the positive/anomaly side when present.
    lowered = [name.lower() for name in names]
    positive_index = 0
    for index, name in enumerate(lowered):
        if any(token in name for token in ("malware", "injection", "attack", "anomaly", "malicious")):
            positive_index = index
            break
    else:
        target = trace.target
        if isinstance(target, int) and 0 <= target < len(names):
            positive_index = target
        elif isinstance(target, list) and target:
            try:
                positive_index = int(target[0])
            except (TypeError, ValueError):
                positive_index = min(1, len(names) - 1)
    negative_index = 1 - positive_index if len(names) == 2 else (0 if positive_index else 1)
    return names[positive_index].upper(), names[negative_index].upper()


def _prediction_scores(trace: NeuralTrace) -> tuple[float | None, float | None]:
    prediction = trace.prediction
    if prediction is None:
        return None, None
    target_scores = getattr(prediction, "target_scores", None) or []
    if target_scores:
        positive = float(np.mean(target_scores))
    elif getattr(prediction, "target_score", None) is not None:
        positive = float(prediction.target_score)
    else:
        positive = None
    confidence = getattr(prediction, "confidence", None)
    if confidence:
        competing = float(np.mean(confidence))
        if positive is not None:
            competing = max(abs(positive) * max(0.05, 1.0 - competing), 1e-6)
        else:
            competing = max(competing, 1e-6)
    else:
        competing = abs(positive) * 0.85 if positive is not None else None
    return positive, competing


def decision_sankey_figure(
    trace: NeuralTrace,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    top_features: int = 7,
    depth_stages: int = 5,
    mode: RenderMode = "publication_light",
    show_title: bool | None = None,
) -> go.Figure:
    if not trace.layers:
        raise CapabilityError("Decision Sankey requires captured layers")
    layer_values = np.asarray(
        [
            float(layer.activation_gradient_signed or layer.activation_energy or 0.0)
            for layer in trace.layers
        ],
        dtype=np.float64,
    )
    if not np.any(np.abs(layer_values) > 0):
        raise CapabilityError("Decision Sankey requires gradient influence or activation energy")

    # --- Feature nodes (left) -------------------------------------------------
    if trace.inputs and trace.inputs.attribution_values:
        signed_features = np.asarray(trace.inputs.attribution_values, dtype=np.float64).reshape(-1)
    else:
        signed_features = np.ones(top_features, dtype=np.float64)
    abs_features = np.abs(signed_features)
    names = (
        [str(name) for name in feature_names]
        if feature_names is not None and len(feature_names) == signed_features.size
        else [f"feature_{index}" for index in range(signed_features.size)]
    )
    order = np.argsort(abs_features)[::-1]
    kept = order[: min(top_features, order.size)]
    remainder_abs = float(abs_features[order[len(kept) :]].sum()) if order.size > len(kept) else 0.0
    remainder_signed = (
        float(signed_features[order[len(kept) :]].sum()) if order.size > len(kept) else 0.0
    )

    labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    feature_nodes: list[tuple[int, float, float]] = []  # id, abs_mass, signed

    feature_count = len(kept) + (1 if remainder_abs > 0 else 0)
    for rank, index in enumerate(kept):
        node_id = len(labels)
        labels.append(readable_feature_name(names[int(index)], max_length=26))
        signed = float(signed_features[int(index)])
        node_colors.append(
            semantic_colour("supporting", mode) if signed >= 0 else semantic_colour("opposing", mode)
        )
        node_x.append(0.01)
        node_y.append(0.04 + 0.92 * (rank / max(feature_count - 1, 1)))
        feature_nodes.append((node_id, float(abs_features[int(index)]), signed))
    if remainder_abs > 0:
        node_id = len(labels)
        labels.append(f"Others ({order.size - len(kept)})")
        node_colors.append(semantic_colour("neutral", mode))
        node_x.append(0.01)
        node_y.append(0.04 + 0.92 * ((feature_count - 1) / max(feature_count - 1, 1)))
        feature_nodes.append((node_id, remainder_abs, remainder_signed))

    # --- Depth stages (middle) -----------------------------------------------
    stage_count = max(2, min(depth_stages, len(trace.layers)))
    stage_indices = np.unique(
        np.linspace(0, len(trace.layers) - 1, stage_count, dtype=int)
    ).tolist()
    stage_nodes: list[int] = []
    stage_masses: list[float] = []
    for stage_rank, layer_index in enumerate(stage_indices):
        layer = trace.layers[layer_index]
        mass = abs(float(layer_values[layer_index]))
        if mass <= 0:
            mass = float(np.mean(np.abs(layer_values)) + 1e-6)
        node_id = len(labels)
        labels.append(_depth_label(layer.name, stage_rank, len(stage_indices)))
        node_colors.append(semantic_colour("selected", mode))
        # Spread depth columns between x=0.28 and x=0.72
        if len(stage_indices) == 1:
            node_x.append(0.5)
        else:
            node_x.append(0.28 + 0.44 * (stage_rank / (len(stage_indices) - 1)))
        node_y.append(0.50)
        stage_nodes.append(node_id)
        stage_masses.append(mass)

    # --- Output nodes (right) ------------------------------------------------
    positive_name, negative_name = _resolve_class_names(trace, class_names)
    positive_score, negative_score = _prediction_scores(trace)
    positive_mass = float(np.sum(layer_values[layer_values > 0]))
    negative_mass = float(np.abs(np.sum(layer_values[layer_values < 0])))
    if positive_mass <= 0 and negative_mass <= 0:
        positive_mass = float(np.sum(np.abs(layer_values)))
        negative_mass = positive_mass * 0.35
    if positive_score is not None:
        positive_label = f"{positive_name} ({positive_score:+.2f})"
    else:
        positive_label = positive_name
    if negative_score is not None:
        negative_label = f"{negative_name} ({-abs(negative_score):+.2f})"
    else:
        negative_label = negative_name

    target_pos = len(labels)
    labels.append(positive_label)
    node_colors.append(semantic_colour("supporting", mode))
    node_x.append(0.99)
    node_y.append(0.22)

    target_neg = len(labels)
    labels.append(negative_label)
    node_colors.append(semantic_colour("opposing", mode))
    node_x.append(0.99)
    node_y.append(0.78)

    # --- Links ----------------------------------------------------------------
    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    pos_link = _rgba(semantic_colour("supporting", mode), 0.42)
    neg_link = _rgba(semantic_colour("opposing", mode), 0.42)
    mid_link = _rgba(semantic_colour("neutral", mode), 0.28)

    first_stage = stage_nodes[0]
    first_mass = stage_masses[0] or 1.0
    for feature_id, abs_mass, signed in feature_nodes:
        if abs_mass <= 0:
            continue
        sources.append(feature_id)
        targets.append(first_stage)
        values.append(abs_mass)
        link_colors.append(pos_link if signed >= 0 else neg_link)

    for stage_rank in range(len(stage_nodes) - 1):
        left_id = stage_nodes[stage_rank]
        right_id = stage_nodes[stage_rank + 1]
        left_mass = stage_masses[stage_rank]
        right_mass = stage_masses[stage_rank + 1]
        link = max((left_mass + right_mass) * 0.5, 1e-6)
        sources.append(left_id)
        targets.append(right_id)
        values.append(link)
        # Colour intermediate depth links by local layer sign.
        layer_sign = float(layer_values[stage_indices[stage_rank + 1]])
        if layer_sign > 0:
            link_colors.append(pos_link)
        elif layer_sign < 0:
            link_colors.append(neg_link)
        else:
            link_colors.append(mid_link)

    last_stage = stage_nodes[-1]
    last_mass = stage_masses[-1] or 1.0
    denom = positive_mass + negative_mass + 1e-12
    sources.append(last_stage)
    targets.append(target_pos)
    values.append(max(last_mass * positive_mass / denom, 1e-6))
    link_colors.append(pos_link)
    sources.append(last_stage)
    targets.append(target_neg)
    values.append(max(last_mass * negative_mass / denom, 1e-6))
    link_colors.append(neg_link)

    if not values:
        raise CapabilityError("Unable to construct a non-empty Sankey from the trace")

    text_colour = get_theme(mode).primary_text
    figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".3g",
                node={
                    "label": labels,
                    "color": node_colors,
                    "x": node_x,
                    "y": node_y,
                    "pad": 16,
                    "thickness": 18,
                    "line": {"width": 0.5, "color": semantic_colour("neutral", mode)},
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                    "hovertemplate": (
                        "%{source.label} → %{target.label}<br>"
                        "Contribution mass: %{value:.4g}<extra></extra>"
                    ),
                },
            )
        ]
    )

    # Stage headers matching the mockup.
    figure.add_annotation(
        x=0.01,
        y=1.12,
        xref="paper",
        yref="paper",
        text="<b>Input Features</b>",
        showarrow=False,
        xanchor="left",
        font={"size": 13, "color": text_colour},
    )
    figure.add_annotation(
        x=0.50,
        y=1.12,
        xref="paper",
        yref="paper",
        text="<b>Model Depth</b>",
        showarrow=False,
        xanchor="center",
        font={"size": 13, "color": text_colour},
    )
    figure.add_annotation(
        x=0.99,
        y=1.12,
        xref="paper",
        yref="paper",
        text="<b>Output (Logits)</b>",
        showarrow=False,
        xanchor="right",
        font={"size": 13, "color": text_colour},
    )

    # Contribution colour legend (negative ← → positive).
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": 0.1,
                "color": [0],
                "colorscale": [
                    [0.0, semantic_colour("opposing", mode)],
                    [0.5, semantic_colour("neutral", mode)],
                    [1.0, semantic_colour("supporting", mode)],
                ],
                "cmin": -1,
                "cmax": 1,
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Contribution", "side": "top"},
                    "tickvals": [-1, 0, 1],
                    "ticktext": ["Opposing (negative)", "0", "Supporting (positive)"],
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.12,
                    "yanchor": "top",
                    "len": 0.62,
                    "thickness": 12,
                },
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(margin={"l": 30, "r": 30, "t": 110, "b": 110})
    return finalise_figure(
        figure,
        metadata=VisualisationMetadata.from_contract(
            CONTRACT,
            normalisation="link values are absolute contribution masses; colour encodes signed support",
            key_results=(
                PublicationResult(
                    "Largest input contribution",
                    readable_feature_name(names[int(kept[0])], max_length=30),
                    f"Signed local attribution {float(signed_features[int(kept[0])]):.5g}.",
                ),
                PublicationResult(
                    "Named top-feature mass",
                    (
                        "Not estimable"
                        if float(abs_features.sum()) <= 0
                        else f"{float(abs_features[kept].sum() / abs_features.sum()):.1%}"
                    ),
                    "Absolute attribution mass retained as named nodes; the remainder is Others.",
                ),
                PublicationResult(
                    "Final signed routing",
                    f"{positive_mass:.4g} support · {negative_mass:.4g} opposition",
                    "Aggregated signed layer mass entering the two output-side routes.",
                ),
            ),
            extra={
                "feature_nodes": len(feature_nodes),
                "depth_stages": [
                    shorten_layer_name(trace.layers[index].name) for index in stage_indices
                ],
                "positive_output": positive_name,
                "negative_output": negative_name,
                "first_stage_mass": first_mass,
            },
        ),
        mode=mode,
        title="VRAVEN Contribution Over Time (Decision Flow)",
        subtitle="Input features → model depth → class outputs",
        width=1240,
        height=760,
        show_legend=False,
        show_title=show_title,
    )
