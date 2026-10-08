"""Measured circuit wiring from retained DarkSide cores + live Linear weights."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

from vraven.core.tree import call_model
from vraven.visualisation.labels import readable_unit_group_label, shorten_layer_name

from .contracts import AnalysisResult
from .interventions import evaluation_mode


def _parse_layer(layer_field: str) -> str:
    # decision_circuitry stores "net.0#0"
    return str(layer_field).split("#", 1)[0]


def build_circuit_wiring(
    model: nn.Module,
    inputs: Any,
    circuitry: AnalysisResult,
    *,
    include_removed: bool = False,
    minimum_edge_mass: float = 0.0,
) -> AnalysisResult:
    """Build a neuron-group wiring graph from an ε-sufficient circuit core.

    Nodes are retained (optionally removed) unit groups from
    ``discover_decision_circuitry``. Edges between consecutive Linear layers
    use the measured mass:

        mass(G → H) = Σ_{i∈G, j∈H} |W_{j i}| · |a_i|

    where ``a`` is the captured input activation to the downstream Linear layer
    and ``W`` is that layer's weight matrix. This is not a claim of unique
    causal edges; it is the measured weight×activation coupling of the core.
    """

    if circuitry.analysis != "decision_circuitry":
        raise ValueError("build_circuit_wiring requires a decision_circuitry result")
    groups = list(circuitry.tables.get("groups") or [])
    if not groups:
        raise ValueError("decision_circuitry result has no groups table")

    selected = [row for row in groups if row.get("retained") or include_removed]
    if not any(row.get("retained") for row in selected):
        raise ValueError("No retained circuit groups are available to wire")

    # Capture inputs to every Linear module
    acts: dict[str, torch.Tensor] = {}
    modules = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    }
    handles = []
    for name, module in modules.items():

        def _hook(_m, inp, _out, *, layer_name: str = name):
            if isinstance(inp[0], torch.Tensor):
                acts[layer_name] = inp[0].detach()

        handles.append(module.register_forward_hook(_hook))
    try:
        with evaluation_mode(model):
            call_model(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    # Order layers by circuit table order
    layer_order = list(dict.fromkeys(_parse_layer(row["layer"]) for row in selected))
    nodes = []
    for row in selected:
        layer = _parse_layer(row["layer"])
        indices = list(row["indices"])
        nodes.append(
            {
                "id": row["group"],
                "label": readable_unit_group_label(row["group"], indices=indices, compact=True),
                "label_full": readable_unit_group_label(
                    row["group"], indices=indices, compact=False
                ),
                "layer": layer,
                "layer_short": shorten_layer_name(layer, max_length=18),
                "layer_field": row["layer"],
                "indices": indices,
                "retained": bool(row["retained"]),
                "score_drop": float(row["individual_score_drop"]),
                "mean_unit": float(np.mean(indices)) if indices else 0.0,
            }
        )

    nodes_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_layer[node["layer"]].append(node)

    edges: list[dict[str, Any]] = []
    linear_layers = [name for name in layer_order if name in modules]
    for upstream, downstream in zip(linear_layers, linear_layers[1:], strict=False):
        module = modules[downstream]
        weight = module.weight.detach().float().cpu().numpy()  # [out, in]
        activation = acts.get(downstream)
        if activation is None:
            continue
        # Linear in_features are the last axis; average away batch/token axes.
        act = activation.detach().float()
        if act.ndim == 0:
            continue
        if act.ndim == 1:
            a = act.cpu().numpy()
        else:
            a = act.reshape(-1, act.shape[-1]).mean(dim=0).cpu().numpy()
        if a.shape[0] != weight.shape[1]:
            # ReLU-separated Linear: input width should match W.in_features
            continue
        for src in nodes_by_layer.get(upstream, []):
            if not src["retained"] and not include_removed:
                continue
            src_idx = [i for i in src["indices"] if 0 <= i < a.shape[0]]
            if not src_idx:
                continue
            for dst in nodes_by_layer.get(downstream, []):
                if not dst["retained"] and not include_removed:
                    continue
                dst_idx = [j for j in dst["indices"] if 0 <= j < weight.shape[0]]
                if not dst_idx:
                    continue
                # Structural mass M = Σ|W|; sample flow F = Σ|a_i W_ji|; signed F = Σ a_i W_ji
                block = weight[np.ix_(dst_idx, src_idx)]
                a_block = a[src_idx]
                structural_mass = float(np.sum(np.abs(block)))
                activation_transfer = float(np.sum(np.abs(block) * np.abs(a_block)[None, :]))
                signed_transfer = float(np.sum(block * a_block[None, :]))
                # Keep drawing/ranking on sample-dependent flow; filter zeros with structural floor.
                if activation_transfer < minimum_edge_mass and structural_mass < minimum_edge_mass:
                    continue
                edges.append(
                    {
                        "source": src["id"],
                        "target": dst["id"],
                        "source_layer": upstream,
                        "target_layer": downstream,
                        "mass": activation_transfer,
                        "structural_mass": structural_mass,
                        "activation_transfer": activation_transfer,
                        "signed_transfer": signed_transfer,
                        "mean_abs_weight": float(np.mean(np.abs(block))),
                        "retained_path": bool(src["retained"] and dst["retained"]),
                    }
                )

    if edges:
        max_flow = max(float(edge["activation_transfer"]) for edge in edges) or 1.0
        max_struct = max(float(edge["structural_mass"]) for edge in edges) or 1.0
        for edge in edges:
            edge["mass_normalised"] = float(edge["activation_transfer"]) / max_flow
            edge["structural_mass_normalised"] = float(edge["structural_mass"]) / max_struct

    return AnalysisResult(
        analysis="circuit_wiring",
        metrics={
            "nodes": len(nodes),
            "edges": len(edges),
            "retained_nodes": sum(1 for node in nodes if node["retained"]),
            "layers": len(layer_order),
            "total_edge_mass": float(sum(float(edge["activation_transfer"]) for edge in edges)),
            "total_structural_mass": float(sum(float(edge["structural_mass"]) for edge in edges)),
        },
        tables={"nodes": nodes, "edges": edges, "layer_order": layer_order},
        metadata={
            "claim_type": "measured Linear structural mass and sample-dependent activation flow of an intervention core",
            "parent_analysis": "decision_circuitry",
            "edge_formula_structural": "M(G→H)=Σ_{i∈G,j∈H}|W_ji|",
            "edge_formula_flow": "F(G→H)=Σ_{i∈G,j∈H}|a_i W_ji|",
            "edge_formula_signed": "F_signed(G→H)=Σ_{i∈G,j∈H}a_i W_ji",
        },
        limitations=[
            "Edges are measured Linear couplings of the declared core, not proof of unique causal pathways.",
            "Structural mass and activation transfer are distinct: wired≠active.",
            "Non-Linear modules between layers are not expanded into explicit edges.",
            "Group aggregation can hide opposing units inside the same bag.",
        ],
        warnings=list(circuitry.warnings),
    )
