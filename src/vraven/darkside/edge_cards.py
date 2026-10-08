"""Precompute interactive CONNECTION cards for DarkSide circuit-wiring edges.

All quantities are measured from the live model + focus sample (and optional
reference batch). Semantic concept names are never invented.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from vraven.core.config import VRavenConfig
from vraven.core.device import normalise_model_inputs
from vraven.core.explainer import explain
from vraven.decompiler.semantics import OperatorSemanticsRegistry
from vraven.visualisation.labels import readable_unit_group_label, shorten_layer_name

from .contracts import AnalysisResult, UnitGroup
from .interventions import InterventionEngine


def _edge_key(source: str, target: str) -> str:
    return f"{source}\u2192{target}"


def _unit_group_from_node(node: Mapping[str, Any]) -> UnitGroup:
    layer_field = str(node.get("layer_field") or node["layer"])
    if "#" in layer_field:
        layer_name, call_text = layer_field.rsplit("#", 1)
        call_index = int(call_text)
    else:
        layer_name = layer_field
        call_index = 0
    indices = tuple(int(v) for v in node["indices"])
    return UnitGroup(
        layer_name=layer_name,
        indices=indices,
        call_index=call_index,
        label=str(node.get("label_full") or readable_unit_group_label(node["id"], indices=indices)),
    )


def _percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _contribution_map(model: nn.Module, inputs: Any, target: int | None) -> dict[str, np.ndarray]:
    """Map layer name → unit_contributions vector from one gradient pass."""

    try:
        report = explain(model, inputs, target=target, mode="gradient")
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, np.ndarray] = {}
    for layer in report.trace.layers:
        if layer.unit_contributions is None:
            continue
        name = str(layer.name).split("#", 1)[0]
        out[name] = np.asarray(layer.unit_contributions, dtype=np.float64)
    return out


def _group_sensitivity(contrib: np.ndarray | None, indices: Sequence[int]) -> float | None:
    if contrib is None or contrib.size == 0:
        return None
    selected = [i for i in indices if 0 <= int(i) < contrib.size]
    if not selected:
        return None
    return float(np.sum(contrib[selected]))


def _operator_role(layer_name: str, module: nn.Module | None) -> dict[str, str]:
    registry = OperatorSemanticsRegistry()
    module_type = type(module).__name__ if module is not None else "Linear"
    entry = registry.resolve(layer_name, op=module_type, module_type=module_type)
    return {
        "role": entry.role.value,
        "family": entry.family,
        "description": entry.description,
        "claim": "operator semantics of the Linear hop, not learned feature meaning",
    }


def _class_label(class_names: Mapping[int, str] | None, target: int | None) -> str:
    if target is None:
        return "target class"
    if class_names and int(target) in class_names:
        return str(class_names[int(target)])
    return f"class {int(target)}"


def _decision_margin(logits: torch.Tensor | None, target: int | None) -> float | None:
    """Decision margin m = z_target − max_{j≠target} z_j (binary: z_t − z_other)."""

    if logits is None or target is None:
        return None
    tensor = logits.detach().float().cpu()
    if tensor.ndim == 0:
        return None
    if tensor.ndim == 1:
        # [classes] or [batch] binary-logit — treat length-2 as class logits
        if tensor.numel() == 2:
            values = tensor
        elif tensor.numel() == 1:
            # single logit: margin vs 0 (sigmoid decision)
            z = float(tensor.item())
            return z if int(target) == 1 else -z
        else:
            values = tensor
    elif tensor.ndim == 2:
        values = tensor[0]
    else:
        return None
    t = int(target)
    if t < 0 or t >= int(values.numel()):
        return None
    if values.numel() == 1:
        z = float(values.item())
        return z if t == 1 else -z
    z_t = float(values[t].item())
    others = torch.cat([values[:t], values[t + 1 :]])
    if others.numel() == 0:
        return z_t
    return z_t - float(others.max().item())


def _intervention_outcome(card_intervention: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify pathway ablation result (separate from display status)."""

    if not card_intervention or not card_intervention.get("computed"):
        return {
            "code": "NOT_TESTED",
            "tested": False,
            "detail": "No pathway intervention was computed for this edge (outside top-N by mass).",
        }
    flip_rate = card_intervention.get("flip_rate")
    mean_delta = card_intervention.get("mean_score_delta")
    focus_delta = card_intervention.get("focus_score_delta")
    delta = mean_delta if mean_delta is not None else focus_delta
    flips = float(flip_rate) if flip_rate is not None else (
        1.0 if card_intervention.get("focus_prediction_flipped") else 0.0
    )
    strong_drop = delta is not None and float(delta) <= -1e-3
    if strong_drop and flips >= 0.5:
        return {
            "code": "INTERVENTION_SUPPORTED",
            "tested": True,
            "detail": "Joint endpoint ablation moves the target score and flips often on the cohort.",
        }
    if delta is not None and abs(float(delta)) > 1e-6:
        return {
            "code": "WEAK_INTERVENTION",
            "tested": True,
            "detail": "Endpoint ablation changes the score, but flip rate or effect size is limited.",
        }
    return {
        "code": "NULL_INTERVENTION",
        "tested": True,
        "detail": "Endpoint ablation did not move the decision materially.",
    }


def _status_from_evidence(
    ladder: Mapping[str, Any],
    intervention_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Status = highest evidence rung reached (not stuck at STRUCTURAL ONLY)."""

    level = str(ladder.get("level") or "structural")
    catalog = {
        "structural": (
            "STRUCTURAL_ONLY",
            "STRUCTURAL ONLY",
            "Measured Linear coupling only — no sample flow above threshold.",
        ),
        "active": (
            "ACTIVE_FLOW",
            "ACTIVE (SAMPLE FLOW)",
            "Connection carries sample-dependent activation transfer on the focus input.",
        ),
        "sensitive": (
            "SENSITIVITY_SUPPORTED",
            "OBSERVATIONAL + FIRST-ORDER SENSITIVITY",
            "Structure, live flow, and first-order target-score sensitivity are present.",
        ),
        "intervention_tested": (
            "INTERVENTION_TESTED",
            "INTERVENTION TESTED",
            "Pathway endpoints were ablated; effect did not meet causal-support criteria.",
        ),
        "unstable": (
            "WEAK_INTERVENTION",
            "INTERVENTION TESTED — WEAK / UNSTABLE",
            "Ablation moves the score, but flip rate or effect size is limited.",
        ),
        "causal_support": (
            "CAUSAL_SUPPORT",
            "CAUSAL SUPPORT (INTERVENTION)",
            "Intervention-supported pathway under the declared ablation criteria.",
        ),
        "semantically_grounded": (
            "SEMANTICALLY_GROUNDED",
            "SEMANTICALLY GROUNDED",
            "Association evidence is attached; not feature identity by itself.",
        ),
    }
    code, label, detail = catalog.get(level, catalog["structural"])
    gaps: list[str] = []
    if not ladder.get("intervention_tested"):
        gaps.append("Not intervention tested")
    if not ladder.get("causal_support"):
        gaps.append("Not causally validated")
    if not ladder.get("semantically_grounded"):
        gaps.append("No semantic grounding")
    return {
        "code": code,
        "label": label,
        "detail": detail,
        "level": level,
        "gaps": gaps,
        "intervention_outcome": str(intervention_outcome.get("code") or "NOT_TESTED"),
    }


def _evidence_ladder(
    *,
    structural_mass: float,
    activation_transfer: float,
    pathway_sens: float | None,
    intervention: Mapping[str, Any] | None,
    status_code: str,
    semantic_available: bool,
    hop_flow_total: float,
) -> dict[str, Any]:
    """Progressive evidence rungs for a connection."""

    active = float(activation_transfer) > max(1e-12, 1e-6 * max(hop_flow_total, 1.0))
    sensitive = pathway_sens is not None and abs(float(pathway_sens)) > 1e-9
    tested = bool(intervention and intervention.get("computed"))
    causal = status_code == "INTERVENTION_SUPPORTED"
    semantic = bool(semantic_available)
    unstable = status_code == "WEAK_INTERVENTION"

    rungs = [
        ("structural", True),
        ("active", active),
        ("sensitive", sensitive),
        ("intervention_tested", tested),
        ("causal_support", causal),
        ("semantically_grounded", semantic),
    ]
    level = "structural"
    for name, ok in rungs:
        if ok:
            level = name
    if unstable and level in {"intervention_tested", "causal_support"}:
        level = "unstable"

    return {
        "structural": True,
        "active": active,
        "sensitive": sensitive,
        "intervention_tested": tested,
        "causal_support": causal,
        "semantically_grounded": semantic,
        "unstable": unstable,
        "level": level,
        "rungs": [
            {"id": "structural", "label": "STRUCTURAL", "ok": True},
            {"id": "active", "label": "ACTIVE", "ok": active},
            {"id": "sensitive", "label": "SENSITIVE", "ok": sensitive},
            {"id": "intervention_tested", "label": "INTERVENTION", "ok": tested},
            {"id": "causal_support", "label": "CAUSAL", "ok": causal},
            {"id": "semantically_grounded", "label": "SEMANTIC", "ok": semantic},
        ],
    }


def _contribution_direction(
    *,
    target_label: str,
    intervention: Mapping[str, Any] | None,
    pathway_sens: float | None,
) -> dict[str, Any]:
    """Direction relative to decision margin / target score.

    Intervention: ablating the pathway and observing Δm (or Δ target score).
    If Δm < 0 (score drops), the pathway supported the target class.
    """

    if intervention and intervention.get("computed"):
        d_margin = intervention.get("focus_margin_delta")
        d_score = intervention.get("focus_score_delta")
        delta = float(d_margin) if d_margin is not None else (
            float(d_score) if d_score is not None else None
        )
        if delta is None:
            return {
                "label": "unavailable",
                "polarity": "unavailable",
                "basis": "intervention",
                "delta_margin": None,
                "definition": "m = z_target − max_{j≠target} z_j; direction from Δm under ablation",
            }
        if delta < -1e-9:
            polarity, label = "supports", f"Supports {target_label}"
        elif delta > 1e-9:
            polarity, label = "opposes", f"Opposes {target_label}"
        else:
            polarity, label = "neutral", "Neutral"
        return {
            "label": label,
            "polarity": polarity,
            "basis": "intervention_margin" if d_margin is not None else "intervention_score",
            "delta_margin": float(d_margin) if d_margin is not None else None,
            "delta_score": float(d_score) if d_score is not None else None,
            "definition": (
                "Ablation Δm < 0 ⇒ pathway supported target "
                "(m = z_target − max other logits)"
            ),
        }

    if pathway_sens is None:
        return {
            "label": "unavailable",
            "polarity": "unavailable",
            "basis": "none",
            "delta_margin": None,
            "definition": "m = z_target − max_{j≠target} z_j",
        }
    if pathway_sens > 1e-9:
        polarity, label = "supports", f"Supports {target_label}"
    elif pathway_sens < -1e-9:
        polarity, label = "opposes", f"Opposes {target_label}"
    else:
        polarity, label = "neutral", "Neutral"
    return {
        "label": label,
        "polarity": polarity,
        "basis": "first_order_target_score",
        "delta_margin": None,
        "definition": (
            "First-order unit contributions to target score "
            "(proxy until pathway intervention computes Δm)"
        ),
    }


def _slice_batch(inputs: Any, index: int) -> Any:
    args, kwargs = normalise_model_inputs(inputs)
    if args and isinstance(args[0], torch.Tensor):
        tensor = args[0]
        if tensor.ndim == 0:
            return inputs
        item = tensor[index : index + 1]
        return (item, *args[1:]) if len(args) > 1 else item
    if isinstance(inputs, torch.Tensor):
        return inputs[index : index + 1]
    raise TypeError("reference_batch must be a tensor or tuple whose first element is a tensor")


def _batch_len(inputs: Any) -> int:
    args, _kwargs = normalise_model_inputs(inputs)
    if args and isinstance(args[0], torch.Tensor):
        return int(args[0].shape[0])
    if isinstance(inputs, torch.Tensor):
        return int(inputs.shape[0])
    return 0


def _run_pathway_ablation(
    engine: InterventionEngine,
    source: UnitGroup,
    dest: UnitGroup,
    *,
    baseline: str,
    target: int | None,
) -> dict[str, Any]:
    original = engine.evaluate()
    original_score = float(original.trace.target_score)
    original_pred = original.trace.predicted_index
    original_margin = _decision_margin(original.logits, target)
    both = engine.intervene([source, dest], baseline=baseline)
    both_score = float(both.trace.target_score)
    both_margin = _decision_margin(both.logits, target)
    margin_delta = (
        None
        if original_margin is None or both_margin is None
        else float(both_margin - original_margin)
    )
    return {
        "focus_score_delta": both_score - original_score,
        "focus_margin_delta": margin_delta,
        "original_margin": original_margin,
        "intervened_margin": both_margin,
        "focus_prediction_flipped": (
            original_pred is not None and both.trace.predicted_index != original_pred
        ),
        "original_score": original_score,
        "intervened_score": both_score,
        "original_prediction": original_pred,
        "intervened_prediction": both.trace.predicted_index,
    }


def enrich_circuit_wiring_edges(
    model: nn.Module,
    inputs: Any,
    wiring: AnalysisResult,
    *,
    target: int | None = 1,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    reference_batch: Any | None = None,
    top_k_intervene: int = 24,
    bootstrap_samples: int = 100,
    baseline: str = "zero",
    concept_associations: Mapping[str, Any] | None = None,
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Attach structured CONNECTION cards to every wiring edge."""

    if wiring.analysis != "circuit_wiring":
        raise ValueError("enrich_circuit_wiring_edges requires a circuit_wiring result")
    nodes = list(wiring.tables.get("nodes") or [])
    edges = list(wiring.tables.get("edges") or [])
    if not edges:
        metrics = dict(wiring.metrics)
        metrics["edge_cards"] = 0
        metrics["intervened_edges"] = 0
        return AnalysisResult(
            analysis=wiring.analysis,
            metrics=metrics,
            tables={**wiring.tables, "edge_cards": []},
            metadata={
                **dict(wiring.metadata),
                "edge_cards": {"enriched": True, "intervened_edges": 0},
            },
            warnings=list(wiring.warnings),
            limitations=list(wiring.limitations),
            schema_version=wiring.schema_version,
        )

    names: dict[int, str] = {}
    if isinstance(class_names, Mapping):
        names = {int(k): str(v) for k, v in class_names.items()}
    elif isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        names = {i: str(v) for i, v in enumerate(class_names)}
    target_label = _class_label(names, target)

    node_by_id = {str(node["id"]): node for node in nodes}
    modules = {name: module for name, module in model.named_modules() if name}
    contrib = _contribution_map(model, inputs, target)

    hop_flow_totals: dict[tuple[str, str], float] = defaultdict(float)
    hop_struct_totals: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        hop = (str(edge["source_layer"]), str(edge["target_layer"]))
        flow = float(edge.get("activation_transfer", edge.get("mass", 0.0)))
        struct = float(edge.get("structural_mass", flow))
        hop_flow_totals[hop] += flow
        hop_struct_totals[hop] += struct

    ranked = sorted(
        edges,
        key=lambda row: float(row.get("activation_transfer", row.get("mass", 0.0))),
        reverse=True,
    )
    intervene_keys = {
        _edge_key(str(edge["source"]), str(edge["target"]))
        for edge in ranked[: max(0, int(top_k_intervene))]
    }

    focus_engine = InterventionEngine(
        model, inputs, target=target, config=config or VRavenConfig(mode="fast")
    )
    focus_eval = focus_engine.evaluate()
    focus_margin = _decision_margin(focus_eval.logits, target)

    unit_cache: dict[str, UnitGroup] = {}
    for node_id, node in node_by_id.items():
        try:
            unit_cache[node_id] = _unit_group_from_node(node)
        except Exception:  # noqa: BLE001
            continue

    cards: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    intervened = 0
    level_counts: dict[str, int] = defaultdict(int)

    for edge in edges:
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        key = _edge_key(source_id, target_id)
        src_node = node_by_id.get(source_id, {})
        dst_node = node_by_id.get(target_id, {})
        src_indices = list(src_node.get("indices") or [])
        dst_indices = list(dst_node.get("indices") or [])
        src_layer = str(edge["source_layer"])
        dst_layer = str(edge["target_layer"])

        activation_transfer = float(edge.get("activation_transfer", edge.get("mass", 0.0)))
        structural_mass = float(edge.get("structural_mass", activation_transfer))
        signed_transfer = edge.get("signed_transfer")
        signed_transfer_f = float(signed_transfer) if signed_transfer is not None else None

        hop_flow = hop_flow_totals[(src_layer, dst_layer)] or 1.0
        hop_struct = hop_struct_totals[(src_layer, dst_layer)] or 1.0
        relative_flow = activation_transfer / hop_flow
        relative_mass = structural_mass / hop_struct

        src_sens = _group_sensitivity(contrib.get(src_layer), src_indices)
        dst_sens = _group_sensitivity(contrib.get(dst_layer), dst_indices)
        pathway_sens = src_sens if src_sens is not None else dst_sens

        unit_noun = "neurons"
        src_label = readable_unit_group_label(
            source_id, indices=src_indices, compact=False
        ).replace("units ", f"{unit_noun} ").replace("unit ", f"{unit_noun[:-1]} ")
        dst_label = readable_unit_group_label(
            target_id, indices=dst_indices, compact=False
        ).replace("units ", f"{unit_noun} ").replace("unit ", f"{unit_noun[:-1]} ")

        role = _operator_role(dst_layer, modules.get(dst_layer))

        semantic: dict[str, Any] = {"available": False, "status": "unavailable"}
        if concept_associations:
            assoc = concept_associations.get(key) or concept_associations.get(source_id)
            if assoc:
                semantic = {
                    "available": True,
                    "status": str(assoc.get("semantic_status", "association_only")),
                    "source_concept": assoc.get("source_concept"),
                    "destination_concept": assoc.get("destination_concept"),
                    "claim": "association evidence only — not feature identity",
                }

        intervention: dict[str, Any] | None = None
        if key in intervene_keys and source_id in unit_cache and target_id in unit_cache:
            intervened += 1
            src_g = unit_cache[source_id]
            dst_g = unit_cache[target_id]
            focus_stats = _run_pathway_ablation(
                focus_engine, src_g, dst_g, baseline=baseline, target=target
            )
            deltas: list[float] = [float(focus_stats["focus_score_delta"])]
            margin_deltas: list[float] = []
            if focus_stats["focus_margin_delta"] is not None:
                margin_deltas.append(float(focus_stats["focus_margin_delta"]))
            flips = [bool(focus_stats["focus_prediction_flipped"])]

            if reference_batch is not None:
                n_ref = _batch_len(reference_batch)
                if n_ref > 0:
                    rng = np.random.default_rng(24032026)
                    count = min(int(bootstrap_samples), n_ref)
                    picks = rng.choice(n_ref, size=count, replace=False)
                    for idx in picks:
                        sample = _slice_batch(reference_batch, int(idx))
                        eng = InterventionEngine(
                            model,
                            sample,
                            target=target,
                            config=config or VRavenConfig(mode="fast"),
                        )
                        try:
                            stats = _run_pathway_ablation(
                                eng, src_g, dst_g, baseline=baseline, target=target
                            )
                        except Exception:  # noqa: BLE001
                            continue
                        deltas.append(float(stats["focus_score_delta"]))
                        if stats["focus_margin_delta"] is not None:
                            margin_deltas.append(float(stats["focus_margin_delta"]))
                        flips.append(bool(stats["focus_prediction_flipped"]))

            lo, hi = _percentile_ci(deltas) if len(deltas) >= 2 else (None, None)
            m_lo, m_hi = _percentile_ci(margin_deltas) if len(margin_deltas) >= 2 else (None, None)
            flip_count = int(sum(flips))
            flip_n = int(len(flips))
            intervention = {
                "computed": True,
                "baseline": baseline,
                "method": "joint_endpoint_ablation",
                "claim": (
                    "Ablates source and destination unit groups (activation intervention), "
                    "not the exact Linear weight sub-block."
                ),
                "focus_score_delta": float(focus_stats["focus_score_delta"]),
                "focus_margin_delta": focus_stats["focus_margin_delta"],
                "original_margin": focus_stats["original_margin"],
                "intervened_margin": focus_stats["intervened_margin"],
                "focus_prediction_flipped": bool(focus_stats["focus_prediction_flipped"]),
                "mean_score_delta": float(np.mean(deltas)),
                "mean_margin_delta": float(np.mean(margin_deltas)) if margin_deltas else None,
                "score_delta_ci95": [lo, hi] if lo is not None else None,
                "margin_delta_ci95": [m_lo, m_hi] if m_lo is not None else None,
                "flips": flip_count,
                "flip_trials": flip_n,
                "flip_rate": float(flip_count / flip_n) if flip_n else None,
            }

        intervention_outcome = _intervention_outcome(intervention)
        direction = _contribution_direction(
            target_label=target_label,
            intervention=intervention,
            pathway_sens=pathway_sens,
        )
        ladder = _evidence_ladder(
            structural_mass=structural_mass,
            activation_transfer=activation_transfer,
            pathway_sens=pathway_sens,
            intervention=intervention,
            status_code=str(intervention_outcome["code"]),
            semantic_available=bool(semantic.get("available")),
            hop_flow_total=hop_flow,
        )
        status = _status_from_evidence(ladder, intervention_outcome)
        level_counts[str(ladder["level"])] += 1

        card = {
            "key": key,
            "title": "CONNECTION",
            "header": f"{src_label} → {dst_label}",
            "source_id": source_id,
            "target_id": target_id,
            "source_label": src_label,
            "target_label": dst_label,
            "structure": {
                "connection_mass": structural_mass,
                "structural_mass": structural_mass,
                "mean_abs_weight": float(edge["mean_abs_weight"]),
                "mass_normalised": float(
                    edge.get("structural_mass_normalised", edge.get("mass_normalised", 0.0))
                ),
                "relative_mass": float(relative_mass),
                "source_layer": src_layer,
                "target_layer": dst_layer,
                "source_layer_short": shorten_layer_name(src_layer, max_length=22),
                "target_layer_short": shorten_layer_name(dst_layer, max_length=22),
                "retained_path": bool(edge.get("retained_path")),
                "source_indices": src_indices,
                "target_indices": dst_indices,
                "formula": "M = Σ|W_ji|",
            },
            "live_flow": {
                "activation_transfer": activation_transfer,
                "signed_transfer": signed_transfer_f,
                "contribution_direction": direction["label"],
                "direction": direction,
                "relative_flow": float(relative_flow),
                "relative_flow_percent": float(100.0 * relative_flow),
                "focus_margin": focus_margin,
                "formula": "F = Σ|a_i W_ji|",
            },
            "sensitivity": {
                "available": pathway_sens is not None,
                "target_score_sensitivity": pathway_sens,
                "source_group_sensitivity": src_sens,
                "destination_group_sensitivity": dst_sens,
                "claim": "first-order unit contributions (a⊙∂s/∂a), not causal pathway strength",
            },
            "causal_proof": intervention
            or {
                "computed": False,
                "detail": "Intervention reserved for top-mass edges only.",
            },
            "mechanistic_role": role,
            "semantic": semantic,
            "status": status,
            "evidence_ladder": ladder,
            "limitations": [
                "Structural mass M=Σ|W| and activation transfer F=Σ|a W| are distinct.",
                "Causal proof ablates endpoint unit groups, not a unique causal pathway.",
                "Semantic concept names appear only when Reverse association evidence is attached.",
            ],
        }
        cards.append(card)
        new_edge = dict(edge)
        new_edge["card"] = card
        new_edge["evidence_level"] = ladder["level"]
        new_edges.append(new_edge)

    metrics = dict(wiring.metrics)
    metrics["edge_cards"] = len(cards)
    metrics["intervened_edges"] = intervened
    metrics["evidence_levels"] = dict(level_counts)
    metadata = dict(wiring.metadata)
    metadata["edge_cards"] = {
        "enriched": True,
        "intervened_edges": intervened,
        "top_k_intervene": int(top_k_intervene),
        "bootstrap_samples": int(bootstrap_samples),
        "baseline": baseline,
        "target": target,
        "target_label": target_label,
        "focus_margin": focus_margin,
        "evidence_ladder": [
            "structural",
            "active",
            "sensitive",
            "intervention_tested",
            "causal_support",
            "semantically_grounded",
        ],
        "decision_margin_definition": "m = z_target − max_{j≠target} z_j",
    }
    limitations = list(wiring.limitations)
    for note in (
        "CONNECTION cards separate structural mass Σ|W| from sample flow Σ|a W|.",
        "Supports/Opposes uses ablation Δm when available; else first-order target-score proxy.",
        "INTERVENTION SUPPORTED means joint endpoint ablation often flips the decision — not unique causality.",
    ):
        if note not in limitations:
            limitations.append(note)

    return AnalysisResult(
        analysis=wiring.analysis,
        metrics=metrics,
        tables={
            **wiring.tables,
            "nodes": nodes,
            "edges": new_edges,
            "edge_cards": cards,
        },
        metadata=metadata,
        warnings=list(wiring.warnings),
        limitations=limitations,
        schema_version=wiring.schema_version,
    )
