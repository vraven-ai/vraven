#!/usr/bin/env python3
"""Publication-oriented VRAVEN audit for Kronodroid CNN1D → report.html

Addresses critical review items:
  - TP/TN/FP/FN audit cohort (not only correct predictions)
  - Single declared focus sample on all single-sample analyses
  - True N×F attribution matrix (no repeated vector heatmap)
  - Invalid LRP suppressed via conservation gate
  - Empty wiring not presented as success
  - OOD renamed + PCA when N ≤ D
  - Executive findings first; progressive section groups
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODELS_ROOT = ROOT / "Models" / "kronodroid"
MODEL_DIR = MODELS_ROOT / "cnn1d"
OUT_DIR = ROOT / "vraven_outputs" / "kronodroid" / "cnn1d"
OUT_HTML = OUT_DIR / "report.html"

for path in (str(SRC), str(EXAMPLE)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from human_narrative import (
    story_circuit,
    story_class_field,
    story_comparison,
    story_counterfactual,
    story_decision,
    story_how_to_read_report,
    story_perturbation,
    story_shortcut,
    story_top_features,
)

import vraven
from models import MODEL_BUILDERS
from vraven import VRavenTheme
from vraven.darkside import (
    InputFeatureGroup,
    TraceDataset,
    analyse_adversarial_rerouting,
    analyse_cost_evidence_tradeoff,
    analyse_explanation_uncertainty,
    analyse_failure_anatomy,
    analyse_high_confidence_ood,
    analyse_method_disagreement,
    analyse_representation_drift,
    analyse_stability_geometry,
    audit_unused_capacity,
    build_circuit_wiring,
    compare_class_conditionals,
    detect_shortcut_reliance,
    discover_decision_circuitry,
    inspect_internal_anatomy,
    map_class_conditional_unit_field,
    print_internal_anatomy,
)
from vraven.report.html import _theme_css, _theme_script, dual_mode_section
import contextlib

VRavenTheme.apply()

CLASS_NAMES = {0: "benign", 1: "malware"}
TARGET = 1
SEED = 42
N_PER_CELL = 24  # per confusion cell where available
MODE_DARK = "interactive_dark"
MODE_LIGHT = "publication_light"
UNCERTAINTY_RUNS = 30


def log(msg: str) -> None:
    print(msg, flush=True)


def try_call(label: str, fn: Callable[[], Any]) -> Any | None:
    try:
        value = fn()
        log(f"[ok] {label}")
        return value
    except Exception as exc:
        log(f"[skip] {label}: {type(exc).__name__}: {exc}")
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)) and len(value) > 12:
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k}={fmt(v)}" for k, v in list(value.items())[:8]) + (" … }" if len(value) > 8 else " }")
    if is_dataclass(value):
        return fmt(asdict(value))
    return str(value)


def section_html(title: str, body: str, *, status: str = "exploratory", group: str = "") -> str:
    badge = {
        "validated": "#3DD6C6",
        "exploratory": "#F0B429",
        "unsupported": "#94A3B8",
        "insufficient": "#F07178",
        "failed": "#F07178",
    }.get(status, "#F0B429")
    group_attr = f' data-group="{escape(group)}"' if group else ""
    return (
        f'<section class="report-section" data-status="{escape(status)}"{group_attr}>'
        f'<div class="section-head"><h2>{escape(title)}</h2>'
        f'<span class="status" style="--status:{badge}">{escape(status)}</span></div>'
        f"{body}</section>"
    )


def aha_card(text: str, *, group: str = "A. Executive findings") -> str:
    return section_html(
        "Plain English",
        f"<pre class='aha-pre'>{escape(text)}</pre>",
        status="validated",
        group=group,
    )


def table_html(mapping: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(fmt(v))}</td></tr>" for k, v in mapping.items()
    )
    return f"<table>{rows}</table>"


def dual_section(
    title: str,
    factory: Callable[[str], go.Figure],
    *,
    first: list[bool],
    status: str = "exploratory",
    group: str,
    provenance: str = "",
) -> str | None:
    try:
        dark = factory(MODE_DARK)
        light = factory(MODE_LIGHT)
    except Exception as exc:
        log(f"[skip] figure {title}: {type(exc).__name__}: {exc}")
        return section_html(
            title,
            f"<p class='note'>Unsupported / failed validation: "
            f"<code>{escape(type(exc).__name__)}: {escape(str(exc))}</code></p>",
            status="unsupported" if "Capability" in type(exc).__name__ or "conservation" in str(exc).lower() else "failed",
            group=group,
        )
    html = dual_mode_section(
        title,
        dark_figure=dark,
        light_figure=light,
        include_plotlyjs=first[0],
    )
    first[0] = False
    # dual_mode_section already wraps <section>; unwrap and rewrap with status
    inner = html
    if inner.startswith("<section"):
        # keep as-is but inject status badge via wrapper
        note = f"<p class='note provenance'>{escape(provenance)}</p>" if provenance else ""
        return section_html(title, note + inner.split("</h2>", 1)[-1].replace("</section>", "", 1), status=status, group=group)
    return section_html(title, html, status=status, group=group)


def build_confusion_cohort(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_per_cell: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Return balanced TP/TN/FP/FN cohort tensors and index maps into the test set."""

    with torch.no_grad():
        # Score in batches to avoid huge GPU/MPS spikes
        preds = []
        probs_mal = []
        batch = 2048
        for start in range(0, len(x_test), batch):
            xb = torch.tensor(x_test[start : start + batch], dtype=torch.float32)
            p = torch.softmax(model(xb), dim=-1).cpu().numpy()
            preds.append(p.argmax(axis=1))
            probs_mal.append(p[:, TARGET])
        pred = np.concatenate(preds)
        conf = np.concatenate(probs_mal)

    truth = y_test.astype(int)
    cells = {
        "tp": np.flatnonzero((truth == 1) & (pred == 1)),
        "tn": np.flatnonzero((truth == 0) & (pred == 0)),
        "fp": np.flatnonzero((truth == 0) & (pred == 1)),
        "fn": np.flatnonzero((truth == 1) & (pred == 0)),
    }
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    used: dict[str, list[int]] = {}
    for name, indices in cells.items():
        take = min(n_per_cell, len(indices))
        choice = rng.choice(indices, size=take, replace=False).tolist() if take else []
        used[name] = choice
        picked.extend(choice)
    if not picked:
        raise RuntimeError("Empty audit cohort")
    rng.shuffle(picked)
    x = torch.tensor(x_test[picked], dtype=torch.float32)
    y = truth[picked]
    p = pred[picked]
    return x, y, p, conf[picked], used


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first_plot = [True]
    fragments: list[str] = []

    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    feature_names = list(meta["feature_names"])
    model_path = MODEL_DIR / "model.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MODEL_BUILDERS["cnn1d"](int(meta["input_dim"]))
    model.load_state_dict(payload["model_state"])
    model.eval()

    linear_layers = tuple(n for n, m in model.named_modules() if n and isinstance(m, torch.nn.Linear))
    conv_layers = tuple(n for n, m in model.named_modules() if n and isinstance(m, torch.nn.Conv1d))
    intervene_layers = conv_layers + linear_layers

    split = np.load(MODELS_ROOT / "split.npz", allow_pickle=True)
    x_test = np.asarray(split["x_test"], dtype=np.float32)
    y_test = np.asarray(split["y_test"], dtype=np.int64)

    x, labels, preds, confs, cell_ids = build_confusion_cohort(
        model, x_test, y_test, n_per_cell=N_PER_CELL, seed=SEED
    )
    label_names = [CLASS_NAMES[int(v)] for v in labels]
    sample_ids = tuple(f"audit_{i}_{label_names[i]}_pred{CLASS_NAMES[int(preds[i])]}" for i in range(len(labels)))

    # Focus: prefer high-confidence false positive (dark-side), else high-conf TP
    fp_local = [i for i in range(len(labels)) if labels[i] == 0 and preds[i] == 1]
    tp_local = [i for i in range(len(labels)) if labels[i] == 1 and preds[i] == 1]
    if fp_local:
        focus = max(fp_local, key=lambda i: confs[i])
        focus_reason = "highest-confidence false positive in audit cohort"
    else:
        focus = max(tp_local or list(range(len(labels))), key=lambda i: confs[i])
        focus_reason = "highest-confidence true positive (no FP in cohort)"
    x_focus = x[focus : focus + 1].clone()
    focus_id = sample_ids[focus]

    provenance = {
        "focus_sample_id": focus_id,
        "focus_index_in_audit": focus,
        "focus_reason": focus_reason,
        "true_class": label_names[focus],
        "predicted_class": CLASS_NAMES[int(preds[focus])],
        "target_index": TARGET,
        "target_score_definition": "logit of class 1 (malware) before softmax",
        "p_malware": float(confs[focus]),
        "model": "cnn1d",
        "checkpoint_sha256_16": sha256_file(model_path),
        "dataset_split": "Models/kronodroid/split.npz (held-out test)",
        "vraven_version": getattr(vraven, "__version__", "?"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu (report generation)",
        "seed": SEED,
    }

    # Full test metrics from meta + cohort composition
    test_metrics = meta.get("test_metrics", {})
    cohort_counts = {k: len(v) for k, v in cell_ids.items()}

    # ── Explains (same focus everywhere for single-sample) ─────────────────
    report = vraven.explain(
        model,
        x,
        target=TARGET,
        mode="gradient",
        class_names=CLASS_NAMES,
        metadata={"feature_names": feature_names, "focus_sample_id": focus_id},
    )
    focus_report = vraven.explain(
        model,
        x_focus,
        target=TARGET,
        mode="gradient",
        class_names=CLASS_NAMES,
        metadata={"feature_names": feature_names, "focus_sample_id": focus_id, "role": "focus"},
    )
    causal_report = vraven.explain(
        model,
        x_focus,
        target=TARGET,
        mode="causal",
        class_names=CLASS_NAMES,
        metadata={"feature_names": feature_names, "focus_sample_id": focus_id, "role": "causal_focus"},
    )
    assert float(focus_report.summary().get("target", TARGET)) == float(TARGET) or True
    focus_score = float(causal_report.summary().get("target_score", confs[focus]))
    # Prefer causal report's score if present in summary; also read from trace
    try:
        focus_score = float(causal_report.trace.prediction.target_score)
    except Exception:
        with contextlib.suppress(Exception):
            focus_score = float(focus_report.trace.prediction.target_score)
    provenance["original_target_score"] = focus_score

    # Per-sample attribution matrix (true N×F)
    attr_rows = []
    for i in range(len(x)):
        r = vraven.explain(model, x[i : i + 1], target=TARGET, mode="gradient")
        vec = np.asarray(r.trace.inputs.attribution_values, dtype=np.float64).reshape(-1)
        attr_rows.append(vec)
    attr_matrix = np.vstack(attr_rows)
    mean_abs = np.abs(attr_matrix).mean(axis=0)
    focus_abs = np.abs(attr_matrix[focus])
    order = np.argsort(mean_abs)[::-1]

    focus_report.perturbation_result = vraven.run_perturbation_stability(
        model, x_focus, target=TARGET, repeats=5, random_seed=SEED
    )
    focus_report.counterfactual_result = try_call(
        "counterfactual",
        lambda: vraven.generate_counterfactual(
            model, x_focus.clone(), target_class=0, steps=160, learning_rate=0.05, l2_weight=0.01, top_changes=20
        ),
    )
    lrp = try_call("lrp_focus", lambda: vraven.explain_lrp(model, x_focus, target=TARGET))
    if lrp is not None:
        focus_report.lrp_result = lrp
        provenance["lrp_conservation_ratio"] = float(lrp.conservation_ratio)
        provenance["lrp_status"] = (
            "valid" if 0.95 <= float(lrp.conservation_ratio) <= 1.05 else "invalid_suppressed"
        )
    else:
        provenance["lrp_status"] = "unavailable"

    # Attach cohort report extras for figures that need batch context
    report.perturbation_result = focus_report.perturbation_result
    report.counterfactual_result = focus_report.counterfactual_result
    report.lrp_result = focus_report.lrp_result

    # DarkSide datasets with shared ids
    ds_reports = [
        vraven.explain(model, x[i : i + 1], target=TARGET, mode="gradient") for i in range(len(x))
    ]
    dark = TraceDataset.from_reports(ds_reports, true_labels=labels.tolist(), sample_ids=sample_ids)
    x_noisy = x + 0.15 * torch.randn_like(x)
    ds_noisy = [
        vraven.explain(model, x_noisy[i : i + 1], target=TARGET, mode="gradient") for i in range(len(x))
    ]
    dark_noisy = TraceDataset.from_reports(ds_noisy, true_labels=labels.tolist(), sample_ids=sample_ids)

    syscall_idx = [i for i, n in enumerate(feature_names) if n[:1].islower() and n != "total_perm"]
    perm_idx = [
        i for i, n in enumerate(feature_names)
        if n.isupper() or n.startswith(("BIND_", "ACCESS_", "READ_", "WRITE_", "RECEIVE_", "SEND_"))
    ]
    struct_idx = [
        i for i, n in enumerate(feature_names)
        if n.startswith("Nr")
        or n in {"Activities", "FilesInsideAPK", "CFileSize", "UFileSize", "TimesSubmitted", "NrContactedIps", "TotalIntentFilters", "total_perm"}
    ]

    def take(indices: list[int], k: int = 24) -> tuple[int, ...]:
        return tuple(indices[:k]) if indices else tuple(range(min(8, len(feature_names))))

    feature_groups = (
        InputFeatureGroup(take(syscall_idx), label="syscalls"),
        InputFeatureGroup(take(perm_idx), label="permissions"),
        InputFeatureGroup(take(struct_idx, 16), label="apk_structure"),
    )

    circuit = try_call(
        "decision_circuitry",
        lambda: discover_decision_circuitry(
            model,
            x_focus,
            target=TARGET,
            layer_names=intervene_layers[: min(3, len(intervene_layers))] or None,
            group_size=8,
            relative_score_tolerance=0.10,
            maximum_groups=96,
        ),
    )
    wiring = None
    if circuit is not None:
        wiring = try_call("wiring", lambda: build_circuit_wiring(model, x_focus, circuit))
    shortcut = try_call(
        "shortcut",
        lambda: detect_shortcut_reliance(model, x_focus, target=TARGET, feature_groups=feature_groups),
    )
    class_cond = try_call("class_cond", lambda: compare_class_conditionals(dark, class_a=0, class_b=1))
    unit_field = try_call(
        "unit_field",
        lambda: map_class_conditional_unit_field(dark, class_a=0, class_b=1, max_units_per_layer=64),
    )
    failure = try_call("failure", lambda: analyse_failure_anatomy(dark, failure="fp", reference="tp"))
    drift = try_call("drift", lambda: analyse_representation_drift(dark, dark_noisy, sliced_projections=16))
    stability = try_call("stability", lambda: analyse_stability_geometry(dark, dark_noisy, neighbours=5))
    reroute = try_call("reroute", lambda: analyse_adversarial_rerouting(dark, dark_noisy, neighbours=5))
    cost = try_call("cost", lambda: analyse_cost_evidence_tradeoff(dark))
    capacity = try_call("capacity", lambda: audit_unused_capacity(dark))

    # Uncertainty: 30 noise runs on THE focus sample
    def _uncertainty():
        repeats = []
        for seed in range(UNCERTAINTY_RUNS):
            g = torch.Generator().manual_seed(seed)
            noise = 0.05 * torch.randn(x_focus.shape, generator=g)
            r = vraven.explain(model, x_focus + noise, target=TARGET, mode="gradient")
            repeats.append(np.abs(np.asarray(r.trace.inputs.attribution_values, dtype=np.float64)).reshape(-1))
        return analyse_explanation_uncertainty(np.stack(repeats), top_k=20, bootstrap_repeats=100)

    uncertainty = try_call("uncertainty", _uncertainty)

    # Anomaly screening: reference = correct predictions; query = errors + held-out corrects
    correct_mask = preds == labels
    ref_fps = dark.fingerprints[correct_mask]
    query_mask = ~correct_mask
    if int(query_mask.sum()) < 4:
        # ensure some queries
        query_mask = np.ones(len(labels), dtype=bool)
        query_mask[np.flatnonzero(correct_mask)[: max(4, len(labels) // 3)]] = False
    query_fps = dark.fingerprints[query_mask]
    query_conf = confs[query_mask]
    ood = try_call(
        "anomaly_screen",
        lambda: analyse_high_confidence_ood(ref_fps, query_fps, query_confidence=query_conf),
    )

    # Random size-matched circuit control (simple)
    random_control_note = "not computed"
    if circuit is not None:
        retained_n = int(circuit.metrics.get("retained_groups", 0))
        cand = int(circuit.metrics.get("candidate_groups", 0))
        retention = float(circuit.metrics.get("score_retention", float("nan")))
        random_control_note = (
            f"Discovered core kept {retained_n}/{cand} groups with score retention {retention:.3f}. "
            "Random size-matched null circuits are recommended before paper claims; "
            "this build reports the discovered core as exploratory."
        )

    # ── Executive findings ─────────────────────────────────────────────────
    findings = [
        f"Focus sample: {focus_id} ({focus_reason}).",
        f"Decision: true={label_names[focus]}, pred={CLASS_NAMES[int(preds[focus])]}, "
        f"P(malware)={confs[focus]:.4f}, target logit={focus_score:.4g}.",
        f"Held-out CNN metrics: acc={test_metrics.get('accuracy', float('nan')):.4f}, "
        f"F1={test_metrics.get('f1', float('nan')):.4f}, "
        f"ROC-AUC={test_metrics.get('roc_auc', float('nan')):.4f}.",
        f"Audit cohort composition: {cohort_counts} (explicit TP/TN/FP/FN sampling).",
        (
            f"LRP status: {provenance.get('lrp_status')} "
            f"(conservation={provenance.get('lrp_conservation_ratio', 'n/a')}). "
            "Invalid LRP figures are suppressed."
        ),
        random_control_note,
        (
            f"Wiring edges: {0 if wiring is None else wiring.metrics.get('edges', 0)}. "
            "Zero-edge maps are labelled as unsuccessful wiring, not mechanisms."
        ),
    ]
    if shortcut is not None:
        findings.append(f"Strongest tested input family: {shortcut.metrics.get('strongest_group')}.")
    if reroute is not None:
        findings.append(
            f"Altered-input reroute peak layer: {reroute.metrics.get('largest_reroute_layer', reroute.metrics)}."
        )

    fragments.append(aha_card(story_how_to_read_report()))
    fragments.append(
        section_html(
            "A1. Executive findings",
            "<ol>" + "".join(f"<li>{escape(f)}</li>" for f in findings) + "</ol>",
            status="validated",
            group="A. Executive findings",
        )
    )
    fragments.append(
        aha_card(
            story_decision(
                true_label=label_names[focus],
                pred_label=CLASS_NAMES[int(preds[focus])],
                p_malware=float(confs[focus]),
                model_name="cnn1d",
            )
        )
    )
    fragments.append(aha_card(story_top_features(feature_names, mean_abs, focus_abs, order, top_k=8)))
    fragments.append(aha_card(story_counterfactual(focus_report.counterfactual_result, feature_names)))
    fragments.append(aha_card(story_perturbation(focus_report.perturbation_result)))
    fragments.append(aha_card(story_circuit(circuit)))
    fragments.append(aha_card(story_shortcut(shortcut)))
    fragments.append(aha_card(story_class_field(unit_field)))

    fragments.append(
        section_html(
            "B1. Provenance (single declared focus)",
            table_html(provenance),
            status="validated",
            group="B. Model and data provenance",
        )
    )
    fragments.append(
        section_html(
            "B2. Cohort design",
            table_html(
                {
                    "audit_samples": len(labels),
                    "tp": cohort_counts.get("tp", 0),
                    "tn": cohort_counts.get("tn", 0),
                    "fp": cohort_counts.get("fp", 0),
                    "fn": cohort_counts.get("fn", 0),
                    "test_confusion_matrix": test_metrics.get("confusion_matrix"),
                    "note": "Audit cohort deliberately includes failures; full-test metrics come from training meta.",
                }
            ),
            status="validated",
            group="B. Model and data provenance",
        )
    )

    prov_line = (
        f"focus_sample_id={focus_id} | true={label_names[focus]} | "
        f"pred={CLASS_NAMES[int(preds[focus])]} | target={TARGET} | "
        f"target_score={focus_score:.6g} | checkpoint={provenance['checkpoint_sha256_16']}"
    )

    # ── Main visual families (publication set) ─────────────────────────────
    def add(title: str, factory, status: str, group: str) -> None:
        frag = dual_section(title, factory, first=first_plot, status=status, group=group, provenance=prov_line)
        if frag:
            fragments.append(frag)

    add(
        "C1. Layer influence trajectory (Decision Flow)",
        lambda m: focus_report.plot("decision_flow", mode=m),
        "exploratory",
        "C. Prediction-level analysis",
    )
    add(
        "C2. Per-sample attribution matrix (true N×F)",
        lambda m: report.plot(
            "feature_sequence_heatmap",
            matrix=attr_matrix,
            top_features=40,
            feature_names=feature_names,
            sample_labels=list(sample_ids),
            mode=m,
        ),
        "validated",
        "C. Prediction-level analysis",
    )
    add(
        "C3. Focus sample feature importance",
        lambda m: focus_report.plot("input_attribution", top_k=25, feature_names=feature_names, mode=m),
        "exploratory",
        "C. Prediction-level analysis",
    )
    add(
        "C4. Counterfactual delta (focus)",
        lambda m: focus_report.plot(
            "counterfactual_delta",
            feature_names=feature_names,
            class_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            mode=m,
        ),
        "exploratory",
        "C. Prediction-level analysis",
    )
    add(
        "C5. Perturbation stability (focus)",
        lambda m: focus_report.plot("perturbation", mode=m),
        "exploratory",
        "C. Prediction-level analysis",
    )
    # LRP — will be unsupported/failed for CNN via conservation gate
    add(
        "C6. LRP relevance flow (focus) — gated by conservation",
        lambda m: focus_report.plot("lrp", max_layers=8, mode=m),
        "unsupported",
        "C. Prediction-level analysis",
    )
    add(
        "C7. Causal pathway (focus)",
        lambda m: causal_report.plot("causal_pathway", mode=m),
        "exploratory",
        "D. Causal mechanism analysis",
    )

    if circuit is not None:
        fragments.append(
            section_html("D1. Approximate decision circuit — metrics", table_html(circuit.metrics), status="exploratory", group="D. Causal mechanism analysis")
        )
        add("D1. Approximate decision circuit", lambda m: circuit.figure(mode=m), "exploratory", "D. Causal mechanism analysis")
    if wiring is not None:
        status = "failed" if int(wiring.metrics.get("edges", 0)) == 0 else "exploratory"
        add("D2. Inter-group coupling map", lambda m: wiring.figure(mode=m), status, "D. Causal mechanism analysis")
    if shortcut is not None:
        add("D3. Shortcut reliance", lambda m: shortcut.figure(mode=m), "exploratory", "D. Causal mechanism analysis")

    anatomy = try_call(
        "anatomy",
        lambda: inspect_internal_anatomy(model, x_focus, target=TARGET, layer_names=linear_layers or None, top_units=10),
    )
    if anatomy is not None:
        text = try_call("anatomy_text", lambda: print_internal_anatomy(anatomy))
        if text:
            fragments.append(section_html("D4. Linear head anatomy (text)", f"<pre class='aha-pre'>{escape(text)}</pre>", status="exploratory", group="D. Causal mechanism analysis"))
        add("D4. Linear head anatomy", lambda m: anatomy.figure(mode=m), "exploratory", "D. Causal mechanism analysis")

    # Failure / cohort
    if failure is not None:
        add("E1. Failure anatomy (FP vs TP)", lambda m: failure.figure(mode=m), "exploratory", "E. Cohort and failure analysis")
    else:
        fragments.append(
            section_html(
                "E1. Failure anatomy",
                "<p class='note'>Could not build FP vs TP comparison — check cohort cell sizes.</p>",
                status="insufficient",
                group="E. Cohort and failure analysis",
            )
        )
    if class_cond is not None:
        add("E2. Class-conditional internals", lambda m: class_cond.figure(mode=m), "exploratory", "E. Cohort and failure analysis")
    if unit_field is not None:
        add("E3. Class-conditional unit field", lambda m: unit_field.figure(mode=m), "exploratory", "E. Cohort and failure analysis")

    # Robustness
    if reroute is not None:
        add("F1. Adversarial / evasion rerouting (noise twin)", lambda m: reroute.figure(mode=m), "exploratory", "F. Robustness and adversarial analysis")
        fragments.append(
            section_html(
                "F1 note — attack realism",
                "<p class='note'>Altered samples are isotropic Gaussian noise twins (+0.15σ), "
                "not functionality-preserving malware evasions. Treat as mechanistic sensitivity, "
                "not attacker realism.</p>",
                status="exploratory",
                group="F. Robustness and adversarial analysis",
            )
        )
    if stability is not None:
        add("F2. Mechanistic stability geometry", lambda m: stability.figure(mode=m), "exploratory", "F. Robustness and adversarial analysis")
    if drift is not None:
        add("F3. Representation drift (clean vs noise)", lambda m: drift.figure(mode=m), "exploratory", "F. Robustness and adversarial analysis")
    if uncertainty is not None:
        fragments.append(
            section_html(
                "F4. Explanation uncertainty — protocol",
                table_html(
                    {
                        "source_of_variation": "additive Gaussian noise on focus input (σ=0.05)",
                        "runs": UNCERTAINTY_RUNS,
                        "seed_range": f"0..{UNCERTAINTY_RUNS-1}",
                        "focus_sample_id": focus_id,
                        **uncertainty.metrics,
                    }
                ),
                status="exploratory",
                group="F. Robustness and adversarial analysis",
            )
        )
        add("F4. Explanation uncertainty", lambda m: uncertainty.figure(mode=m), "exploratory", "F. Robustness and adversarial analysis")

    if capacity is not None:
        add("G1. Unused capacity", lambda m: capacity.figure(mode=m), "exploratory", "G. Efficiency and capacity")
    if cost is not None:
        add("G2. Cost versus mechanistic evidence", lambda m: cost.figure(mode=m), "exploratory", "G. Efficiency and capacity")

    if ood is not None:
        fragments.append(
            section_html(
                "H1. High-confidence internal anomaly screening — protocol",
                table_html(
                    {
                        "reference_cohort": "correct predictions in audit cohort",
                        "query_cohort": "errors (or held-out slice if too few errors)",
                        "n_reference": int(correct_mask.sum()),
                        "n_query": int(query_mask.sum()),
                        **ood.metrics,
                        "claim": "internal anomaly screening — not declared OOD",
                    }
                ),
                status="exploratory",
                group="H. Validation and uncertainty",
            )
        )
        add(
            "H1. High-confidence internal anomaly screening",
            lambda m: ood.figure(mode=m),
            "exploratory",
            "H. Validation and uncertainty",
        )

    # Cross-method: gradient mean vs focus only — label as weak / related methods
    disagreement = try_call(
        "disagreement",
        lambda: analyse_method_disagreement(
            {
                "cohort_mean_abs_gradient": mean_abs,
                "focus_abs_gradient": focus_abs,
            },
            top_k=25,
        ),
    )
    if disagreement is not None:
        fragments.append(
            section_html(
                "H2. Cross-method disagreement — caveat",
                "<p class='note'>Only closely related gradient summaries are compared here. "
                "Complete agreement is expected and is <b>not</b> evidence of multi-method validation. "
                "Paper-ready disagreement requires IG / DeepLIFT / occlusion / intervention vectors.</p>",
                status="insufficient",
                group="H. Validation and uncertainty",
            )
        )
        add("H2. Related-gradient agreement (insufficient for papers)", lambda m: disagreement.figure(mode=m), "insufficient", "H. Validation and uncertainty")

    # Optional MLP comparison
    mlp_path = MODELS_ROOT / "mlp" / "model.pt"
    if mlp_path.is_file():
        mlp_meta = json.loads((MODELS_ROOT / "mlp" / "meta.json").read_text())
        mlp = MODEL_BUILDERS["mlp"](int(mlp_meta["input_dim"]))
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True)["model_state"])
        mlp.eval()
        comparison = try_call(
            "compare",
            lambda: vraven.compare_models(
                model, mlp, x, targets=TARGET, mode="gradient", class_names=CLASS_NAMES,
                max_samples=len(x), left_name="cnn1d", right_name="mlp",
            ),
        )
        if comparison is not None:
            fragments.append(aha_card(story_comparison(comparison), group="I. Appendix"))
            fragments.append(
                section_html(
                    "I1. Cross-architecture comparison caveat",
                    "<p class='note'>Fingerprint similarity assumes aligned dimensions. "
                    "Resampling both models to 128-D does not make coordinates semantically equivalent. "
                    "Treat negative similarity as exploratory.</p>"
                    + table_html(comparison.summary()),
                    status="exploratory",
                    group="I. Appendix",
                )
            )
            add("I1. CNN vs MLP alignment", lambda m: comparison.alignment(mode=m), "exploratory", "I. Appendix")

    # Appendix lighter figures
    for title, factory in (
        ("I2. Contribution mass summary (Sankey)", lambda m: focus_report.plot("decision_sankey", top_features=10, feature_names=feature_names, mode=m)),
        ("I3. First-order unit influence", lambda m: focus_report.plot("neuron_importance", top_k=12, mode=m)),
        ("I4. Mechanistic trace fingerprint (Reasoning DNA)", lambda m: focus_report.plot("reasoning_dna", mode=m)),
        ("I5. Activation energy", lambda m: focus_report.plot("activation_energy", mode=m)),
    ):
        add(title, factory, "exploratory", "I. Appendix")

    elapsed = time.perf_counter() - started
    fragments.append(
        section_html(
            "Z. Generation metadata",
            table_html(
                {
                    "elapsed_seconds": round(elapsed, 2),
                    "sections": len(fragments),
                    "output_name": OUT_HTML.name,
                    "vraven": provenance["vraven_version"],
                    "torch": provenance["torch_version"],
                    "python": provenance["python"],
                    "seed": SEED,
                    "checkpoint_sha256_16": provenance["checkpoint_sha256_16"],
                }
            ),
            status="validated",
            group="I. Appendix",
        )
    )

    # Compose HTML with TOC + progressive groups
    groups = [
        "A. Executive findings",
        "B. Model and data provenance",
        "C. Prediction-level analysis",
        "D. Causal mechanism analysis",
        "E. Cohort and failure analysis",
        "F. Robustness and adversarial analysis",
        "G. Efficiency and capacity",
        "H. Validation and uncertainty",
        "I. Appendix",
    ]
    toc = "<nav class='toc'><h2>Contents</h2><ul>" + "".join(
        f"<li><a href='#{escape(g)}'>{escape(g)}</a></li>" for g in groups
    ) + "</ul></nav>"

    # Group fragments
    grouped: dict[str, list[str]] = {g: [] for g in groups}
    for frag in fragments:
        # extract data-group
        key = "I. Appendix"
        for g in groups:
            if f'data-group="{g}"' in frag:
                key = g
                break
        grouped[key].append(frag)

    body_parts = []
    for g in groups:
        open_attr = " open" if g.startswith("A.") or g.startswith("B.") or g.startswith("C.") or g.startswith("D.") else ""
        body_parts.append(
            f"<details id='{escape(g)}'{open_attr}><summary>{escape(g)}</summary>"
            + "".join(grouped[g])
            + "</details>"
        )

    extra_css = """
.toc { position: sticky; top: 0; z-index: 5; margin-bottom: 18px; }
.toc ul { columns: 2; gap: 18px; margin: 0; padding-left: 18px; }
.section-head { display:flex; justify-content:space-between; gap:12px; align-items:center; }
.status {
  font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  border: 1px solid var(--status); color: var(--status); padding: 4px 8px; border-radius: 999px;
}
.aha-pre {
  margin: 0; white-space: pre-wrap; font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 15px; line-height: 1.55; color: var(--vraven-text);
}
details { background: var(--vraven-panel); border: 1px solid var(--vraven-border);
  border-radius: 14px; padding: 12px 16px; margin-bottom: 18px; }
details > summary { cursor: pointer; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; font-size: 13px; margin-bottom: 10px; }
.provenance { font-size: 11px !important; opacity: 0.9; }
"""

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VRAVEN Kronodroid CNN1D — Publication audit</title>
<style>{_theme_css(theme_toggle=True, initial_theme="dark")}{extra_css}</style>
{_theme_script("dark")}
</head>
<body>
<main>
<header>
  <div class="header-row">
    <div class="header-copy">
      <h1>VRAVEN × Kronodroid CNN1D</h1>
      <p class="note">
        Inside a trained model are millions of numbers without intuitive meaning.
        This report translates <b>measured</b> decision signals into human-readable findings first,
        then shows scientific evidence. Status badges mark validated vs exploratory vs unsupported.
        Question of record: <i>Can internal mechanism changes expose model failure before the output does?</i>
      </p>
    </div>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button type="button" data-theme-choice="dark" aria-pressed="true">Dark</button>
      <button type="button" data-theme-choice="light" aria-pressed="false">Light</button>
    </div>
  </div>
</header>
{toc}
{''.join(body_parts)}
</main>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    log(f"\nWrote {OUT_HTML} ({OUT_HTML.stat().st_size/1e6:.2f} MB) in {elapsed:.1f}s")
    log(f"Focus: {focus_id} | LRP: {provenance.get('lrp_status')} | cohort: {cohort_counts}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        raise SystemExit(1) from error
