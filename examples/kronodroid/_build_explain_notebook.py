#!/usr/bin/env python3
"""Regenerate the Kronodroid deep single-model audit notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples" / "notebooks" / "kronodroid_vraven_explain.ipynb"

nb: dict = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "torch-mps",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.11"},
    },
    "cells": [],
}


def _lines(src: str) -> list[str]:
    text = src.strip("\n")
    return [line + "\n" for line in text.split("\n")]


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})


def code(src: str) -> None:
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(src),
        }
    )


md(
    """
# Kronodroid × VRAVEN — Deep Single-Model Audit

**One model at a time.** Set `MODEL_NAME` below (`mlp` | `cnn1d` | `resnet1d` | `bigru`), then run top → bottom.

This notebook exercises **every VRAVEN surface that applies** to the selected model:

1. Core explain (gradient cohort + causal focus)
2. LRP / perturbation / counterfactual
3. All standalone core figures (skipped cleanly if unsupported)
4. Dataset landscape / similarity / outliers
5. Full DarkSide internals (circuitry, wiring, anatomy, capacity, …)
6. Optional second-model comparison + HTML export

Insights are printed as measured numbers, not decorative captions.
"""
)

md("## 0 — Install into this kernel")
code(
    """
from pathlib import Path
import importlib.util
import subprocess
import sys

VRAVEN_ROOT = next(
    (
        candidate
        for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "src" / "vraven").is_dir()
    ),
    None,
)
if VRAVEN_ROOT is None:
    raise RuntimeError("Start Jupyter from the root of a VRAVEN source checkout")

def _missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None

needs = []
if _missing("vraven"):
    needs.append(("vraven", ["-e", str(VRAVEN_ROOT)]))
if _missing("torch"):
    needs.append(("torch", ["torch"]))
if _missing("plotly"):
    needs.append(("plotly", ["plotly"]))
if _missing("sklearn"):
    needs.append(("scikit-learn", ["scikit-learn"]))

if needs:
    print("Installing:", [n for n, _ in needs])
    for _, args in needs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
    print("Restart kernel, then continue from cell 1.")
else:
    print("Kernel OK:", sys.executable)
"""
)

md(
    """
## 1 — Choose ONE model

Change only `MODEL_NAME`. Re-run from this cell after switching architectures.
"""
)

code(
    """
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EXAMPLE = VRAVEN_ROOT / "examples" / "kronodroid"
MODELS_ROOT = VRAVEN_ROOT / "Models" / "kronodroid"
SRC = VRAVEN_ROOT / "src"

# ═══ ONE MODEL AT A TIME ═══
MODEL_NAME = "mlp"            # mlp | cnn1d | resnet1d | bigru
COMPARE_MODEL_NAME = "cnn1d"  # optional paired model for compare_models
N_SAMPLES = 48
TARGET = 1                    # malware
PLOT_MODE = "interactive_dark"
EXPLAIN_MODE = "gradient"     # cohort-wide; causal used separately
# ═════════════════════════════

MODEL_DIR = MODELS_ROOT / MODEL_NAME
OUTPUT_DIR = VRAVEN_ROOT / "vraven_outputs" / "kronodroid" / MODEL_NAME
for path in (EXAMPLE, MODEL_DIR, SRC, MODELS_ROOT / "split.npz"):
    assert path.exists(), path

for path in (str(SRC), str(EXAMPLE)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

for name in list(sys.modules):
    if name == "vraven" or name.startswith("vraven."):
        del sys.modules[name]

import vraven
from vraven import VRavenTheme
from models import MODEL_BUILDERS

from vraven.darkside import (
    DarkSideSession,
    InputFeatureGroup,
    TraceDataset,
    UnitGroup,
    analyse_activation_mediation,
    analyse_adversarial_rerouting,
    analyse_causal_synergy,
    analyse_circuit_multiplicity,
    analyse_cost_evidence_tradeoff,
    analyse_explanation_uncertainty,
    analyse_failure_anatomy,
    analyse_high_confidence_ood,
    analyse_method_disagreement,
    analyse_prediction_depth,
    analyse_representation_drift,
    analyse_stability_geometry,
    analyse_topological_shift,
    audit_unused_capacity,
    build_circuit_wiring,
    compare_class_conditionals,
    inspect_internal_anatomy,
    layerwise_shortcut_probe,
    map_class_conditional_unit_field,
    print_internal_anatomy,
    recommend_features,
)

VRavenTheme.apply()
CLASS_NAMES = {0: "benign", 1: "malware"}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("vraven", getattr(vraven, "__version__", "?"))
print("ACTIVE MODEL →", MODEL_NAME)
print("compare with →", COMPARE_MODEL_NAME)
print("outputs →", OUTPUT_DIR)
"""
)

md("## 2 — Load active model + scaled test split")
code(
    """
meta = json.loads((MODEL_DIR / "meta.json").read_text())
payload = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
feature_names = list(meta["feature_names"])
display_names = feature_names

model = MODEL_BUILDERS[MODEL_NAME](int(meta["input_dim"]))
model.load_state_dict(payload["model_state"])
model.eval()

split = np.load(MODELS_ROOT / "split.npz", allow_pickle=True)
x_test = np.asarray(split["x_test"], dtype=np.float32)
y_test = np.asarray(split["y_test"], dtype=np.int64)

LINEAR_LAYERS = tuple(
    name for name, mod in model.named_modules()
    if name and isinstance(mod, torch.nn.Linear)
)
CONV_LAYERS = tuple(
    name for name, mod in model.named_modules()
    if name and isinstance(mod, torch.nn.Conv1d)
)
INTERVENE_LAYERS = LINEAR_LAYERS or CONV_LAYERS

print("architecture", MODEL_NAME, "input_dim", meta["input_dim"])
print("held-out test metrics:")
print(json.dumps(meta.get("test_metrics", {}), indent=2))
print("linear layers", LINEAR_LAYERS)
print("conv layers", CONV_LAYERS)
print("test pool", x_test.shape)
"""
)

md("## 3 — Helpers (safe plot + insight printer)")
code(
    """
def insight(title: str, mapping=None, **kwargs) -> None:
    print(f"\\n══ {title} ══")
    data = dict(mapping or {})
    data.update(kwargs)
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")


def show_plot(obj, name: str, **kwargs) -> bool:
    try:
        if hasattr(obj, "plot"):
            fig = obj.plot(name, mode=PLOT_MODE, **kwargs)
        else:
            fig = getattr(obj, name)(mode=PLOT_MODE, **kwargs)
        fig.show()
        print(f"[ok] {name}")
        return True
    except Exception as exc:
        print(f"[skip] {name}: {type(exc).__name__}: {exc}")
        return False


def show_darkside(result, title: str | None = None) -> None:
    if title:
        print(f"\\n── {title} ──")
    if hasattr(result, "summary"):
        print(result.summary())
    try:
        result.figure(mode=PLOT_MODE).show()
        print("[ok] darkside figure")
    except Exception as exc:
        print(f"[skip] darkside figure: {type(exc).__name__}: {exc}")
"""
)

md("## 4 — Balanced cohort + focus malware sample")
code(
    """
rng = np.random.default_rng(42)
mal_idx = np.flatnonzero(y_test == 1)
ben_idx = np.flatnonzero(y_test == 0)
n_mal = min(N_SAMPLES // 2 + N_SAMPLES % 2, len(mal_idx))
n_ben = min(N_SAMPLES - n_mal, len(ben_idx))
pick = np.concatenate([
    rng.choice(mal_idx, size=n_mal, replace=False),
    rng.choice(ben_idx, size=n_ben, replace=False),
])
rng.shuffle(pick)

x = torch.tensor(x_test[pick], dtype=torch.float32)
labels = y_test[pick]
label_names = [CLASS_NAMES[int(v)] for v in labels]

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = probs.argmax(axis=1)

focus = int(np.argmax(probs[:, TARGET]))
x_focus = x[focus : focus + 1].clone()
correct = pred == labels

insight(
    "Cohort prediction snapshot",
    samples=len(x),
    malware=int((labels == 1).sum()),
    benign=int((labels == 0).sum()),
    cohort_accuracy=float(correct.mean()),
    focus_index=focus,
    focus_true=label_names[focus],
    focus_pred=CLASS_NAMES[int(pred[focus])],
    focus_p_malware=float(probs[focus, TARGET]),
    focus_margin=float(probs[focus].max() - np.partition(probs[focus], -2)[-2]),
)
"""
)

md("## 5 — Core explain (gradient cohort) + causal focus")
code(
    """
report = vraven.explain(
    model,
    x,
    target=TARGET,
    mode=EXPLAIN_MODE,
    class_names=CLASS_NAMES,
    metadata={
        "task": "android_kronodroid_malware_binary",
        "dataset": "KronoDroid",
        "model_name": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "feature_names": feature_names,
    },
)

causal_report = vraven.explain(
    model,
    x_focus,
    target=TARGET,
    mode="causal",
    class_names=CLASS_NAMES,
    metadata={
        "dataset": "KronoDroid",
        "model_name": MODEL_NAME,
        "feature_names": feature_names,
        "role": "causal_focus",
    },
)

insight("Gradient report", report.summary())
insight("Causal focus report", causal_report.summary())
caps = report.capabilities()
insight("Enabled core figures", {k: v for k, v in sorted(caps.items()) if v})
insight("Disabled / unavailable figures", {k: v for k, v in sorted(caps.items()) if not v})
"""
)

md("## 6 — Attach LRP, perturbation, counterfactual")
code(
    """
try:
    report.lrp_result = vraven.explain_lrp(model, x, target=TARGET)
    print("LRP attached on cohort:", report.lrp_result is not None)
except Exception as exc:
    report.lrp_result = None
    print(f"LRP skipped: {type(exc).__name__}: {exc}")

report.perturbation_result = vraven.run_perturbation_stability(
    model, x, target=TARGET, repeats=5, random_seed=17
)
print("perturbation curves", len(report.perturbation_result.curves))
for curve in report.perturbation_result.curves:
    print(f"  {curve.kind}: levels={curve.levels} stability={np.round(curve.stability, 4)}")

try:
    report.counterfactual_result = vraven.generate_counterfactual(
        model,
        x_focus.clone(),
        target_class=0,
        steps=160,
        learning_rate=0.05,
        l2_weight=0.01,
        top_changes=20,
    )
    cf = report.counterfactual_result
    insight(
        "Counterfactual malware→benign",
        success=cf.success,
        original_class=getattr(cf, "original_class", None),
        final_class=getattr(cf, "final_class", None),
        l2=getattr(cf, "l2_distance", getattr(cf, "distance", None)),
    )
except Exception as exc:
    report.counterfactual_result = None
    print(f"counterfactual skipped: {type(exc).__name__}: {exc}")
"""
)

md("## 7 — Deep attribution insight (top features + layer signals)")
code(
    """
attrs = np.asarray(report.trace.inputs.attribution_values, dtype=np.float64)
if attrs.ndim == 1:
    attrs = attrs.reshape(1, -1)
mean_abs = np.abs(attrs).mean(axis=0)
# VRAVEN may return a batch-aggregated attribution vector (shape [F] or [1, F]).
focus_row = min(int(focus), attrs.shape[0] - 1)
focus_abs = np.abs(attrs[focus_row]).reshape(-1)
order = np.argsort(mean_abs)[::-1]

print("Top 25 features by mean |attribution| across cohort:")
for rank, idx in enumerate(order[:25], 1):
    print(
        f"  {rank:2d}. {feature_names[idx]:40s}  "
        f"mean|a|={mean_abs[idx]:.5g}  focus|a|={focus_abs[idx]:.5g}"
    )

print("\\nLayer reduced metrics:")
for layer in report.trace.layers:
    metrics = getattr(layer, "metrics", None) or {}
    name = getattr(layer, "name", None) or getattr(layer, "layer_name", "?")
    bits = []
    for key in ("energy", "mean_energy", "sparsity", "abs_interaction", "influence"):
        if key in metrics:
            bits.append(f"{key}={metrics[key]:.4g}")
    if bits:
        print(f"  {name}: " + ", ".join(bits))

fp = report.reasoning_fingerprint(dimensions=128)
insight(
    "Reasoning fingerprint",
    width=fp.size,
    l2=float(np.linalg.norm(fp)),
    mean=float(fp.mean()),
    std=float(fp.std()),
)
"""
)

md("## 8 — Consistency explanations (multi-method agreement)")
code(
    """
grad_vec = mean_abs.copy()
consistency_explanations = {"input_times_gradient": grad_vec.tolist()}

lrp_feature = np.zeros_like(grad_vec)
lrp_count = 0
for i in range(min(12, x.shape[0])):
    try:
        single = vraven.explain_lrp(model, x[i : i + 1], target=TARGET)
    except Exception:
        continue
    if getattr(single, "input_relevance", None) is None:
        continue
    vec = np.abs(np.asarray(single.input_relevance, dtype=np.float64)).reshape(-1)
    if vec.size != grad_vec.size:
        continue
    lrp_feature += vec
    lrp_count += 1

if lrp_count:
    consistency_explanations["lrp_epsilon"] = (lrp_feature / lrp_count).tolist()
else:
    consistency_explanations["focus_abs_gradient"] = focus_abs.reshape(-1).tolist()

print("consistency methods", list(consistency_explanations), "n_lrp", lrp_count)
"""
)

md("---\n# Core VRAVEN figures (all that apply)")

core_figures = [
    (
        "F01 — Feature Importance",
        'show_plot(report, "input_attribution", top_k=25, feature_names=display_names)',
    ),
    (
        "F02 — Feature Attribution Heatmap",
        'show_plot(report, "feature_sequence_heatmap", top_features=48, feature_names=display_names)',
    ),
    (
        "F03 — Contribution Sankey",
        'show_plot(report, "decision_sankey", top_features=12, feature_names=display_names)',
    ),
    ("F04 — Decision Flow", 'show_plot(report, "decision_flow")'),
    ("F05 — Neuron Impact Field", 'show_plot(report, "neuron_impact")'),
    ("F06 — Neuron Importance", 'show_plot(report, "neuron_importance", top_k=16)'),
    (
        "F07 — Latent Flight / Representation Landscape",
        'show_plot(report, "latent_flight", method="tsne")',
    ),
    (
        "F08 — LRP Relevance Flow",
        """
if report.lrp_result is not None:
    show_plot(report, "lrp", max_layers=8)
else:
    print("[skip] lrp: not attached for this architecture")
""",
    ),
    (
        "F09 — Counterfactual Delta",
        """
if report.counterfactual_result is not None:
    show_plot(
        report,
        "counterfactual_delta",
        feature_names=display_names,
        class_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
else:
    print("[skip] counterfactual_delta")
""",
    ),
    ("F10 — Perturbation Stability", 'show_plot(report, "perturbation")'),
    (
        "F11 — Consistency Matrix",
        'show_plot(report, "consistency_matrix", explanations=consistency_explanations, metric="rank_correlation")',
    ),
    ("F12 — Reasoning DNA", 'show_plot(report, "reasoning_dna")'),
    ("F13 — Spectral Trace", 'show_plot(report, "spectral_trace")'),
    ("F14 — Conflict Field", 'show_plot(report, "conflict_field")'),
    ("F15 — Evidence Trajectory", 'show_plot(report, "evidence_trajectory")'),
    ("F16 — Causal Pathway", 'show_plot(causal_report, "causal_pathway")'),
    ("F17 — Activation Energy", 'show_plot(report, "activation_energy")'),
    ("F18 — Reasoning Graph", 'show_plot(report, "reasoning_graph")'),
    ("F19 — Activation Community Graph", 'show_plot(report, "community_graph")'),
    ("F20 — Attention Structure", 'show_plot(report, "attention_structure")'),
]

for title, body in core_figures:
    md(f"## {title}")
    code(body)

md("---\n# Dataset-level VRAVEN")
md("## D00 — Independent per-sample dataset report")
code(
    """
dataset_report = vraven.explain_dataset(
    model,
    [
        vraven.DatasetItem(
            inputs=x[i : i + 1],
            target=TARGET,
            label=int(labels[i]),
            sample_id=f"{label_names[i]}_{i}",
        )
        for i in range(len(labels))
    ],
    mode=EXPLAIN_MODE,
    class_names=CLASS_NAMES,
    max_samples=len(labels),
    fingerprint_dimensions=128,
)
print("dataset samples", len(labels))
if hasattr(dataset_report, "summary"):
    print(dataset_report.summary())
"""
)

md("## D01 — Reasoning Landscape")
code(
    """
dataset_report.reasoning_landscape(
    colour_by="label",
    method="tsne",
    highlight_index=focus,
    random_seed=17,
    perplexity=min(15.0, max(5.0, len(labels) / 4.0)),
    mode=PLOT_MODE,
).show()
"""
)

md("## D02 — Reasoning Similarity")
code("dataset_report.reasoning_similarity(mode=PLOT_MODE).show()")

md("## D03 — Reasoning Outliers")
code("dataset_report.reasoning_outliers(neighbours=5, mode=PLOT_MODE).show()")

md("---\n# DarkSide — deeper internal mechanism audit")
md("## DS00 — Session + TraceDataset + feature groups")
code(
    """
session = DarkSideSession(model, x_focus, target=TARGET)

syscall_idx = [i for i, n in enumerate(feature_names) if n[:1].islower() and n != "total_perm"]
perm_idx = [
    i for i, n in enumerate(feature_names)
    if n.isupper() or n.startswith(("BIND_", "ACCESS_", "READ_", "WRITE_", "RECEIVE_", "SEND_"))
]
struct_idx = [
    i for i, n in enumerate(feature_names)
    if n.startswith("Nr")
    or n in {
        "Activities", "FilesInsideAPK", "CFileSize", "UFileSize",
        "TimesSubmitted", "NrContactedIps", "TotalIntentFilters", "total_perm",
    }
]

def _take(indices, k=24):
    return tuple(indices[:k]) if indices else tuple(range(min(8, len(feature_names))))

feature_groups = (
    InputFeatureGroup(_take(syscall_idx), label="syscalls"),
    InputFeatureGroup(_take(perm_idx), label="permissions"),
    InputFeatureGroup(_take(struct_idx, 16), label="apk_structure"),
)

# Shared ids so paired DarkSide analyses (drift/stability/reroute/topo) align.
paired_ids = tuple(f"kronodroid_{i}_{label_names[i]}" for i in range(len(labels)))

_ds_reports = [
    vraven.explain(model, x[i : i + 1], target=TARGET, mode=EXPLAIN_MODE)
    for i in range(len(labels))
]
dark_dataset = TraceDataset.from_reports(
    _ds_reports, true_labels=labels.tolist(), sample_ids=paired_ids
)

x_noisy = x + 0.15 * torch.randn_like(x)
_ds_noisy = [
    vraven.explain(model, x_noisy[i : i + 1], target=TARGET, mode=EXPLAIN_MODE)
    for i in range(len(labels))
]
dark_dataset_noisy = TraceDataset.from_reports(
    _ds_noisy, true_labels=labels.tolist(), sample_ids=paired_ids
)

print("feature groups", [g.identifier for g in feature_groups])
print("recommend(security)", [f.name for f in recommend_features("security")])
print("intervene layers", INTERVENE_LAYERS)
"""
)

darkside = [
    (
        "DS01 — Decision circuitry",
        """
circuit = session.decision_circuitry(
    layer_names=INTERVENE_LAYERS[: min(3, len(INTERVENE_LAYERS))] or None,
    group_size=16 if MODEL_NAME == "mlp" else 8,
    relative_score_tolerance=0.10,
    maximum_groups=96,
)
show_darkside(circuit, "decision circuitry")
retained = [r for r in circuit.tables.get("groups", []) if r.get("retained")]
print("retained groups", len(retained), "of", len(circuit.tables.get("groups", [])))
""",
    ),
    (
        "DS02 — Measured circuit wiring",
        """
try:
    wiring = build_circuit_wiring(model, x_focus, circuit, include_removed=False)
    show_darkside(wiring, "circuit wiring")
    print("edges", wiring.metrics.get("edges"), "total mass", wiring.metrics.get("total_edge_mass"))
except Exception as exc:
    print(f"[skip] circuit_wiring: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS03 — Internal Linear anatomy",
        """
try:
    anatomy = inspect_internal_anatomy(
        model, x_focus, target=TARGET,
        layer_names=LINEAR_LAYERS or None,
        top_units=12,
    )
    print(print_internal_anatomy(anatomy))
    show_darkside(anatomy, "internal anatomy")
except Exception as exc:
    print(f"[skip] internal_anatomy: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS04 — Shortcut reliance",
        """
shortcut = session.shortcut_detection(feature_groups, baseline="zero")
show_darkside(shortcut, "shortcut reliance")
""",
    ),
    (
        "DS05 — Class-conditional internals",
        """
class_cond = compare_class_conditionals(dark_dataset, class_a=0, class_b=1)
show_darkside(class_cond, "class conditionals")
""",
    ),
    (
        "DS06 — Class-conditional unit field",
        """
unit_field = map_class_conditional_unit_field(
    dark_dataset, class_a=0, class_b=1, max_units_per_layer=64
)
show_darkside(unit_field, "unit field")
""",
    ),
    (
        "DS07 — Representation drift",
        """
drift = analyse_representation_drift(dark_dataset, dark_dataset_noisy, sliced_projections=16)
show_darkside(drift, "representation drift")
""",
    ),
    (
        "DS08 — Failure anatomy",
        """
try:
    failure = analyse_failure_anatomy(dark_dataset, failure="fp", reference="tp")
    show_darkside(failure, "failure anatomy")
except Exception as exc:
    print(f"[skip] failure_anatomy: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS09 — Stability geometry",
        """
stability = analyse_stability_geometry(dark_dataset, dark_dataset_noisy, neighbours=5)
show_darkside(stability, "stability geometry")
""",
    ),
    (
        "DS10 — Cross-method disagreement",
        """
vectors = {
    "gradient": np.asarray(consistency_explanations["input_times_gradient"]),
}
for key, value in consistency_explanations.items():
    if key != "input_times_gradient":
        vectors[key] = np.asarray(value)
disagreement = analyse_method_disagreement(vectors, top_k=25)
show_darkside(disagreement, "method disagreement")
""",
    ),
    (
        "DS11 — Unused capacity",
        """
capacity = audit_unused_capacity(dark_dataset)
show_darkside(capacity, "unused capacity")
""",
    ),
    (
        "DS12 — Adversarial / evasion rerouting",
        """
reroute = analyse_adversarial_rerouting(dark_dataset, dark_dataset_noisy, neighbours=5)
show_darkside(reroute, "adversarial rerouting")
""",
    ),
    (
        "DS13 — Cost–evidence tradeoff",
        """
cost = analyse_cost_evidence_tradeoff(dark_dataset)
show_darkside(cost, "cost-evidence")
""",
    ),
    (
        "DS14 — Prediction depth",
        """
depth = analyse_prediction_depth(dark_dataset, neighbours=5)
show_darkside(depth, "prediction depth")
""",
    ),
    (
        "DS15 — Layerwise shortcut probe",
        """
try:
    probe = layerwise_shortcut_probe(dark_dataset)
    show_darkside(probe, "layerwise shortcut probe")
except Exception as exc:
    print(f"[skip] layerwise_shortcut_probe: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS16 — Circuit multiplicity",
        """
masks = []
universe = None
for seed in range(4):
    torch.manual_seed(seed)
    r = session.decision_circuitry(
        layer_names=INTERVENE_LAYERS[: min(2, len(INTERVENE_LAYERS))] or None,
        group_size=16 if MODEL_NAME == "mlp" else 8,
        relative_score_tolerance=0.12,
        maximum_groups=64,
    )
    if universe is None:
        universe = [row["group"] for row in r.tables["groups"]]
    retained = {row["group"] for row in r.tables["groups"] if row["retained"]}
    masks.append([1 if g in retained else 0 for g in universe])
mask_arr = np.asarray(masks, dtype=int)
multiplicity = analyse_circuit_multiplicity(mask_arr, confidence=0.80, stable_threshold=0.50)
show_darkside(multiplicity, "circuit multiplicity")
""",
    ),
    (
        "DS17 — Causal synergy",
        """
try:
    groups = []
    for row in circuit.tables["groups"]:
        if not row.get("retained"):
            continue
        layer_name = str(row["layer"]).split("#", 1)[0]
        groups.append(UnitGroup(layer_name, tuple(row["indices"][:8]), label=str(row["group"])[:24]))
        if len(groups) >= 3:
            break
    if len(groups) >= 2:
        synergy = analyse_causal_synergy(model, x_focus, target=TARGET, groups=groups[:3])
        show_darkside(synergy, "causal synergy")
    else:
        print("[skip] causal_synergy: need >=2 retained groups")
except Exception as exc:
    print(f"[skip] causal_synergy: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS18 — Activation mediation",
        """
try:
    corrupt = torch.zeros_like(x_focus)
    mediation = analyse_activation_mediation(
        model,
        x_focus,
        corrupt,
        target=TARGET,
        layer_names=INTERVENE_LAYERS[: min(3, len(INTERVENE_LAYERS))],
    )
    show_darkside(mediation, "activation mediation")
except Exception as exc:
    print(f"[skip] activation_mediation: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS19 — Topological shift",
        """
try:
    topo = analyse_topological_shift(dark_dataset, dark_dataset_noisy, maximum_points=40)
    show_darkside(topo, "topological shift")
except Exception as exc:
    print(f"[skip] topological_shift: {type(exc).__name__}: {exc}")
""",
    ),
    (
        "DS20 — Explanation uncertainty",
        """
repeats = []
for seed in range(6):
    g = torch.Generator().manual_seed(seed)
    noise = 0.05 * torch.randn(x_focus.shape, generator=g)
    r = vraven.explain(model, x_focus + noise, target=TARGET, mode=EXPLAIN_MODE)
    vec = np.abs(np.asarray(r.trace.inputs.attribution_values, dtype=np.float64)).reshape(-1)
    repeats.append(vec)
uncertainty = analyse_explanation_uncertainty(np.stack(repeats, axis=0), top_k=20, bootstrap_repeats=80)
show_darkside(uncertainty, "explanation uncertainty")
""",
    ),
    (
        "DS21 — High-confidence OOD",
        """
fps = np.asarray(dark_dataset.fingerprints, dtype=np.float64)
conf = probs[:, TARGET]
half = max(4, len(fps) // 2)
ood = analyse_high_confidence_ood(fps[:half], fps[half:], query_confidence=conf[half:])
show_darkside(ood, "high-confidence OOD")
""",
    ),
]

for title, body in darkside:
    md(f"## {title}")
    code(body)

md("---\n# Optional: matched comparison against a second model")
md("## C01 — compare_models")
code(
    """
compare_path = MODELS_ROOT / COMPARE_MODEL_NAME / "model.pt"
if COMPARE_MODEL_NAME == MODEL_NAME:
    print("Set COMPARE_MODEL_NAME to a different architecture to compare.")
elif not compare_path.is_file():
    print("Missing compare model", compare_path)
else:
    other_meta = json.loads((MODELS_ROOT / COMPARE_MODEL_NAME / "meta.json").read_text())
    other = MODEL_BUILDERS[COMPARE_MODEL_NAME](int(other_meta["input_dim"]))
    other_payload = torch.load(compare_path, map_location="cpu", weights_only=True)
    other.load_state_dict(other_payload["model_state"])
    other.eval()
    comparison = vraven.compare_models(
        model,
        other,
        x,
        targets=TARGET,
        mode=EXPLAIN_MODE,
        class_names=CLASS_NAMES,
        max_samples=len(x),
        left_name=MODEL_NAME,
        right_name=COMPARE_MODEL_NAME,
    )
    print(comparison.summary() if hasattr(comparison, "summary") else comparison)
    plotted = False
    for method in ("figure", "plot"):
        try:
            getattr(comparison, method)(mode=PLOT_MODE).show()
            print("[ok] comparison", method)
            plotted = True
            break
        except Exception as exc:
            print(f"[skip] comparison.{method}: {type(exc).__name__}: {exc}")
    if not plotted:
        print("Comparison object created; inspect summary above.")
"""
)

md("---\n# Export")
md("## E01 — Save trace + HTML for the active model")
code(
    """
trace_path = OUTPUT_DIR / f"{MODEL_NAME}_focus_trace.json"
# save the causal focus report if available, else gradient report
try:
    causal_report.save_trace(OUTPUT_DIR / f"{MODEL_NAME}_causal_focus_trace.json")
except Exception:
    pass
report.save_trace(trace_path)
print("saved trace", trace_path)

html_path = OUTPUT_DIR / f"{MODEL_NAME}_vraven_report.html"
try:
    report.export_html(html_path, mode=PLOT_MODE, theme_toggle=True)
    print("saved HTML", html_path)
except Exception as exc:
    print(f"[skip] export_html: {type(exc).__name__}: {exc}")

insight(
    "Audit complete",
    active_model=MODEL_NAME,
    cohort=len(x),
    focus=focus,
    focus_p_malware=float(probs[focus, TARGET]),
    outputs=str(OUTPUT_DIR),
)
print("\\nNext model: change MODEL_NAME → re-run from cell 1 → Run All.")
"""
)

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(nb['cells'])} cells ({OUT.stat().st_size} bytes)")
