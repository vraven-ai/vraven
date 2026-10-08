"""VRAVEN Adversarial Mechanism Audit (stress + internal ΔM; not a certificate)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .adversarial_evidence import (
    RobustnessProfileReport,
    attack_adequacy_audit,
    build_robustness_profile,
)
from .adversarial_mechanism import (
    MechanismProbe,
    MechanismSnapshot,
    build_activation_probe,
    build_concept_probe_from_directions,
    cluster_failure_families,
    compare_mechanisms,
    cross_attack_agreement,
    ensure_unit_names,
    failure_mode_spectrum,
    mechanism_instability_radius,
    survival_margin_rows,
)
from .runtime import extract_logits, predicted_class, target_margin

LossKind = Literal[
    "untargeted",
    "cross_entropy",
    "cw_logit",
    "dlr",
    "targeted",
    "targeted_dlr",
]

DEFAULT_STRESS_SUITE: tuple[str, ...] = (
    "fgsm",
    "bim",
    "mifgsm",
    "pgd_linf",
    "pgd_l2",
    "apgd_ce",
    "apgd_dlr",
    "fab",
    "deepfool",
    "square",
)

# Croce & Hein-inspired order with deliberately small VRAVEN pilot budgets.  This
# is not the reference AutoAttack protocol and must not be reported as one.
AUTOATTACK_SUITE: tuple[str, ...] = ("apgd_ce", "apgd_dlr", "fab", "square")

_ATTACK_LABELS = {
    "fgsm": "FGSM",
    "bim": "BIM / I-FGSM",
    "mifgsm": "MI-FGSM",
    "pgd_linf": "PGD-L∞",
    "pgd_l2": "PGD-L2",
    "apgd": "APGD-CE (VRAVEN-lite)",  # alias → apgd_ce
    "apgd_ce": "APGD-CE (VRAVEN-lite)",
    "apgd_dlr": "APGD-DLR (VRAVEN-lite)",
    "pgd_dlr": "PGD-DLR / CW-logit",
    "fab": "FAB (VRAVEN-lite)",
    "deepfool": "DeepFool",
    "square": "Flat-block query search",
    "autoattack": "VRAVEN ensemble-lite",
}

_LIMITATIONS = (
    "Layer 1 reports empirical flip radii on a finite ε grid / iteration budget — "
    "not certified robustness radii.",
    "Layer 2 mechanism survival uses a frozen probe (activation fingerprint or "
    "supplied concept directions); it is not a second full reverse() rediscovery.",
    "White-box means model-artefact gradients (∂loss/∂x), not feature semantics / "
    "owner intent. Input-coordinate labels are indices unless a schema is supplied.",
    "This is mathematical adversarial stress, not domain-valid evasion "
    "(e.g. functionality-preserving malware mutations).",
    "The VRAVEN ensemble-lite path uses APGD-CE → APGD-DLR → FAB-lite → "
    "flat-block query search with pilot budgets. It is not AutoAttack or a "
    "RobustBench-compatible evaluation.",
    "No GAN or diffusion / generative attacks are executed here.",
    "Batch robust accuracy is a finite-sample estimate, not a RobustBench score.",
)


@dataclass(slots=True)
class AttackResult:
    """One measured attack trial (Layer 1 stress + optional Layer 2 ΔM)."""

    attack_name: str
    attack_family: str
    succeeded: bool
    epsilon: float
    steps: int
    prediction_flipped: bool
    perturbation_norm_l2: float
    perturbation_norm_linf: float
    original_margin: float
    adversarial_margin: float
    margin_delta: float
    original_prediction: int
    adversarial_prediction: int
    vulnerable_concepts: list[str] = field(default_factory=list)
    norm: str = "linf"  # "linf" | "l2" | "unrestricted"
    mechanism_survival: float | None = None
    mechanism_delta_l1: float | None = None
    mechanism_delta_linf: float | None = None
    mechanism_suppressed: list[str] = field(default_factory=list)
    mechanism_activated: list[str] = field(default_factory=list)
    first_mechanism_divergence: str | None = None
    objective: str = "untargeted_negative_logit"
    mechanism_cosine_survival: float | None = None
    mechanism_energy_survival: float | None = None
    mechanism_top_active_cosine_survival: float | None = None
    mechanism_worst_layer_cosine_survival: float | None = None
    mechanism_worst_layer: str | None = None

    @property
    def succeded(self) -> bool:
        """Deprecated spelling retained for one release."""

        return self.succeeded


@dataclass(slots=True)
class AdversarialAuditReport:
    """Adversarial Mechanism Audit: stress testing + internal failure analysis."""

    model_name: str
    empirical_min_flip_eps_linf: float | None
    empirical_min_flip_eps_l2: float | None
    attacks_tested: list[AttackResult]
    most_vulnerable_concept: str
    vulnerable_concepts: list[str]
    recommendations: list[str]
    limitations: list[str] = field(default_factory=lambda: list(_LIMITATIONS))
    metadata: dict[str, Any] = field(default_factory=dict)
    # Alias field name kept for API stability; display as input-coordinate sensitivity.
    mechanism_instability_radius: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    def write_html(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_report_html(self), encoding="utf-8")
        return dest


AdversarialCertificate = AdversarialAuditReport


@dataclass(slots=True)
class RobustAccuracyReport:
    """Finite-sample clean / robust accuracy under a fixed threat model."""

    model_name: str
    protocol: str
    epsilon: float
    n_samples: int
    n_correct_clean: int
    n_robust: int
    clean_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    per_attack_flips: dict[str, int] = field(default_factory=dict)
    conditional_robust_accuracy: float | None = None
    prediction_invariance: float | None = None
    per_attack_successes: dict[str, int] = field(default_factory=dict)
    per_attack_unique_flips: dict[str, int] = field(default_factory=dict)
    attack_union_curve: list[dict[str, Any]] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)
    macro_robust_accuracy: float | None = None
    per_sample_outcomes: list[dict[str, Any]] = field(default_factory=list)
    sampling_protocol: str = "head"
    limitations: list[str] = field(default_factory=lambda: list(_LIMITATIONS))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    abs_v = abs(float(value))
    if abs_v != 0 and (abs_v >= 1e4 or abs_v < 1e-3):
        return f"{float(value):.3e}"
    return f"{float(value):.{digits}g}"


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for one binomial proportion.

    The default ``z`` is the 97.5th standard-normal percentile, producing a
    nominal 95% interval.  Wilson intervals remain informative at proportions
    zero and one, unlike the uncorrected Wald interval.
    """

    if trials < 1:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials")
    if z <= 0 or not math.isfinite(z):
        raise ValueError("z must be positive and finite")
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    if lower < 1e-15:
        lower = 0.0
    if 1.0 - upper < 1e-15:
        upper = 1.0
    return lower, upper


def _report_html(report: AdversarialAuditReport) -> str:
    families = sorted({a.attack_family for a in report.attacks_tested})
    family_summary = []
    for fam in families:
        trials = [a for a in report.attacks_tested if a.attack_family == fam]
        flips = sum(1 for a in trials if a.succeeded)
        label = _ATTACK_LABELS.get(fam, fam)
        family_summary.append(
            f"<div class='card'><div class='k'>{escape(label)}</div>"
            f"<div class='v' style='font-size:14px'>{flips}/{len(trials)} flips</div></div>"
        )
    rows = []
    for attack in report.attacks_tested:
        status = "FLIP" if attack.succeeded else "hold"
        status_cls = "flip" if attack.succeeded else "hold"
        rows.append(
            "<tr>"
            f"<td>{escape(attack.attack_name)}</td>"
            f"<td>{escape(_ATTACK_LABELS.get(attack.attack_family, attack.attack_family))}</td>"
            f"<td>{escape(attack.norm)}</td>"
            f"<td>{_fmt(attack.epsilon)}</td>"
            f"<td>{attack.steps}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"<td>{_fmt(attack.original_margin)} → {_fmt(attack.adversarial_margin)}</td>"
            f"<td>{_fmt(attack.margin_delta)}</td>"
            f"<td>{_fmt(attack.mechanism_survival)}</td>"
            f"<td>{_fmt(attack.perturbation_norm_l2)} / {_fmt(attack.perturbation_norm_linf)}</td>"
            "</tr>"
        )
    recs = "".join(f"<li>{escape(item)}</li>" for item in report.recommendations)
    limits = "".join(f"<li>{escape(item)}</li>" for item in report.limitations)
    vulns = "".join(f"<li>{escape(item)}</li>" for item in report.vulnerable_concepts)
    meta = report.metadata or {}
    suite = ", ".join(_ATTACK_LABELS.get(a, a) for a in meta.get("attack_suite", []))
    robust = meta.get("robust_accuracy")
    robust_block = ""
    if isinstance(robust, dict):
        interval = robust.get("robust_accuracy_wilson_95")
        interval_text = (
            f", Wilson 95% CI=[{_fmt(interval[0])}, {_fmt(interval[1])}]"
            if isinstance(interval, (list, tuple)) and len(interval) == 2
            else ""
        )
        robust_block = (
            f"<strong>Batch robust accuracy:</strong> "
            f"{escape(str(robust.get('robust_accuracy', '—')))} "
            f"(clean={escape(str(robust.get('clean_accuracy', '—')))}, "
            f"n={escape(str(robust.get('n_samples', '—')))}, "
            f"ε={escape(str(robust.get('epsilon', '—')))}, "
            f"protocol={escape(str(robust.get('protocol', '—')))}{interval_text})<br/>"
        )
    aa_flag = meta.get("autoattack_robust")
    aa_block = ""
    if aa_flag is not None:
        aa_block = (
            f"<strong>VRAVEN ensemble-lite @ ε="
            f"{escape(str(meta.get('autoattack_epsilon', '—')))}:</strong> "
            f"{'no flip found on this sample' if aa_flag else 'FLIPPED by ' + escape(str(meta.get('autoattack_flipped_by')))}"
            f"<br/>"
        )
    formal = meta.get("formal_robustness")
    formal_block = ""
    if isinstance(formal, dict):
        formal_block = (
            "<strong>Formal certificate:</strong> proven focus-sample L∞ radius "
            f"≥{_fmt(formal.get('certified_radius_lower_bound'))}; "
            "<a href='formal_robustness_certificate.html' style='color:var(--accent)'>"
            "open proof ledger</a><br/>"
        )
    formal_mechanism = meta.get("formal_causal_mechanism")
    formal_mechanism_block = ""
    if isinstance(formal_mechanism, dict):
        formal_mechanism_block = (
            "<strong>Formal causal-mechanism certificate:</strong> joint decision, "
            "representation and intervention-effect radius "
            f"≥{_fmt(formal_mechanism.get('joint_radius_lower_bound'))}; "
            "<a href='formal_causal_mechanism_certificate.html' "
            "style='color:var(--accent)'>open mechanism proof ledger</a><br/>"
        )

    curve = meta.get("mechanism_survival_curve") or []
    curve_rows = []
    for row in curve:
        curve_rows.append(
            "<tr>"
            f"<td>{escape(_ATTACK_LABELS.get(row.get('family', ''), str(row.get('family', ''))))}</td>"
            f"<td>{_fmt(row.get('epsilon'))}</td>"
            f"<td>{_fmt(row.get('margin'))}</td>"
            f"<td>{_fmt(row.get('mechanism_survival'))}</td>"
            f"<td>{'FLIP' if row.get('flipped') else 'hold'}</td>"
            "</tr>"
        )

    traces = []
    for attack in report.attacks_tested:
        if not attack.succeeded:
            continue
        if attack.mechanism_survival is None and not attack.mechanism_suppressed:
            continue
        traces.append(
            "<div class='trace'>"
            f"<div class='trace-h'>{escape(attack.attack_name)} · "
            f"{escape(str(attack.original_prediction))} → "
            f"{escape(str(attack.adversarial_prediction))}</div>"
            f"<div>S(ε)={_fmt(attack.mechanism_survival)} · "
            f"largest coordinate divergence: "
            f"<span class='amber'>{escape(str(attack.first_mechanism_divergence or '—'))}</span></div>"
            f"<div><span class='blue'>suppressed:</span> "
            f"{escape(', '.join(attack.mechanism_suppressed[:4]) or '—')}</div>"
            f"<div><span class='magenta'>activated:</span> "
            f"{escape(', '.join(attack.mechanism_activated[:4]) or '—')}</div>"
            f"<div><span class='muted'>input direction:</span> "
            f"{escape(', '.join(attack.vulnerable_concepts[:2]) or '—')}</div>"
            "</div>"
        )
        if len(traces) >= 8:
            break

    agreement = meta.get("cross_attack_agreement") or {}
    agree_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{_fmt(v)}</td></tr>"
        for k, v in sorted(agreement.items(), key=lambda kv: -float(kv[1]))
    )
    fail_fams = meta.get("failure_families") or []
    fail_items = "".join(
        "<li><strong>"
        + escape(str(ff.get("family_id", "?")))
        + "</strong>: "
        + escape(", ".join(_ATTACK_LABELS.get(m, m) for m in ff.get("members", [])))
        + "</li>"
        for ff in fail_fams
    )
    instab = report.mechanism_instability_radius
    if instab is None:
        instab = meta.get("mechanism_instability_radius")
    tau = meta.get("mechanism_instability_tau", 0.7)
    probe_kind = escape(str(meta.get("mechanism_probe_kind", "activation_fingerprint")))
    adequacy = meta.get("attack_adequacy") or {}
    adequacy_warnings = adequacy.get("warnings") or []
    adequacy_html = "".join(f"<li>{escape(str(item))}</li>" for item in adequacy_warnings)
    if not adequacy_html:
        adequacy_html = (
            "<li>No implemented diagnostic triggered. This does not establish attack "
            "completeness or model security.</li>"
        )
    spectrum = meta.get("failure_mode_spectrum") or {}

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root {{
  --bg:#030A12; --panel:#07111F; --line:#1C344D; --text:#F1F5F9;
  --muted:#94A3B8; --accent:#00D4FF; --ok:#32D74B; --warn:#FF8A00;
  --blue:#168BFF; --magenta:#E83E8C; --amber:#FFD60A; --cyan:#00D4FF; --danger:#FF2D55;
}}
html[data-theme="light"] {{ --bg:#FFFFFF; --panel:#F8FAFC; --line:#CBD5E1; --text:#0F172A; --muted:#475569; --accent:#0369A1; --ok:#047857; --warn:#92400E; --blue:#1D4ED8; --magenta:#A21CAF; --amber:#92400E; --cyan:#0E7490; --danger:#BE123C; }}
* {{ box-sizing:border-box; }}
body {{
  margin:0; padding:28px 32px 48px; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--text); background:var(--bg);
  background-image:radial-gradient(ellipse 70% 40% at 50% -10%, rgba(0,212,255,0.14), transparent 70%);
}}
h1 {{ margin:0 0 6px; font-size:22px; letter-spacing:-0.02em; }}
.sub {{ color:var(--muted); font-size:13px; margin:0 0 18px; max-width:82ch; line-height:1.45; }}
.banner {{
  border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:var(--panel);
  margin-bottom:18px; font-size:13px; line-height:1.5;
}}
.banner strong {{ color:var(--accent); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:18px; }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; background:var(--panel); }}
.card .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }}
.card .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:16px; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; margin:8px 0 18px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }}
td.flip {{ color:var(--danger); font-weight:800; }}
td.hold {{ color:var(--muted); }}
h2 {{ font-size:14px; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); margin:20px 0 8px; }}
ul {{ margin:0 0 12px; padding-left:18px; color:var(--text); font-size:13px; line-height:1.5; }}
li {{ margin:4px 0; }}
.foot {{ color:var(--muted); font-size:11px; margin-top:20px; }}
.trace {{
  border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:0 0 10px;
  background:rgba(10,16,30,0.85); font-size:12px; line-height:1.55;
}}
.trace-h {{ font-weight:700; margin-bottom:4px; color:var(--cyan); }}
.blue {{ color:var(--blue); }} .magenta {{ color:var(--magenta); }}
.amber {{ color:var(--amber); }} .muted {{ color:var(--muted); }}
.cyan {{ color:var(--cyan); }}
.legend span {{ margin-right:14px; font-size:12px; }}
.toolbar {{ position:fixed; right:20px; top:18px; z-index:3; padding:5px; border:1px solid var(--line); border-radius:9px; background:var(--panel); }}
.toolbar button {{ border:0; border-radius:6px; padding:7px 9px; background:transparent; color:var(--muted); cursor:pointer; }}
.toolbar button.active {{ background:var(--text); color:var(--bg); }}
</style>
<script>function theme(t){{document.documentElement.dataset.theme=t;document.querySelectorAll('.toolbar button').forEach(b=>b.classList.toggle('active',b.dataset.t===t))}}</script>
</head>
<body>
<div class="toolbar"><button data-t="dark" class="active" onclick="theme('dark')">Dark</button><button data-t="light" onclick="theme('light')">Publication light</button></div>
<h1>VRAVEN Adversarial Mechanism Audit</h1>
<p class="sub">
How easily can <strong>{escape(report.model_name)}</strong> be destabilised under controlled
perturbation — and <em>which internal mechanism failed</em> when an attack succeeded?
Layer&nbsp;1 = robustness stress · Layer&nbsp;2 = frozen-M ΔM analysis.
Empirical flip radii are <strong>not</strong> certificates.
</p>
<div class="banner">
  <strong>Suite:</strong> {escape(suite or "default")}<br/>
  <strong>Mechanism probe:</strong> {probe_kind}<br/>
  <strong>Claim type:</strong> {escape(str(meta.get("claim_type", "measured attack success + ΔM")))}<br/>
  {aa_block}{robust_block}{formal_block}{formal_mechanism_block}
  <strong>Access model:</strong> white-box = artefact gradients; flat-block search = black-box.
  Mathematical stress ≠ domain-valid evasion.<br/>
  Decision margin: {escape(str(meta.get("decision_margin_definition", "m = z_target − logsumexp(z_other)")))}
</div>
<div class="grid">
  <div class="card"><div class="k">Smallest successful tested L∞ budget</div><div class="v">{_fmt(report.empirical_min_flip_eps_linf)}</div></div>
  <div class="card"><div class="k">Observed pre-flip instability budget</div><div class="v">{_fmt(None if instab is None else float(instab))}</div></div>
  <div class="card"><div class="k">Instability τ (survival)</div><div class="v">{_fmt(float(tau))}</div></div>
  <div class="card"><div class="k">Most sensitive input coordinate</div><div class="v" style="font-size:13px">{escape(report.most_vulnerable_concept)}</div></div>
</div>
<p class="legend">
  <span class="cyan">● shared / survival</span>
  <span class="blue">● suppressed (original-strong)</span>
  <span class="magenta">● activated (adv-strong)</span>
  <span class="amber">● largest coordinate divergence</span>
</p>
<h2>Layer 2 · Mechanism survival vs margin</h2>
<p class="sub" style="margin-top:0">S(ε) is the conservative minimum of coordinate-aligned cosine, retained activation energy, top-active cosine and worst-layer cosine. Components remain available separately; a low value before a label flip is observed pre-flip instability, not a causal certificate.</p>
<table>
<thead><tr><th>Family</th><th>ε</th><th>Margin</th><th>S(ε)</th><th>Label</th></tr></thead>
<tbody>
{"".join(curve_rows) or '<tr><td colspan="5">No survival curve (probe unavailable)</td></tr>'}
</tbody>
</table>
<h2>Layer 2 · Adversarial Mechanism Trace</h2>
{"".join(traces) or '<p class="sub">No successful flips with ΔM captured on this sample.</p>'}
<h2>Layer 2 · Cross-attack ΔM agreement</h2>
<table>
<thead><tr><th>Pair</th><th>sim(ΔMᵢ, ΔMⱼ)</th></tr></thead>
<tbody>
{agree_rows or '<tr><td colspan="2">Need ≥2 successful families with ΔM</td></tr>'}
</tbody>
</table>
<h2>Layer 2 · Adversarial Failure Families</h2>
<ul>{fail_items or "<li>Insufficient successful flips to cluster</li>"}</ul>
<div class="grid">
  <div class="card"><div class="k">Effective failure modes</div><div class="v">{_fmt(spectrum.get("effective_failure_modes"))}</div></div>
  <div class="card"><div class="k">Dominant failure energy</div><div class="v">{_fmt(spectrum.get("dominant_mode_energy_fraction"))}</div></div>
  <div class="card"><div class="k">Successful family vectors</div><div class="v">{escape(str(len(spectrum.get("families", []))))}</div></div>
</div>
<h2>Layer 1 · Focus-sample grid outcomes</h2>
<div class="grid">{"".join(family_summary)}</div>
<h2>Layer 1 · Attack trials</h2>
<table>
<thead><tr>
<th>Attack</th><th>Family</th><th>Norm</th><th>ε</th><th>Steps</th><th>Result</th>
<th>Margin</th><th>Δm</th><th>S(ε)</th><th>‖δ‖₂ / ‖δ‖∞</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<h2>Attack-adequacy diagnostics</h2>
<p class="sub" style="margin-top:0">Observed checks can reveal evaluation failure; passing them cannot prove that the attacks are complete.</p>
<ul>{adequacy_html}</ul>
<h2>Findings</h2>
<ul>{recs or "<li>None</li>"}</ul>
<h2>Most sensitive tested input coordinates</h2>
<ul>{vulns or "<li>None ranked</li>"}</ul>
<h2>Limitations</h2>
<ul>{limits}</ul>
<p class="foot">Original prediction {escape(str(meta.get("original_prediction", "—")))} ·
target {escape(str(meta.get("decision_target", "—")))} ·
device {escape(str(meta.get("device", "—")))} ·
seed {escape(str(meta.get("seed", "—")))}</p>
</body>
</html>
"""


def _module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _module_dtype(model: nn.Module) -> torch.dtype:
    for value in (*model.parameters(), *model.buffers()):
        if value.is_floating_point():
            return value.dtype
    return torch.float32


def _ensure_batch(x: Tensor) -> Tensor:
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def _clamp(x: Tensor, clip_min: float | None, clip_max: float | None) -> Tensor:
    if clip_min is None and clip_max is None:
        return x
    lo = float("-inf") if clip_min is None else float(clip_min)
    hi = float("inf") if clip_max is None else float(clip_max)
    return x.clamp(min=lo, max=hi)


def _normalise_logits(logits: Tensor) -> Tensor:
    if logits.ndim == 0:
        return logits.reshape(1, 1)
    if logits.ndim == 1:
        return logits.unsqueeze(0) if logits.numel() > 1 else logits.reshape(1, 1)
    if logits.ndim > 2:
        raise ValueError(
            "adversarial classification audit requires scalar, [B], or [B,C] logits; "
            "structured sequence/segmentation outputs require an explicit task adapter"
        )
    return logits


def _normalise_batched_logits(logits: Tensor, batch_size: int) -> Tensor:
    """Normalise logits while preserving a known input batch dimension."""

    if logits.ndim == 1 and logits.shape[0] == int(batch_size):
        return logits.reshape(int(batch_size), 1)
    result = _normalise_logits(logits)
    if result.shape[0] != int(batch_size):
        raise ValueError("model output does not preserve the requested batch dimension")
    return result


def _decision_state(
    model: nn.Module,
    x: Tensor,
    *,
    target: int,
    output_path: str | Sequence[str | int] | None,
) -> tuple[Tensor, int, float]:
    logits = _normalise_logits(extract_logits(model(x), output_path))
    pred = int(predicted_class(logits)[0].item())
    margin = float(target_margin(logits, target)[0].item())
    return logits, pred, margin


def _untargeted_loss(logits: Tensor, original_pred: int) -> Tensor:
    logits = _normalise_logits(logits)
    if logits.shape[-1] == 1:
        logit = logits[:, 0]
        return -logit if original_pred == 1 else logit
    return -logits[:, int(original_pred)]


def _cross_entropy_loss(logits: Tensor, original_pred: int) -> Tensor:
    """Cross-entropy against the clean class, returned for gradient ascent."""

    logits = _normalise_logits(logits)
    if logits.shape[-1] == 1:
        return _untargeted_loss(logits, original_pred)
    y = int(original_pred)
    return torch.logsumexp(logits, dim=1) - logits[:, y]


def _cw_logit_loss(logits: Tensor, original_pred: int) -> Tensor:
    """CW margin loss: ascend on max_{i≠y} z_i − z_y."""

    logits = _normalise_logits(logits)
    if logits.shape[-1] == 1:
        return _untargeted_loss(logits, original_pred)
    y = int(original_pred)
    z_y = logits[:, y]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, y] = False
    z_other = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return z_other - z_y


def _dlr_loss(logits: Tensor, original_pred: int) -> Tensor:
    """Untargeted difference-of-logits-ratio objective from AutoAttack."""

    logits = _normalise_logits(logits)
    if logits.shape[-1] < 3:
        return _cw_logit_loss(logits, original_pred)
    sorted_logits, sorted_indices = torch.sort(logits, dim=1)
    y = int(original_pred)
    z_y = logits[:, y]
    top_is_y = sorted_indices[:, -1] == y
    strongest_other = torch.where(top_is_y, sorted_logits[:, -2], sorted_logits[:, -1])
    denominator = (sorted_logits[:, -1] - sorted_logits[:, -3]).clamp_min(1e-12)
    return -(z_y - strongest_other) / denominator


def _targeted_logit_loss(logits: Tensor, original_pred: int, target_class: int) -> Tensor:
    """Ascend on z_t − z_y to push toward ``target_class``."""

    logits = _normalise_logits(logits)
    if logits.shape[-1] == 1:
        return _untargeted_loss(logits, original_pred)
    return logits[:, int(target_class)] - logits[:, int(original_pred)]


def _targeted_dlr_loss(logits: Tensor, original_pred: int, target_class: int) -> Tensor:
    """Targeted DLR objective; falls back to a logit margin below four classes."""

    logits = _normalise_logits(logits)
    if logits.shape[-1] < 4:
        return _targeted_logit_loss(logits, original_pred, target_class)
    sorted_logits, _ = torch.sort(logits, dim=1)
    numerator = logits[:, int(original_pred)] - logits[:, int(target_class)]
    denominator = (
        sorted_logits[:, -1] - 0.5 * (sorted_logits[:, -3] + sorted_logits[:, -4])
    ).clamp_min(1e-12)
    return -numerator / denominator


def _input_gradient(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    output_path: str | Sequence[str | int] | None,
    loss_kind: LossKind = "cross_entropy",
    target_class: int | None = None,
) -> Tensor:
    x_var = x.detach().clone().requires_grad_(True)
    logits = _normalise_logits(extract_logits(model(x_var), output_path))
    if loss_kind == "cross_entropy":
        loss = _cross_entropy_loss(logits, original_pred).sum()
    elif loss_kind == "cw_logit":
        loss = _cw_logit_loss(logits, original_pred).sum()
    elif loss_kind == "dlr":
        loss = _dlr_loss(logits, original_pred).sum()
    elif loss_kind == "targeted":
        if target_class is None:
            raise ValueError("target_class required for targeted loss")
        loss = _targeted_logit_loss(logits, original_pred, target_class).sum()
    elif loss_kind == "targeted_dlr":
        if target_class is None:
            raise ValueError("target_class required for targeted DLR loss")
        loss = _targeted_dlr_loss(logits, original_pred, target_class).sum()
    else:
        loss = _untargeted_loss(logits, original_pred).sum()
    gradient = torch.autograd.grad(
        loss,
        x_var,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(x)
    return gradient.detach().clone()


def _project_linf(adv: Tensor, x0: Tensor, epsilon: float) -> Tensor:
    """Project with inward-rounded floating-point endpoints.

    Computing ``x0 + epsilon`` in float32 can round away from ``x0`` by more
    than the real-valued budget when coordinates are large relative to
    epsilon.  Moving both endpoints one representable value toward ``x0``
    preserves the declared constraint in the tensor's own arithmetic.
    """

    upper = torch.nextafter(x0 + float(epsilon), x0)
    lower = torch.nextafter(x0 - float(epsilon), x0)
    return torch.maximum(torch.minimum(adv, upper), lower)


def _project_l2(adv: Tensor, x0: Tensor, epsilon: float) -> Tensor:
    delta = adv - x0
    flat = delta.reshape(delta.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(1e-12)
    scale = torch.clamp(float(epsilon) / norms, max=1.0).reshape(-1, *([1] * (delta.ndim - 1)))
    return x0 + delta * scale


def _fgsm_step(
    x: Tensor,
    grad: Tensor,
    epsilon: float,
    *,
    clip_min: float | None,
    clip_max: float | None,
) -> Tensor:
    return _clamp(x + float(epsilon) * grad.sign(), clip_min, clip_max)


def _iterative_linf(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    epsilon: float,
    steps: int,
    step_size: float,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    initial: Tensor | None = None,
    momentum: float = 0.0,
    loss_kind: LossKind = "cross_entropy",
) -> Tensor:
    """BIM / MI-FGSM / PGD-L∞ core."""

    x0 = x.detach()
    adv = _clamp(initial.detach() if initial is not None else x0.clone(), clip_min, clip_max)
    best_adv = adv.detach().clone()
    with torch.inference_mode():
        best_loss = _eval_loss(
            _normalise_logits(extract_logits(model(best_adv), output_path)),
            original_pred=original_pred,
            loss_kind=loss_kind,
            target_class=None,
        )
    g_mom = torch.zeros_like(x0)
    for _ in range(max(1, int(steps))):
        grad = _input_gradient(
            model,
            adv,
            original_pred=original_pred,
            output_path=output_path,
            loss_kind=loss_kind,
        )
        if momentum > 0:
            grad_flat = grad.reshape(grad.shape[0], -1)
            grad_norm = torch.linalg.vector_norm(grad_flat, ord=1, dim=1).clamp_min(1e-12)
            grad = grad / grad_norm.reshape(-1, *([1] * (grad.ndim - 1)))
            g_mom = momentum * g_mom + grad
            update = g_mom.sign()
        else:
            update = grad.sign()
        adv = adv.detach() + float(step_size) * update
        adv = _project_linf(adv, x0, epsilon)
        adv = _clamp(adv, clip_min, clip_max)
        with torch.inference_mode():
            loss_value = _eval_loss(
                _normalise_logits(extract_logits(model(adv), output_path)),
                original_pred=original_pred,
                loss_kind=loss_kind,
                target_class=None,
            )
        if loss_value > best_loss:
            best_adv = adv.detach().clone()
            best_loss = loss_value
    return best_adv.detach()


def _pgd_l2(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    epsilon: float,
    steps: int,
    step_size: float,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    initial: Tensor | None = None,
) -> Tensor:
    x0 = x.detach()
    if initial is None:
        noise = torch.randn_like(x0)
        noise = noise / torch.linalg.vector_norm(noise.reshape(1, -1), ord=2).clamp_min(1e-12)
        adv = _project_l2(x0 + noise * epsilon, x0, epsilon)
    else:
        adv = _project_l2(initial.detach(), x0, epsilon)
    adv = _clamp(adv, clip_min, clip_max)
    best_adv = adv.detach().clone()
    with torch.inference_mode():
        best_loss = _eval_loss(
            _normalise_logits(extract_logits(model(best_adv), output_path)),
            original_pred=original_pred,
            loss_kind="cross_entropy",
            target_class=None,
        )
    for _ in range(max(1, int(steps))):
        grad = _input_gradient(model, adv, original_pred=original_pred, output_path=output_path)
        g_flat = grad.reshape(1, -1)
        g_norm = torch.linalg.vector_norm(g_flat, ord=2).clamp_min(1e-12)
        adv = adv.detach() + float(step_size) * (
            grad / g_norm.reshape(-1, *([1] * (grad.ndim - 1)))
        )
        adv = _project_l2(adv, x0, epsilon)
        adv = _clamp(adv, clip_min, clip_max)
        with torch.inference_mode():
            loss_value = _eval_loss(
                _normalise_logits(extract_logits(model(adv), output_path)),
                original_pred=original_pred,
                loss_kind="cross_entropy",
                target_class=None,
            )
        if loss_value > best_loss:
            best_adv = adv.detach().clone()
            best_loss = loss_value
    return best_adv.detach()


def _batch_pgd_linf(
    model: nn.Module,
    x: Tensor,
    targets: Tensor,
    *,
    epsilon: float,
    steps: int,
    restarts: int,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    seed: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Batched multi-restart PGD with per-sample best-path retention.

    The optimiser ascends cross-entropy while candidate selection minimises the
    declared true-class margin.  Every retained point is projected back into
    the same L∞ ball and optional input-domain box after each update.
    """

    if x.ndim < 2 or x.shape[0] < 1:
        raise ValueError("batched PGD requires a non-empty input batch")
    if targets.reshape(-1).numel() != x.shape[0]:
        raise ValueError("batched PGD requires one target per input")
    if int(steps) < 1 or int(restarts) < 1:
        raise ValueError("batched PGD steps and restarts must be positive")

    x0 = x.detach()
    targets = targets.to(device=x0.device, dtype=torch.long).reshape(-1)
    with torch.inference_mode():
        clean_logits = _normalise_batched_logits(
            extract_logits(model(x0), output_path), x0.shape[0]
        )
        if not bool(torch.isfinite(clean_logits).all()):
            raise ValueError("batched PGD received non-finite clean logits")
        best_margin = target_margin(clean_logits, targets).detach()
    best_adv = x0.clone()
    alpha = 2.0 * float(epsilon) / max(int(steps), 1)
    nonfinite_gradient_sample_steps = 0
    zero_gradient_sample_steps = 0
    gradient_sample_steps = 0

    for restart in range(int(restarts)):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed) + 10_007 * restart)
        noise = (
            torch.empty(x0.shape, dtype=torch.float32)
            .uniform_(-float(epsilon), float(epsilon), generator=rng)
            .to(device=x0.device, dtype=x0.dtype)
        )
        adv = _clamp(_project_linf(x0 + noise, x0, epsilon), clip_min, clip_max)
        with torch.inference_mode():
            initial_logits = _normalise_batched_logits(
                extract_logits(model(adv), output_path), x0.shape[0]
            )
            if not bool(torch.isfinite(initial_logits).all()):
                raise ValueError("batched PGD received non-finite random-start logits")
            initial_margins = target_margin(initial_logits, targets)
        initial_improved = initial_margins < best_margin
        if bool(initial_improved.any()):
            mask = initial_improved.reshape(-1, *([1] * (adv.ndim - 1)))
            best_adv = torch.where(mask, adv.detach(), best_adv)
            best_margin = torch.where(initial_improved, initial_margins.detach(), best_margin)
        for _ in range(int(steps)):
            adv_var = adv.detach().clone().requires_grad_(True)
            logits = _normalise_batched_logits(
                extract_logits(model(adv_var), output_path), x0.shape[0]
            )
            if logits.shape[-1] == 1:
                loss = -target_margin(logits, targets).sum()
            else:
                target_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                loss = (torch.logsumexp(logits, dim=1) - target_scores).sum()
            gradient = torch.autograd.grad(
                loss,
                adv_var,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(adv_var)
            gradient_flat = gradient.detach().reshape(gradient.shape[0], -1)
            finite_by_sample = torch.isfinite(gradient_flat).all(dim=1)
            nonfinite_gradient_sample_steps += int((~finite_by_sample).sum().item())
            gradient = torch.where(torch.isfinite(gradient), gradient, torch.zeros_like(gradient))
            zero_by_sample = gradient.reshape(gradient.shape[0], -1).abs().amax(dim=1) <= 1e-12
            zero_gradient_sample_steps += int(zero_by_sample.sum().item())
            gradient_sample_steps += int(gradient.shape[0])
            adv = adv_var.detach() + alpha * gradient.detach().sign()
            adv = _clamp(_project_linf(adv, x0, epsilon), clip_min, clip_max)
            with torch.inference_mode():
                margins = target_margin(
                    _normalise_batched_logits(extract_logits(model(adv), output_path), x0.shape[0]),
                    targets,
                )
                if not bool(torch.isfinite(margins).all()):
                    raise ValueError("batched PGD received non-finite candidate margins")
            improved = margins < best_margin
            if bool(improved.any()):
                mask = improved.reshape(-1, *([1] * (adv.ndim - 1)))
                best_adv = torch.where(mask, adv.detach(), best_adv)
                best_margin = torch.where(improved, margins.detach(), best_margin)

    best_adv_cpu = best_adv.detach().cpu()
    x0_cpu = x0.detach().cpu()
    observed_linf = torch.linalg.vector_norm(
        (best_adv_cpu.to(torch.float64) - x0_cpu.to(torch.float64)).reshape(x0.shape[0], -1),
        ord=float("inf"),
        dim=1,
    )
    max_linf = float(observed_linf.max().item())
    domain_ok = True
    if clip_min is not None:
        domain_ok = domain_ok and bool((best_adv >= float(clip_min) - 1e-6).all())
    if clip_max is not None:
        domain_ok = domain_ok and bool((best_adv <= float(clip_max) + 1e-6).all())
    return best_adv.detach(), {
        "vectorised": True,
        "best_path_retained": True,
        "random_start_candidates_assessed": True,
        "steps": int(steps),
        "restarts": int(restarts),
        "gradient_sample_steps": gradient_sample_steps,
        "nonfinite_gradient_sample_steps": nonfinite_gradient_sample_steps,
        "zero_gradient_sample_steps": zero_gradient_sample_steps,
        "finite_gradient_fraction": (
            1.0 - nonfinite_gradient_sample_steps / gradient_sample_steps
            if gradient_sample_steps
            else 0.0
        ),
        "zero_gradient_fraction": (
            zero_gradient_sample_steps / gradient_sample_steps if gradient_sample_steps else 1.0
        ),
        "maximum_observed_linf": max_linf,
        "linf_constraint_satisfied": max_linf <= float(epsilon) + 1e-6,
        "domain_constraint_satisfied": domain_ok,
    }


def _eval_loss(
    logits: Tensor,
    *,
    original_pred: int,
    loss_kind: LossKind,
    target_class: int | None,
) -> float:
    if loss_kind == "cross_entropy":
        return float(_cross_entropy_loss(logits, original_pred).item())
    if loss_kind == "cw_logit":
        return float(_cw_logit_loss(logits, original_pred).item())
    if loss_kind == "dlr":
        return float(_dlr_loss(logits, original_pred).item())
    if loss_kind == "targeted":
        assert target_class is not None
        return float(_targeted_logit_loss(logits, original_pred, target_class).item())
    if loss_kind == "targeted_dlr":
        assert target_class is not None
        return float(_targeted_dlr_loss(logits, original_pred, target_class).item())
    return float(_untargeted_loss(logits, original_pred).item())


def _apgd_lite(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    epsilon: float,
    steps: int,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    seed: int,
    loss_kind: LossKind = "cross_entropy",
    target_class: int | None = None,
    n_restarts: int = 2,
) -> Tensor:
    """Simplified Auto-PGD with adaptive step size (Croce & Hein style, lite)."""

    x0 = x.detach()
    rng = torch.Generator(device="cpu")
    best_adv = x0.clone()
    best_loss = float("-inf")
    restarts = max(1, int(n_restarts))

    for restart in range(restarts):
        rng.manual_seed(int(seed) + 17 * restart)
        noise = (
            torch.empty(x0.shape, dtype=torch.float32)
            .uniform_(-epsilon, epsilon, generator=rng)
            .to(device=x0.device)
        )
        adv = _clamp(_project_linf(x0 + noise, x0, epsilon), clip_min, clip_max)
        step = 2.0 * float(epsilon) / max(int(steps), 1)
        prev_loss = float("-inf")
        stagnant = 0
        for _ in range(max(1, int(steps))):
            grad = _input_gradient(
                model,
                adv,
                original_pred=original_pred,
                output_path=output_path,
                loss_kind=loss_kind,
                target_class=target_class,
            )
            adv = adv.detach() + step * grad.sign()
            adv = _clamp(_project_linf(adv, x0, epsilon), clip_min, clip_max)
            with torch.inference_mode():
                logits = _normalise_logits(extract_logits(model(adv), output_path))
                loss_val = _eval_loss(
                    logits,
                    original_pred=original_pred,
                    loss_kind=loss_kind,
                    target_class=target_class,
                )
                pred = int(predicted_class(logits)[0].item())
            if loss_val > best_loss:
                best_loss = loss_val
                best_adv = adv.detach()
            if pred != original_pred:
                return adv.detach()
            if loss_val <= prev_loss + 1e-8:
                stagnant += 1
            else:
                stagnant = 0
            prev_loss = loss_val
            if stagnant >= 3:
                step = max(step * 0.5, float(epsilon) / (4.0 * max(int(steps), 1)))
                stagnant = 0
    return best_adv.detach()


def _logit_grads(
    model: nn.Module,
    x: Tensor,
    *,
    output_path: str | Sequence[str | int] | None,
) -> tuple[Tensor, Tensor]:
    """Return (logits[1,C], grads[C,*x.shape]) for each output logit."""

    x_var = x.detach().clone().requires_grad_(True)
    logits = _normalise_logits(extract_logits(model(x_var), output_path))
    n_classes = int(logits.shape[-1])
    grads = []
    for k in range(n_classes):
        gradient = torch.autograd.grad(
            logits[0, k],
            x_var,
            retain_graph=k < n_classes - 1,
            create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is None:
            grads.append(torch.zeros_like(x))
        else:
            grads.append(gradient.detach().clone())
    return logits.detach(), torch.stack(grads, dim=0)


def _fab_linf(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    epsilon: float,
    n_iter: int,
    n_restarts: int,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    seed: int,
    eta: float = 1.05,
    beta: float = 0.9,
    targeted_class: int | None = None,
) -> Tensor:
    """FAB-lite L∞ boundary attack (Croce & Hein 2019 inspired, not bit-identical)."""

    x0 = x.detach()
    best_adv = x0.clone()
    best_linf = float("inf")
    rng = torch.Generator(device="cpu")

    for restart in range(max(1, int(n_restarts))):
        rng.manual_seed(int(seed) + 31 * restart)
        if restart == 0:
            adv = x0.clone()
        else:
            noise = (
                torch.empty(x0.shape, dtype=torch.float32)
                .uniform_(-epsilon, epsilon, generator=rng)
                .to(device=x0.device)
            )
            adv = _clamp(_project_linf(x0 + 0.5 * noise, x0, epsilon), clip_min, clip_max)

        for _ in range(max(1, int(n_iter))):
            logits, grads = _logit_grads(model, adv, output_path=output_path)
            n_classes = int(logits.shape[-1])
            y = int(original_pred)

            if n_classes == 1:
                z = float(logits[0, 0].item())
                w = grads[0]
                w1 = float(w.reshape(-1).abs().sum().item()) + 1e-12
                dist = abs(z) / w1
                if y == 1 and z <= 0:
                    break
                if y == 0 and z >= 0:
                    break
                # Class-1 ⇒ decrease logit; class-0 ⇒ increase.
                step = (-1.0 if y == 1 else 1.0) * float(eta) * dist * w.sign()
                cand = adv.detach() + step
            else:
                candidates: list[int]
                if targeted_class is not None and 0 <= int(targeted_class) < n_classes:
                    candidates = [int(targeted_class)]
                else:
                    candidates = [k for k in range(n_classes) if k != y]
                best_k = None
                best_dist = float("inf")
                best_w = None
                best_df = 0.0
                for k in candidates:
                    df = float((logits[0, k] - logits[0, y]).item())
                    w = grads[k] - grads[y]
                    w1 = float(w.reshape(-1).abs().sum().item()) + 1e-12
                    dist = abs(df) / w1
                    if dist < best_dist:
                        best_dist = dist
                        best_k = k
                        best_w = w
                        best_df = df
                if best_k is None or best_w is None:
                    break
                # Move to increase z_k - z_y (cross boundary from negative df).
                step = float(eta) * best_dist * best_w.sign()
                if best_df > 0:
                    # Already on wrong side of linearisation; small exploratory step.
                    step = float(eta) * (best_dist + 1e-4) * best_w.sign()
                cand = adv.detach() + step

            cand = _clamp(_project_linf(cand, x0, epsilon), clip_min, clip_max)
            with torch.inference_mode():
                pred = int(
                    predicted_class(_normalise_logits(extract_logits(model(cand), output_path)))[
                        0
                    ].item()
                )
            if pred != original_pred:
                linf = float((cand - x0).reshape(-1).abs().max().item())
                if linf < best_linf:
                    best_linf = linf
                    best_adv = cand.detach()
                # FAB backward step toward clean sample.
                adv = x0 + float(beta) * (cand - x0)
                adv = _clamp(_project_linf(adv, x0, epsilon), clip_min, clip_max)
            else:
                adv = cand.detach()

    return best_adv.detach()


def _deepfool(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    max_iter: int,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    overshoot: float = 0.02,
) -> Tensor:
    """Minimal L2 DeepFool-style iterative linearisation (untargeted)."""

    adv = x.detach().clone()
    for _ in range(max(1, int(max_iter))):
        with torch.inference_mode():
            logits = _normalise_logits(extract_logits(model(adv), output_path))
            pred = int(predicted_class(logits)[0].item())
        if pred != original_pred:
            break
        if logits.shape[-1] == 1:
            # Binary: move toward opposite side of 0.
            grad = _input_gradient(model, adv, original_pred=original_pred, output_path=output_path)
            g_norm = torch.linalg.vector_norm(grad.reshape(-1), ord=2).clamp_min(1e-12)
            # Distance to boundary ≈ |logit| / ‖∇logit‖
            with torch.inference_mode():
                z = float(logits[0, 0].item())
            # For class 1, boundary when z=0; step -sign(z)*|z|/‖g‖ along grad of z
            # Use untargeted grad direction.
            step = (abs(z) + 1e-6) / float(g_norm.item())
            adv = adv.detach() + (1.0 + overshoot) * step * grad / g_norm
            adv = _clamp(adv, clip_min, clip_max)
            continue

        # Multiclass: closest hyperplane among alternatives.
        best_w = None
        best_step = None
        for k in range(int(logits.shape[-1])):
            if k == original_pred:
                continue
            # Approximate ∇(z_k - z_y) via finite difference on loss surrogate.
            # Use analytic grad of (z_k - z_y) by custom backward.
            x_var = adv.detach().clone().requires_grad_(True)
            z = _normalise_logits(extract_logits(model(x_var), output_path))
            diff = z[0, k] - z[0, original_pred]
            model.zero_grad(set_to_none=True)
            diff.backward()
            if x_var.grad is None:
                continue
            w = x_var.grad.detach()
            w_norm = torch.linalg.vector_norm(w.reshape(-1), ord=2).clamp_min(1e-12)
            f_val = float(diff.detach().item())
            # Need f + ⟨w, r⟩ = 0 with f = z_k - z_y < 0 typically; step = |f|/‖w‖ toward +
            step = (abs(f_val) + 1e-6) / float(w_norm.item())
            if best_step is None or step < best_step:
                best_step = step
                best_w = w / w_norm
        if best_w is None or best_step is None:
            break
        adv = adv.detach() + (1.0 + overshoot) * float(best_step) * best_w
        adv = _clamp(adv, clip_min, clip_max)
    return adv.detach()


def _square_attack(
    model: nn.Module,
    x: Tensor,
    *,
    original_pred: int,
    epsilon: float,
    n_queries: int,
    clip_min: float | None,
    clip_max: float | None,
    output_path: str | Sequence[str | int] | None,
    seed: int,
) -> Tensor:
    """Black-box Square-style attack: random contiguous patches at ±ε (flat-safe)."""

    x0 = x.detach()
    adv = x0.clone()
    flat_n = int(x0.numel())
    rng = np.random.default_rng(int(seed))
    with torch.inference_mode():
        logits = _normalise_logits(extract_logits(model(adv), output_path))
        best_loss = float(_untargeted_loss(logits, original_pred).item())
        best_adv = adv.clone()
        if int(predicted_class(logits)[0].item()) != original_pred:
            return adv

    for q in range(max(1, int(n_queries))):
        # Shrinking patch size schedule.
        p = max(1, int(flat_n * (0.1 * (1.0 - q / max(n_queries, 1)) + 0.01)))
        start = int(rng.integers(0, max(flat_n - p + 1, 1)))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        cand = best_adv.clone().reshape(-1)
        cand[start : start + p] = x0.reshape(-1)[start : start + p] + sign * float(epsilon)
        cand = _project_linf(cand.reshape_as(x0), x0, epsilon)
        cand = _clamp(cand, clip_min, clip_max)
        with torch.inference_mode():
            logits = _normalise_logits(extract_logits(model(cand), output_path))
            loss_val = float(_untargeted_loss(logits, original_pred).item())
            pred = int(predicted_class(logits)[0].item())
        if loss_val > best_loss or pred != original_pred:
            best_loss = loss_val
            best_adv = cand.detach()
            if pred != original_pred:
                break
    return best_adv.detach()


def _perturbation_norms(delta: Tensor) -> tuple[float, float]:
    flat = delta.reshape(delta.shape[0], -1)
    l2 = float(torch.norm(flat, p=2, dim=1)[0].item())
    linf = float(torch.norm(flat, p=float("inf"), dim=1)[0].item())
    return l2, linf


def _rank_input_features(
    grad: Tensor,
    *,
    concept_names: Sequence[str] | None,
    top_k: int = 5,
) -> list[str]:
    flat = grad.detach().reshape(-1).abs()
    k = min(int(top_k), int(flat.numel()))
    if k <= 0:
        return []
    values, indices = torch.topk(flat, k=k)
    names: list[str] = []
    for idx, value in zip(indices.tolist(), values.tolist(), strict=True):
        if concept_names is not None and idx < len(concept_names):
            label = str(concept_names[idx])
        else:
            label = f"feature_{idx}"
        names.append(f"{label} ( |∇x|={float(value):.4g} )")
    return names


def _rank_concept_directions(
    delta: Tensor,
    directions: Tensor,
    *,
    concept_names: Sequence[str] | None,
    top_k: int = 5,
) -> list[str]:
    delta_flat = delta.detach().reshape(-1).float()
    dirs = directions.detach().float()
    if dirs.ndim == 1:
        dirs = dirs.unsqueeze(0)
    dirs = dirs.reshape(dirs.shape[0], -1)
    if dirs.shape[1] != delta_flat.numel():
        raise ValueError(
            f"concept_directions flat width {dirs.shape[1]} must match "
            f"input width {delta_flat.numel()}"
        )
    norms = torch.linalg.vector_norm(dirs, ord=2, dim=1).clamp_min(1e-12)
    scores = (dirs @ delta_flat).abs() / norms
    k = min(int(top_k), int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    names: list[str] = []
    for idx, value in zip(indices.tolist(), values.tolist(), strict=True):
        if concept_names is not None and idx < len(concept_names):
            label = str(concept_names[idx])
        else:
            label = f"concept_{idx}"
        names.append(f"{label} ( |⟨δ,d⟩|/‖d‖={float(value):.4g} )")
    return names


class AdversarialAuditor:
    """Audit a live PyTorch model with a measured adversarial stress suite."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_name: str | None = None,
        output_path: str | Sequence[str | int] | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name or type(model).__name__
        self.output_path = output_path
        self.device = _module_device(model)
        self.dtype = _module_dtype(model)

    def audit_sample(
        self,
        x: Tensor,
        y: int | Tensor | None = None,
        *,
        target: int | None = None,
        epsilon_grid: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5),
        pgd_steps: int = 10,
        pgd_restarts: int = 1,
        pgd_step_size: float | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        concept_directions: Tensor | np.ndarray | None = None,
        concept_names: Sequence[str] | None = None,
        seed: int = 24032026,
        run_pgd: bool = True,
        attacks: Sequence[str] | None = None,
        deepfool_iters: int = 40,
        square_queries: int = 80,
        fab_iters: int = 20,
        fab_restarts: int = 2,
        n_target_classes: int = 3,
        mechanism_probe: MechanismProbe | None = None,
        mechanism_unit_names: Sequence[str] | None = None,
        mechanism_probe_kind: str | None = None,
        instability_tau: float = 0.7,
        instability_metric: str = "mechanism_survival",
        use_concept_directions_as_mechanism: bool = False,
    ) -> AdversarialAuditReport:
        """Run Layer-1 stress suite + Layer-2 frozen-M ΔM analysis."""

        if not epsilon_grid:
            raise ValueError("epsilon_grid must contain at least one ε")
        epsilons = [float(e) for e in epsilon_grid]
        if any(e < 0 or not math.isfinite(e) for e in epsilons):
            raise ValueError("epsilon_grid values must be non-negative and finite")
        if clip_min is not None and not math.isfinite(float(clip_min)):
            raise ValueError("clip_min must be finite when supplied")
        if clip_max is not None and not math.isfinite(float(clip_max)):
            raise ValueError("clip_max must be finite when supplied")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max")
        if not 0.0 <= float(instability_tau) <= 1.0:
            raise ValueError("instability_tau must lie in [0, 1]")
        if int(pgd_steps) < 1 or int(pgd_restarts) < 1 or int(deepfool_iters) < 1:
            raise ValueError("attack iteration budgets must be positive")
        if int(square_queries) < 1 or int(fab_iters) < 1 or int(fab_restarts) < 1:
            raise ValueError("query, FAB iteration and restart budgets must be positive")

        if attacks is None:
            suite = list(DEFAULT_STRESS_SUITE if run_pgd else ("fgsm",))
        else:
            suite = [str(a).lower() for a in attacks]
        suite = ["apgd_ce" if a == "apgd" else a for a in suite]
        unknown = [a for a in suite if a not in _ATTACK_LABELS or a == "autoattack"]
        if unknown:
            raise ValueError(
                f"Unknown attack(s) {unknown}; choose from "
                f"{sorted(k for k in _ATTACK_LABELS if k != 'autoattack')}"
            )

        was_training = self.model.training
        self.model.eval()
        try:
            x_in = _ensure_batch(x.detach()).to(device=self.device, dtype=self.dtype)
            if x_in.shape[0] != 1:
                raise ValueError("audit_sample currently supports batch size 1")
            if clip_min is not None and bool((x_in < float(clip_min) - 1e-7).any()):
                raise ValueError("clean input lies below clip_min")
            if clip_max is not None and bool((x_in > float(clip_max) + 1e-7).any()):
                raise ValueError("clean input lies above clip_max")

            with torch.inference_mode():
                clean_logits = _normalise_logits(extract_logits(self.model(x_in), self.output_path))
                clean_pred = int(predicted_class(clean_logits)[0].item())
            if target is not None:
                decision_target = int(target)
            elif y is not None:
                decision_target = int(y) if isinstance(y, int) else int(y.reshape(-1)[0].item())
            else:
                decision_target = clean_pred

            with torch.inference_mode():
                _, original_pred, original_margin = _decision_state(
                    self.model,
                    x_in,
                    target=decision_target,
                    output_path=self.output_path,
                )

            base_grad = _input_gradient(
                self.model,
                x_in,
                original_pred=original_pred,
                output_path=self.output_path,
            )

            concept_dirs_t: Tensor | None = None
            if concept_directions is not None:
                if isinstance(concept_directions, np.ndarray):
                    concept_dirs_t = torch.from_numpy(
                        np.asarray(concept_directions, dtype=np.float32)
                    )
                else:
                    concept_dirs_t = concept_directions.detach().float()
                concept_dirs_t = concept_dirs_t.to(device=self.device)

            # Layer-2 frozen mechanism probe (never rediscover concepts per trial).
            unit_names: list[str]
            probe_kind: str
            if mechanism_probe is not None:
                probe = mechanism_probe
                unit_names = list(mechanism_unit_names or [])
                probe_kind = str(mechanism_probe_kind or "custom")
            elif use_concept_directions_as_mechanism and concept_dirs_t is not None:
                probe, unit_names, probe_kind = build_concept_probe_from_directions(
                    concept_dirs_t,
                    concept_names=concept_names,
                )
            else:
                probe, unit_names, probe_kind = build_activation_probe(self.model)
            probe_failures: list[str] = []
            try:
                m_clean = probe(x_in)
                unit_names = ensure_unit_names(probe, unit_names, x_in)
            except Exception as error:  # noqa: BLE001
                probe_failures.append(f"clean probe failed: {type(error).__name__}: {error}")
                probe = None  # type: ignore[assignment]
                m_clean = None
                probe_kind = "unavailable"

            attacks_out: list[AttackResult] = []
            min_flip_eps_linf: float | None = None
            min_flip_eps_l2: float | None = None
            best_delta: Tensor | None = None
            best_delta_l2 = float("inf")
            family_deltas: dict[str, Tensor] = {}
            family_delta_l2: dict[str, float] = {}

            def _vulns(delta: Tensor) -> list[str]:
                if concept_dirs_t is not None:
                    return _rank_concept_directions(
                        delta, concept_dirs_t, concept_names=concept_names
                    )[:3]
                return _rank_input_features(base_grad, concept_names=concept_names)[:3]

            def _strongest_candidate(candidates: Sequence[Tensor]) -> Tensor:
                """Return the candidate with the smallest declared-target margin.

                Every restart remains inside the same threat set.  Selecting the
                minimum target margin makes the multi-restart result a per-sample
                attack union: adding restarts cannot make the retained candidate
                weaker under this objective.
                """

                if not candidates:
                    raise ValueError("at least one adversarial candidate is required")
                best = candidates[0]
                best_margin = float("inf")
                with torch.inference_mode():
                    for candidate in candidates:
                        _, _, margin = _decision_state(
                            self.model,
                            candidate,
                            target=decision_target,
                            output_path=self.output_path,
                        )
                        if margin < best_margin:
                            best = candidate
                            best_margin = margin
                return best.detach()

            def _record(
                *,
                family: str,
                name: str,
                adv: Tensor,
                epsilon: float,
                steps: int,
                norm: str,
                track_linf_grid: bool,
                objective: str = "unspecified",
            ) -> None:
                nonlocal min_flip_eps_linf, min_flip_eps_l2, best_delta, best_delta_l2
                if objective == "unspecified":
                    objective = {
                        "fgsm": "cross_entropy",
                        "bim": "cross_entropy",
                        "mifgsm": "cross_entropy",
                        "pgd_linf": "cross_entropy",
                        "pgd_l2": "cross_entropy",
                        "apgd_ce": "cross_entropy",
                        "apgd_dlr": (
                            "difference_of_logits_ratio"
                            if clean_logits.shape[-1] >= 3
                            else "cw_margin_binary_fallback"
                        ),
                        "pgd_dlr": (
                            "difference_of_logits_ratio"
                            if clean_logits.shape[-1] >= 3
                            else "cw_margin_binary_fallback"
                        ),
                        "fab": "minimum_boundary_distance_linearisation",
                        "deepfool": "minimum_l2_boundary_linearisation",
                        "square": "query_margin_random_search",
                    }.get(family, family)
                with torch.inference_mode():
                    _, adv_pred, adv_margin = _decision_state(
                        self.model,
                        adv,
                        target=decision_target,
                        output_path=self.output_path,
                    )
                delta = adv - x_in
                l2, linf = _perturbation_norms(delta)
                flipped = adv_pred != original_pred
                if flipped:
                    if track_linf_grid and (
                        min_flip_eps_linf is None or epsilon < min_flip_eps_linf
                    ):
                        min_flip_eps_linf = float(epsilon)
                    if min_flip_eps_l2 is None or l2 < min_flip_eps_l2:
                        min_flip_eps_l2 = l2
                    if l2 < best_delta_l2:
                        best_delta = delta.detach()
                        best_delta_l2 = l2
                elif best_delta is None:
                    best_delta = delta.detach()

                snap: MechanismSnapshot | None = None
                if probe is not None and m_clean is not None:
                    try:
                        m_adv = probe(adv)
                        snap = compare_mechanisms(
                            m_clean,
                            m_adv,
                            unit_names=unit_names,
                            probe_kind=probe_kind,
                        )
                        d_mech = m_adv.detach().float().reshape(-1) - m_clean.reshape(-1)
                        if flipped:
                            previous_l2 = family_delta_l2.get(family, float("inf"))
                            if l2 < previous_l2:
                                family_deltas[family] = d_mech
                                family_delta_l2[family] = l2
                    except Exception as error:  # noqa: BLE001
                        probe_failures.append(
                            f"{family}@{epsilon:g}: {type(error).__name__}: {error}"
                        )
                        snap = None

                attacks_out.append(
                AttackResult(
                        attack_name=name,
                        attack_family=family,
                        succeeded=flipped,
                        epsilon=float(epsilon),
                        steps=int(steps),
                        prediction_flipped=flipped,
                        perturbation_norm_l2=l2,
                        perturbation_norm_linf=linf,
                        original_margin=original_margin,
                    adversarial_margin=adv_margin,
                        margin_delta=adv_margin - original_margin,
                        original_prediction=original_pred,
                        adversarial_prediction=adv_pred,
                        vulnerable_concepts=_vulns(delta),
                        norm=norm,
                        mechanism_survival=None if snap is None else snap.survival,
                        mechanism_delta_l1=None if snap is None else snap.delta_l1,
                        mechanism_delta_linf=None if snap is None else snap.delta_linf,
                        mechanism_suppressed=[] if snap is None else list(snap.suppressed_units),
                        mechanism_activated=[] if snap is None else list(snap.activated_units),
                        first_mechanism_divergence=(
                            None if snap is None else snap.first_divergence
                        ),
                        objective=objective,
                        mechanism_cosine_survival=(None if snap is None else snap.cosine_survival),
                        mechanism_energy_survival=(None if snap is None else snap.energy_survival),
                        mechanism_top_active_cosine_survival=(
                            None if snap is None else snap.top_active_cosine_survival
                        ),
                        mechanism_worst_layer_cosine_survival=(
                            None if snap is None else snap.worst_layer_cosine_survival
                        ),
                        mechanism_worst_layer=None if snap is None else snap.worst_layer,
                    )
                )

            steps = max(1, int(pgd_steps))

            for family in suite:
                if family == "deepfool":
                    adv = _deepfool(
                        self.model,
                        x_in,
                        original_pred=original_pred,
                        max_iter=int(deepfool_iters),
                        clip_min=clip_min,
                        clip_max=clip_max,
                        output_path=self.output_path,
                    )
                    _record(
                        family=family,
                        name=f"DeepFool (iters≤{int(deepfool_iters)})",
                        adv=adv,
                        epsilon=0.0,
                        steps=int(deepfool_iters),
                        norm="l2",
                        track_linf_grid=False,
                    )
                    continue

                for eps in epsilons:
                    alpha = (
                        float(pgd_step_size) if pgd_step_size is not None else float(eps) / steps
                    )
                    if family == "fgsm":
                        adv = _fgsm_step(x_in, base_grad, eps, clip_min=clip_min, clip_max=clip_max)
                        _record(
                            family=family,
                            name=f"FGSM (eps={eps:g})",
                            adv=adv,
                            epsilon=eps,
                            steps=1,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "bim":
                        adv = _iterative_linf(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            steps=steps,
                            step_size=alpha,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            initial=x_in,
                            momentum=0.0,
                        )
                        _record(
                            family=family,
                            name=f"BIM (eps={eps:g}, steps={steps})",
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "mifgsm":
                        adv = _iterative_linf(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            steps=steps,
                            step_size=alpha,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            initial=x_in,
                            momentum=1.0,
                        )
                        _record(
                            family=family,
                            name=f"MI-FGSM (eps={eps:g}, steps={steps})",
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "pgd_linf":
                        candidates = []
                        for restart in range(int(pgd_restarts)):
                            rng = torch.Generator(device="cpu")
                            rng.manual_seed(
                                int(seed) + int(round(eps * 1_000_000)) + 10_007 * restart
                            )
                            noise = (
                                torch.empty(x_in.shape, dtype=torch.float32)
                                .uniform_(-eps, eps, generator=rng)
                                .to(device=self.device)
                            )
                            candidates.append(
                                _iterative_linf(
                                    self.model,
                                    x_in,
                                    original_pred=original_pred,
                                    epsilon=eps,
                                    steps=steps,
                                    step_size=alpha,
                                    clip_min=clip_min,
                                    clip_max=clip_max,
                                    output_path=self.output_path,
                                    initial=x_in + noise,
                                )
                            )
                        adv = _strongest_candidate(candidates)
                        _record(
                            family=family,
                            name=(
                                f"PGD-L∞ (eps={eps:g}, steps={steps}, restarts={int(pgd_restarts)})"
                            ),
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "pgd_l2":
                        candidates = []
                        for restart in range(int(pgd_restarts)):
                            rng = torch.Generator(device="cpu")
                            rng.manual_seed(
                                int(seed) + 53 + int(round(eps * 1_000_000)) + 10_007 * restart
                            )
                            noise = torch.randn(
                                x_in.shape,
                                dtype=torch.float32,
                                generator=rng,
                            ).to(device=self.device)
                            flat = noise.reshape(noise.shape[0], -1)
                            scale = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(1e-12)
                            noise = noise / scale.reshape(-1, *([1] * (noise.ndim - 1)))
                            candidates.append(
                                _pgd_l2(
                                    self.model,
                                    x_in,
                                    original_pred=original_pred,
                                    epsilon=eps,
                                    steps=steps,
                                    step_size=alpha,
                                    clip_min=clip_min,
                                    clip_max=clip_max,
                                    output_path=self.output_path,
                                    initial=x_in + float(eps) * noise,
                                )
                            )
                        adv = _strongest_candidate(candidates)
                        _record(
                            family=family,
                            name=(
                                f"PGD-L2 (eps={eps:g}, steps={steps}, restarts={int(pgd_restarts)})"
                            ),
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="l2",
                            track_linf_grid=False,
                        )
                    elif family in {"apgd", "apgd_ce"}:
                        adv = _apgd_lite(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            steps=steps,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            seed=int(seed) + int(round(eps * 1_000_000)),
                            loss_kind="cross_entropy",
                        )
                        _record(
                            family="apgd_ce",
                            name=f"APGD-CE (eps={eps:g}, steps={steps})",
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "apgd_dlr":
                        adv = _apgd_lite(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            steps=steps,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            seed=int(seed) + 41 + int(round(eps * 1_000_000)),
                            loss_kind="dlr",
                        )
                        # Targeted APGD-DLR toward top alternative logits (AA-style).
                        with torch.inference_mode():
                            z0 = _normalise_logits(
                                extract_logits(self.model(adv), self.output_path)
                            )
                            still_ok = int(predicted_class(z0)[0].item()) == original_pred
                        if still_ok and z0.shape[-1] > 1:
                            order = torch.argsort(z0[0], descending=True).tolist()
                            alts = [c for c in order if c != original_pred][
                                : max(1, int(n_target_classes))
                            ]
                            for ti, t_cls in enumerate(alts):
                                cand = _apgd_lite(
                                    self.model,
                                    x_in,
                                    original_pred=original_pred,
                                    epsilon=eps,
                                    steps=max(4, steps // 2),
                                    clip_min=clip_min,
                                    clip_max=clip_max,
                                    output_path=self.output_path,
                                    seed=int(seed) + 100 * (ti + 1),
                                    loss_kind="targeted_dlr",
                                    target_class=int(t_cls),
                                    n_restarts=1,
                                )
                                with torch.inference_mode():
                                    pred_c = int(
                                        predicted_class(
                                            _normalise_logits(
                                                extract_logits(self.model(cand), self.output_path)
                                            )
                                        )[0].item()
                                    )
                                if pred_c != original_pred:
                                    adv = cand
                                    break
                        _record(
                            family=family,
                            name=f"APGD-DLR±targeted (eps={eps:g}, steps={steps})",
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "pgd_dlr":
                        rng = torch.Generator(device="cpu")
                        rng.manual_seed(int(seed) + 91 + int(round(eps * 1_000_000)))
                        noise = (
                            torch.empty(x_in.shape, dtype=torch.float32)
                            .uniform_(-eps, eps, generator=rng)
                            .to(device=self.device)
                        )
                        adv = _iterative_linf(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            steps=steps,
                            step_size=alpha,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            initial=x_in + noise,
                            loss_kind="dlr",
                        )
                        _record(
                            family=family,
                            name=f"PGD-DLR (eps={eps:g}, steps={steps})",
                            adv=adv,
                            epsilon=eps,
                            steps=steps,
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "fab":
                        adv = _fab_linf(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            n_iter=int(fab_iters),
                            n_restarts=int(fab_restarts),
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            seed=int(seed) + int(round(eps * 1_000_000)),
                        )
                        with torch.inference_mode():
                            z0 = _normalise_logits(
                                extract_logits(self.model(adv), self.output_path)
                            )
                            still_ok = int(predicted_class(z0)[0].item()) == original_pred
                        if still_ok and z0.shape[-1] > 2:
                            order = torch.argsort(z0[0], descending=True).tolist()
                            alts = [c for c in order if c != original_pred][
                                : max(1, int(n_target_classes))
                            ]
                            for ti, t_cls in enumerate(alts):
                                cand = _fab_linf(
                                    self.model,
                                    x_in,
                                    original_pred=original_pred,
                                    epsilon=eps,
                                    n_iter=max(4, int(fab_iters) // 2),
                                    n_restarts=1,
                                    clip_min=clip_min,
                                    clip_max=clip_max,
                                    output_path=self.output_path,
                                    seed=int(seed) + 200 * (ti + 1),
                                    targeted_class=int(t_cls),
                                )
                                with torch.inference_mode():
                                    pred_c = int(
                                        predicted_class(
                                            _normalise_logits(
                                                extract_logits(self.model(cand), self.output_path)
                                            )
                                        )[0].item()
                                    )
                                if pred_c != original_pred:
                                    adv = cand
                                    break
                        _record(
                            family=family,
                            name=(
                                f"FAB (eps={eps:g}, iters≤{int(fab_iters)}, "
                                f"restarts={int(fab_restarts)})"
                            ),
                            adv=adv,
                            epsilon=eps,
                            steps=int(fab_iters),
                            norm="linf",
                            track_linf_grid=True,
                        )
                    elif family == "square":
                        adv = _square_attack(
                            self.model,
                            x_in,
                            original_pred=original_pred,
                            epsilon=eps,
                            n_queries=int(square_queries),
                            clip_min=clip_min,
                            clip_max=clip_max,
                            output_path=self.output_path,
                            seed=int(seed) + int(round(eps * 1_000_000)),
                        )
                        _record(
                            family=family,
                            name=f"Square (eps={eps:g}, queries≤{int(square_queries)})",
                            adv=adv,
                            epsilon=eps,
                            steps=int(square_queries),
                            norm="linf",
                            track_linf_grid=True,
                        )

            if concept_dirs_t is not None and best_delta is not None:
                ranked = _rank_concept_directions(
                    best_delta, concept_dirs_t, concept_names=concept_names, top_k=5
                )
            else:
                ranked = _rank_input_features(base_grad, concept_names=concept_names, top_k=5)
            most_vulnerable = ranked[0].split(" (", 1)[0] if ranked else "unavailable"

            recommendations = _build_recommendations(
                epsilons=epsilons,
                attacks=attacks_out,
                min_flip_eps_linf=min_flip_eps_linf,
                most_vulnerable=most_vulnerable,
            )
            curve = survival_margin_rows(attacks_out)
            instability_metrics = (
                "mechanism_cosine_survival",
                "mechanism_energy_survival",
                "mechanism_top_active_cosine_survival",
                "mechanism_worst_layer_cosine_survival",
            )
            instability_radii = {
                metric: mechanism_instability_radius(
                    curve,
                    tau=float(instability_tau),
                    metric=metric,
                )
                for metric in instability_metrics
            }
            instab = mechanism_instability_radius(
                curve,
                tau=float(instability_tau),
                metric=str(instability_metric),
            )
            agreement = cross_attack_agreement(family_deltas)
            fail_families = cluster_failure_families(family_deltas)
            failure_spectrum = failure_mode_spectrum(family_deltas)
            adequacy = attack_adequacy_audit(attacks_out, base_grad)
            if adequacy.warnings:
                recommendations.insert(
                    0,
                    "Attack-adequacy warning: " + " ".join(adequacy.warnings),
                )
            if instab is not None:
                recommendations.insert(
                    0,
                    (
                        f"Mechanism threshold crossing for {instability_metric} "
                        f"<τ={float(instability_tau):g} "
                        f"at ε={instab:g} while the label still held "
                        "(observational pre-flip mechanism drift)."
                    ),
                )
            if fail_families:
                biggest = max(fail_families, key=lambda f: int(f.get("n_members", 0)))
                if int(biggest.get("n_members", 0)) >= 2:
                    members = ", ".join(
                        _ATTACK_LABELS.get(m, m) for m in biggest.get("members", [])
                    )
                    recommendations.insert(
                        0,
                        (
                            f"Failure family {biggest.get('family_id')}: {members} "
                            "converged on similar internal ΔM (shared weakness)."
                        ),
                    )
            recommendations.append(
                f"Most sensitive tested input coordinate: '{most_vulnerable}' "
                "(index/schema label — not proof of semantic causality)."
            )

            return AdversarialAuditReport(
                model_name=self.model_name,
                empirical_min_flip_eps_linf=min_flip_eps_linf,
                empirical_min_flip_eps_l2=min_flip_eps_l2,
                attacks_tested=attacks_out,
            most_vulnerable_concept=most_vulnerable,
                vulnerable_concepts=ranked,
                recommendations=recommendations,
                limitations=list(_LIMITATIONS),
                mechanism_instability_radius=instab,
                metadata={
                    "claim_type": (
                        "Adversarial Mechanism Audit: Layer-1 stress + Layer-2 frozen-M ΔM "
                        "(not a certificate / not AutoAttack-complete / not domain evasion)"
                    ),
                    "audit_layers": [
                        "robustness_stress",
                        "mechanism_failure_analysis",
                    ],
                    "access_model": (
                        "white-box = differentiable model artefact (input gradients); "
                        "not feature semantics / training data / owner intent. "
                        "Flat-block search = black-box query-only."
                    ),
                    "decision_margin_definition": (
                        "m = z_target − logsumexp(z_other); binary single-logit: ±z"
                    ),
                    "empirical_flip_radius_definition": (
                        "smallest successful tested L∞ budget among ε-grid attacks; "
                        "an upper bound on the minimum adversarial radius when a valid "
                        "flip is found, not an exact or certified radius"
                    ),
                    "mechanism_survival_definition": (
                        "conservative survival is the finite minimum of separately reported "
                        "global cosine, relative-energy survival, top-clean-active cosine and "
                        "eligible per-layer cosine on an exactly frozen coordinate basis"
                    ),
                    "mechanism_instability_radius_definition": (
                        f"smallest L∞ ε with {instability_metric}<"
                        f"τ={float(instability_tau):g} while prediction still holds"
                    ),
                    "mechanism_probe_kind": probe_kind,
                    "mechanism_survival_curve": curve,
                    "mechanism_instability_radius": instab,
                    "mechanism_instability_tau": float(instability_tau),
                    "mechanism_instability_metric": str(instability_metric),
                    "mechanism_instability_radii_by_metric": instability_radii,
                    "cross_attack_agreement": agreement,
                    "failure_families": fail_families,
                    "failure_mode_spectrum": failure_spectrum,
                    "attack_adequacy": adequacy.to_dict(),
                    "mechanism_probe_failures": probe_failures,
                    "mechanism_probe_coordinates": len(unit_names),
                    "mechanism_probe_basis": "selected on clean reference and frozen",
                    "attack_suite": suite,
                    "attack_references": {
                        "fgsm": "Goodfellow et al. 2015",
                        "bim": "Kurakin et al. 2017",
                        "mifgsm": "Dong et al. 2018",
                        "pgd_linf": "Madry et al. 2018",
                        "pgd_l2": "Madry et al. 2018 (L2 ball)",
                        "apgd_ce": "Croce & Hein 2020 APGD-CE objective (VRAVEN-lite optimiser)",
                        "apgd_dlr": "Croce & Hein 2020 DLR + targeted DLR objectives (VRAVEN-lite optimiser)",
                        "fab": "Croce & Hein 2019 FAB (lite)",
                        "pgd_dlr": "Carlini & Wagner / DLR-style logit PGD",
                        "deepfool": "Moosavi-Dezfooli et al. 2016",
                        "square": "Andriushchenko et al. 2020 (flat-safe patch variant)",
                        "autoattack": "Croce & Hein 2020 AutoAttack ensemble (lite)",
                    },
                    "epsilon_grid": epsilons,
                    "pgd_steps": steps,
                    "pgd_restarts": int(pgd_restarts),
                    "decision_target": decision_target,
                    "original_prediction": original_pred,
                    "original_margin": original_margin,
                    "seed": int(seed),
                    "device": str(self.device),
                    "dtype": str(self.dtype),
                },
            )
        finally:
            self.model.train(was_training)

    def audit_autoattack(
        self,
        x: Tensor,
        *,
        epsilon: float = 0.1,
        steps: int = 20,
        clip_min: float | None = None,
        clip_max: float | None = None,
        seed: int = 24032026,
        square_queries: int = 200,
        fab_iters: int = 40,
        fab_restarts: int = 2,
        n_target_classes: int = 3,
        target: int | None = None,
    ) -> AdversarialAuditReport:
        """Run the fixed-ε VRAVEN ensemble-lite pilot protocol.

        The method name is retained for API compatibility.  The returned
        protocol is deliberately not labelled AutoAttack because the internal
        implementations and budgets do not claim parity with the reference
        implementation.
        """

        report = self.audit_sample(
            x,
            target=target,
            epsilon_grid=(float(epsilon),),
            pgd_steps=int(steps),
            clip_min=clip_min,
            clip_max=clip_max,
            seed=seed,
            attacks=AUTOATTACK_SUITE,
            square_queries=int(square_queries),
            fab_iters=int(fab_iters),
            fab_restarts=int(fab_restarts),
            n_target_classes=int(n_target_classes),
        )
        # Sequential AA bookkeeping: first family that flips "wins".
        flipped_by = None
        for fam in AUTOATTACK_SUITE:
            hit = next(
                (a for a in report.attacks_tested if a.attack_family == fam and a.succeeded),
                None,
            )
            if hit is not None:
                flipped_by = fam
                break
        report.metadata["protocol"] = "vraven_ensemble_lite"
        report.metadata["claim_type"] = (
            "fixed-ε VRAVEN ensemble-lite pilot on one sample; "
            "not reference AutoAttack and not a robustness certificate"
        )
        report.metadata["autoattack_epsilon"] = float(epsilon)
        report.metadata["autoattack_flipped_by"] = flipped_by
        report.metadata["autoattack_robust"] = flipped_by is None
        return report

    def audit_batch(
        self,
        x: Tensor,
        y: Tensor | np.ndarray | Sequence[int] | None = None,
        *,
        epsilon: float = 0.1,
        protocol: str = "vraven_ensemble_lite",
        max_samples: int | None = None,
        sampling: str = "stratified",
        sample_indices: Sequence[int] | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        seed: int = 24032026,
        steps: int = 10,
        pgd_restarts: int = 1,
        square_queries: int = 80,
        fab_iters: int = 15,
        fab_restarts: int = 1,
        n_target_classes: int = 2,
    ) -> RobustAccuracyReport:
        """Estimate overall and clean-conditional robustness at fixed L∞ ε.

        Labelled clean errors count as non-robust in ``robust_accuracy``.
        ``conditional_robust_accuracy`` retains the historically reported
        clean-correct denominator.  When labels are unavailable the result is
        explicitly prediction invariance, not accuracy.
        """

        epsilon = float(epsilon)
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be non-negative and finite")
        x_all = _ensure_batch(x.detach()).to(device=self.device, dtype=self.dtype)
        total_available = int(x_all.shape[0])
        if total_available <= 0:
            raise ValueError("audit_batch requires at least one sample")

        all_labels: list[int] | None
        if y is None:
            all_labels = None
        elif isinstance(y, np.ndarray):
            all_labels = [int(v) for v in np.asarray(y).reshape(-1)]
        elif isinstance(y, Tensor):
            all_labels = [int(v) for v in y.detach().reshape(-1).tolist()]
        else:
            all_labels = [int(v) for v in list(y)]
        if all_labels is not None and len(all_labels) != total_available:
            raise ValueError("y must contain exactly one label per input sample")

        limit = total_available if max_samples is None else min(int(max_samples), total_available)
        if limit <= 0:
            raise ValueError("max_samples must be positive when supplied")
        if sample_indices is not None:
            selected = [int(index) for index in sample_indices]
            if len(selected) > limit:
                selected = selected[:limit]
            if not selected or len(set(selected)) != len(selected):
                raise ValueError("sample_indices must be non-empty and unique")
            if min(selected) < 0 or max(selected) >= total_available:
                raise ValueError("sample_indices contains an out-of-range index")
            sampling_name = "explicit_indices"
        else:
            sampling_name = str(sampling).lower().strip()
            if sampling_name == "stratified":
                if all_labels is None:
                    raise ValueError("stratified sampling requires labels")
                rng = np.random.default_rng(int(seed))
                pools: dict[int, list[int]] = {}
                for index, label in enumerate(all_labels):
                    pools.setdefault(label, []).append(index)
                for pool in pools.values():
                    rng.shuffle(pool)
                selected = []
                class_order = sorted(pools)
                while len(selected) < limit:
                    added = False
                    for label in class_order:
                        if pools[label] and len(selected) < limit:
                            selected.append(pools[label].pop())
                            added = True
                    if not added:
                        break
            elif sampling_name == "head":
                selected = list(range(limit))
            else:
                raise ValueError("sampling must be 'stratified' or 'head'")

        n = len(selected)
        x_b = x_all[selected]
        labels: list[int | None] = (
            [None] * n if all_labels is None else [all_labels[index] for index in selected]
        )

        proto = str(protocol).lower().strip()
        if proto in {"autoattack", "aa", "autoattack_lite", "vraven_ensemble_lite"}:
            attack_list = AUTOATTACK_SUITE
            proto_name = "vraven_ensemble_lite"
        elif proto in {"pgd", "pgd_linf"}:
            attack_list = ("pgd_linf",)
            proto_name = "pgd_linf"
        else:
            raise ValueError("protocol must be 'vraven_ensemble_lite' or 'pgd'")

        n_correct = 0
        n_robust = 0
        success_sets: list[set[str]] = []
        clean_prediction_values: list[int] = []
        class_counts: dict[int, dict[str, int]] = {}
        batch_attack_diagnostics: dict[str, Any] = {"vectorised": False}
        was_training = self.model.training
        self.model.eval()
        try:
            if clip_min is not None and bool((x_b < float(clip_min) - 1e-7).any()):
                raise ValueError("clean input lies below clip_min")
            if clip_max is not None and bool((x_b > float(clip_max) + 1e-7).any()):
                raise ValueError("clean input lies above clip_max")

            if proto_name == "pgd_linf":
                with torch.inference_mode():
                    clean_logits = _normalise_batched_logits(
                        extract_logits(self.model(x_b), self.output_path), n
                    )
                    clean_predictions = predicted_class(clean_logits)
                clean_prediction_values = [
                    int(value) for value in clean_predictions.detach().cpu().tolist()
                ]
                target_tensor = (
                    clean_predictions.detach().clone()
                    if all_labels is None
                    else torch.tensor(all_labels, device=self.device, dtype=torch.long)[selected]
                )
                clean_correct = (
                    torch.ones(n, device=self.device, dtype=torch.bool)
                    if all_labels is None
                    else clean_predictions == target_tensor
                )
                n_correct = int(clean_correct.sum().item())
                attacked = torch.nonzero(clean_correct, as_tuple=False).flatten()
                flipped = torch.zeros(n, device=self.device, dtype=torch.bool)
                if attacked.numel():
                    adv, batch_attack_diagnostics = _batch_pgd_linf(
                        self.model,
                        x_b[attacked],
                        target_tensor[attacked],
                        epsilon=float(epsilon),
                        steps=int(steps),
                        restarts=int(pgd_restarts),
                        clip_min=clip_min,
                        clip_max=clip_max,
                        output_path=self.output_path,
                        seed=int(seed),
                    )
                    with torch.inference_mode():
                        adv_predictions = predicted_class(
                            _normalise_batched_logits(
                                extract_logits(self.model(adv), self.output_path),
                                int(attacked.numel()),
                            )
                        )
                    flipped[attacked] = adv_predictions != clean_predictions[attacked]
                n_robust = int((clean_correct & ~flipped).sum().item())
                success_sets = [
                    {"pgd_linf"} if bool(flipped[i].item()) else set() for i in range(n)
                ]
                if all_labels is not None:
                    for i, label in enumerate(labels):
                        assert label is not None
                        counts = class_counts.setdefault(
                            int(label), {"n": 0, "clean": 0, "robust": 0}
                        )
                        counts["n"] += 1
                        if bool(clean_correct[i].item()):
                            counts["clean"] += 1
                            if not bool(flipped[i].item()):
                                counts["robust"] += 1
            else:
                for i in range(n):
                    xi = x_b[i : i + 1]
                    with torch.inference_mode():
                        logits = _normalise_logits(extract_logits(self.model(xi), self.output_path))
                        pred = int(predicted_class(logits)[0].item())
                    clean_prediction_values.append(pred)
                    label = labels[i]
                    clean_ok = pred == int(label) if label is not None else True
                    if label is not None:
                        class_counts.setdefault(int(label), {"n": 0, "clean": 0, "robust": 0})
                        class_counts[int(label)]["n"] += 1
                    if label is not None and not clean_ok:
                        success_sets.append(set())
                        continue
                    n_correct += 1
                    if label is not None:
                        class_counts[int(label)]["clean"] += 1

                    sample_report = self.audit_autoattack(
                        xi,
                        epsilon=float(epsilon),
                        steps=int(steps),
                        clip_min=clip_min,
                        clip_max=clip_max,
                        seed=int(seed) + i,
                        square_queries=int(square_queries),
                        fab_iters=int(fab_iters),
                        fab_restarts=int(fab_restarts),
                        n_target_classes=int(n_target_classes),
                        target=int(label) if label is not None else pred,
                    )
                    successful_families = {
                        a.attack_family for a in sample_report.attacks_tested if a.succeeded
                    }
                    success_sets.append(successful_families)
                    sample_flipped = bool(successful_families)
                    if not sample_flipped:
                        n_robust += 1
                        if label is not None:
                            class_counts[int(label)]["robust"] += 1
        finally:
            self.model.train(was_training)

        clean_acc = float(n_correct / n) if n else 0.0
        conditional = float(n_robust / n_correct) if n_correct else 0.0
        robust_acc = float(n_robust / n) if all_labels is not None else conditional
        prediction_invariance = conditional if all_labels is None else None
        asr = float((n_correct - n_robust) / n_correct) if n_correct else 0.0
        clean_interval = wilson_score_interval(n_correct, n)
        robust_interval = wilson_score_interval(n_robust, n)
        conditional_interval = (
            wilson_score_interval(n_robust, n_correct) if n_correct else (0.0, 1.0)
        )

        per_attack_successes = {
            family: sum(family in successes for successes in success_sets) for family in attack_list
        }
        per_attack_unique = {
            family: sum(successes == {family} for successes in success_sets)
            for family in attack_list
        }
        cumulative: set[int] = set()
        union_curve: list[dict[str, Any]] = []
        per_attack_marginal: dict[str, int] = {}
        for family in attack_list:
            hit = {index for index, successes in enumerate(success_sets) if family in successes}
            new = hit - cumulative
            cumulative.update(hit)
            per_attack_marginal[family] = len(new)
            union_curve.append(
                {
                    "attack": family,
                    "raw_successes": len(hit),
                    "unique_successes": per_attack_unique[family],
                    "marginal_new_successes": len(new),
                    "cumulative_successes": len(cumulative),
                }
            )

        per_class: dict[str, dict[str, Any]] = {}
        for label, counts in sorted(class_counts.items()):
            class_n = counts["n"]
            class_clean = counts["clean"]
            class_robust = counts["robust"]
            per_class[str(label)] = {
                **counts,
                "clean_accuracy": class_clean / class_n,
                "robust_accuracy": class_robust / class_n,
                "conditional_robust_accuracy": (class_robust / class_clean if class_clean else 0.0),
                "robust_accuracy_wilson_95": list(wilson_score_interval(class_robust, class_n)),
            }
        macro_robust = (
            float(np.mean([row["robust_accuracy"] for row in per_class.values()]))
            if per_class
            else None
        )
        per_sample_outcomes = []
        for position, (selected_index, successful) in enumerate(
            zip(selected, success_sets, strict=True)
        ):
            label = labels[position]
            prediction = clean_prediction_values[position]
            clean_ok = label is None or prediction == int(label)
            per_sample_outcomes.append(
                {
                    "cohort_position": position,
                    "selected_index": int(selected_index),
                    "label": None if label is None else int(label),
                    "clean_prediction": int(prediction),
                    "clean_correct": bool(clean_ok),
                    "attack_succeeded": bool(successful),
                    "robust": bool(clean_ok and not successful),
                    "successful_attacks": sorted(successful),
                }
            )
        pilot = n < 100
        return RobustAccuracyReport(
            model_name=self.model_name,
            protocol=proto_name,
            epsilon=float(epsilon),
            n_samples=n,
            n_correct_clean=n_correct,
            n_robust=n_robust,
            clean_accuracy=clean_acc,
            robust_accuracy=robust_acc,
            attack_success_rate=asr,
            per_attack_flips=per_attack_marginal,
            conditional_robust_accuracy=conditional,
            prediction_invariance=prediction_invariance,
            per_attack_successes=per_attack_successes,
            per_attack_unique_flips=per_attack_unique,
            attack_union_curve=union_curve,
            selected_indices=selected,
            per_class=per_class,
            macro_robust_accuracy=macro_robust,
            per_sample_outcomes=per_sample_outcomes,
            sampling_protocol=sampling_name,
            metadata={
                "threat_model": f"L∞ ε={epsilon:g}",
                "attack_suite": list(attack_list),
                "pgd_steps": int(steps),
                "pgd_restarts": int(pgd_restarts),
                "attack_diagnostics": batch_attack_diagnostics,
                "robust_accuracy_wilson_95": list(robust_interval),
                "clean_accuracy_wilson_95": list(clean_interval),
                "conditional_robust_accuracy_wilson_95": list(conditional_interval),
                "interval_definition": (
                    "pointwise two-sided nominal 95% Wilson intervals; these quantify "
                    "sampling uncertainty, not missed-attack uncertainty"
                ),
                "label_source": "ground_truth" if all_labels is not None else "clean_prediction",
                "selected_class_counts": {label: row["n"] for label, row in per_class.items()},
                "evidence_scope": "pilot" if pilot else "declared_cohort",
                "note": (
                    "Attack-based robust accuracy is an empirical upper estimate of "
                    "worst-case robust accuracy because an optimiser can miss adversaries."
                ),
            },
        )

    def audit_robustness_curve(
        self,
        x: Tensor,
        y: Tensor | np.ndarray | Sequence[int],
        *,
        epsilon_grid: Sequence[float] = (0.01, 0.05, 0.1, 0.25),
        protocol: str = "vraven_ensemble_lite",
        max_samples: int | None = None,
        sampling: str = "stratified",
        clip_min: float | None = None,
        clip_max: float | None = None,
        seed: int = 24032026,
        **attack_budgets: Any,
    ) -> RobustnessProfileReport:
        """Evaluate a matched cohort across ε and build a monotone RA profile."""

        epsilons = sorted({float(epsilon) for epsilon in epsilon_grid})
        if not epsilons:
            raise ValueError("epsilon_grid must contain at least one value")
        reports = [
            self.audit_batch(
                x,
                y,
                epsilon=epsilon,
                protocol=protocol,
                max_samples=max_samples,
                sampling=sampling,
                clip_min=clip_min,
                clip_max=clip_max,
                seed=seed,
                **attack_budgets,
            )
            for epsilon in epsilons
        ]
        return build_robustness_profile(reports)


def _build_recommendations(
    *,
    epsilons: Sequence[float],
    attacks: Sequence[AttackResult],
    min_flip_eps_linf: float | None,
    most_vulnerable: str,
) -> list[str]:
    recs: list[str] = []
    families = sorted({a.attack_family for a in attacks})
    for fam in families:
        label = _ATTACK_LABELS.get(fam, fam)
        flip = next((a for a in attacks if a.attack_family == fam and a.succeeded), None)
        if flip is not None:
            if fam == "deepfool":
                recs.append(
                    f"{label} flipped with ‖δ‖₂={flip.perturbation_norm_l2:.4g}, "
                    f"‖δ‖∞={flip.perturbation_norm_linf:.4g}."
                )
            else:
                recs.append(f"{label} flipped at ε={flip.epsilon:g} ({flip.norm}).")
        else:
            if fam == "deepfool":
                recs.append(f"{label} did not flip within the iteration budget.")
            else:
                recs.append(
                    f"{label} did not flip within the tested grid (max ε={max(epsilons):g})."
                )
    if min_flip_eps_linf is None:
        recs.append(
            f"No adversarial prediction flip was found through ε={max(epsilons):g} on this "
            "sample under the executed finite attack budgets. This is attack non-discovery, "
            "not a robustness lower bound."
        )
    else:
        recs.append(
            f"Smallest successful tested L∞ budget on this sample: ε={min_flip_eps_linf:g}. "
            "This supplies an empirical upper bound on the true robust radius, not a certificate."
        )
    _ = most_vulnerable  # displayed with precise wording by the caller
    return recs
